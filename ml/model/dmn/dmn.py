# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import time
import sys

from .util import *
from ...module import common
from ...module.node import DataPackage
from ...module.rnn_node import RNNNode
from ... import config_info as ci

class DynamicMemoryNetwork:
    def _optimizer_getter(lr):
        return tf.train.GradientDescentOptimizer(lr)

    def __init__(self, running_type, data_src_dict
               , vocabulary_size_list, embedding_size_list
               , embedding_input_idx_list
               , embedding_question_idx_list, consider_input_context
               , sentence_end_token_idx, need_end_of_passes
               , target_op_list
               , init_scale
               , num_units, keep_prob, num_layers
               , attention_hidden_size, use_softmax_attn, max_episodic_num
               , learning_rate, l1_reg=-1, l2_reg=-1, max_grad_value=-1, max_grad_norm=10
               , dynamic=True
               , optimizer_getter=_optimizer_getter):
        self.num_units=num_units
        self.dynamic=dynamic
        self.num_layers=num_layers

        self.summary=None

        regularizer=None
        if running_type==ci.RunningType.train:
            if l1_reg>0 and l2_reg>0:
                regularizer=tf.contrib.layers.l1_l2_regularizer(l1_reg, l2_reg)
            elif l1_reg>0:
                regularizer=tf.contrib.layers.l1_regularizer(l1_reg)
            elif l2_reg>0:
                regularizer=tf.contrib.layers.l2_regularizer(l2_reg)

        self.data_src_dict=data_src_dict

        if 'input_none_embedding' in data_src_dict:
            self.none_embedding_input=data_src_dict['input_none_embedding']
        else:
            self.none_embedding_input=None

        self.embedding_inputs=None
        for i in range(len(data_src_dict)):
            name='input_embedding_{}'.format(i)
            if name in data_src_dict.keys():
                if self.embedding_inputs is None:
                    self.embedding_inputs=[]
                self.embedding_inputs.append(data_src_dict[name])
            else:
                break

        if self.dynamic and 'input_sequence_length' in data_src_dict.keys():
            self.input_sequence_length=data_src_dict['input_sequence_length']
        else:
            self.input_sequence_length=None

        self.embeddings=[]
        with tf.variable_scope('embedding'):
            for idx, (v_size, e_size) in enumerate(zip(vocabulary_size_list, embedding_size_list)):
                self.embeddings.append(tf.get_variable('eb_{}'.format(idx), [v_size, e_size], initializer=tf.truncated_normal_initializer(stddev=init_scale/e_size/e_size)))
            if need_end_of_passes:
                self.end_of_passes_embedding=tf.get_variable('eb_end_of_passes', [1, self.num_units], initializer=tf.truncated_normal_initializer(stddev=init_scale/self.num_units/self.num_units))

        self.input=DataPackage()
        if self.none_embedding_input is not None and self.embedding_inputs is not None:
            self.input.input_data=tf.concat([self.none_embedding_input]+[tf.nn.embedding_lookup(self.embeddings[eb_input_idx], input_to_be_embedded) for eb_input_idx, input_to_be_embedded in zip(embedding_input_idx_list, self.embedding_inputs)], axis=2)
        elif self.none_embedding_input is not None:
            self.input.input_data=self.none_embedding_input
        elif self.embedding_inputs is not None:
            tf.concat([tf.nn.embedding_lookup(self.embeddings[eb_input_idx], input_to_be_embedded) for eb_input_idx, input_to_be_embedded in zip(embedding_input_idx_list, self.embedding_inputs)], axis=2)
        else:
            raise ValueError('No input detected!')
        if 'input_initial_states' in data_src_dict.keys():
            self.input.initial_states=data_src_dict['input_initial_states']
        else:
            self.input.initial_states=None
        self.input.sequence_length=self.input_sequence_length

        self.has_question='question_none_embedding' in data_src_dict or (embedding_question_idx_list is not None and len(embedding_question_idx_list)>0)

        def rnn_cell_generator(layer_idx):
            return tf.nn.rnn_cell.GRUCell(self.num_units)

        def episodic_rnn_cell_generator(layer_idx):
            return AttentionGRUWrapper(self.num_units, self.num_units)

        scale=init_scale/self.input.input_data.shape[-1].value
        with tf.variable_scope('input_module', initializer=tf.truncated_normal_initializer(stddev=scale)):
            self.input_module=RNNNode(self.input, running_type, rnn_cell_generator, num_layers=self.num_layers
                                    , input_keep_prob=keep_prob, output_keep_prob=keep_prob, state_keep_prob=keep_prob, variational_recurrent=True)

            if sentence_end_token_idx is not None and sentence_end_token_idx>=0:
                sentence_endings_mask=tf.where(tf.equal(self.embedding_inputs[0], sentence_end_token_idx) if self.input_sequence_length is None else tf.logical_and(tf.equal(self.embedding_inputs[0], sentence_end_token_idx), tf.sequence_mask(self.input_sequence_length, maxlen=tf.reduce_sum(tf.ones_like(self.input.input_data[0,:,0])))), tf.ones_like(self.embedding_inputs[0], dtype=tf.bool), tf.zeros_like(self.embedding_inputs[0], dtype=tf.bool))
                self.mid_sequence_len=tf.reduce_sum(tf.cast(sentence_endings_mask, tf.int32), axis=1)
                max_mid_seq_len=tf.reduce_max(self.mid_sequence_len)
                self.dynamic=True
                def gather_sen_end(packed):
                    the_input, seq_len, mask=packed
                    return tf.pad(tf.reshape(tf.gather_nd(the_input, tf.where(mask)), [-1, self.num_units]), paddings=[[0, max_mid_seq_len-seq_len], [0, 0]], mode='CONSTANT'), seq_len, mask
                self.input_module.final_output, _, _=tf.map_fn(gather_sen_end, (self.input_module.output.outputs, self.mid_sequence_len, tf.tile(tf.expand_dims(sentence_endings_mask, axis=2), multiples=[1,1,self.num_units])))
            else:
                self.mid_sequence_len=self.input_sequence_length
                self.input_module.final_output=self.input_module.output.outputs

            if need_end_of_passes:
                if self.mid_sequence_len is not None:
                    def insert_end_of_passes(packed):
                        the_input, seq_len=packed
                        input1, input2=tf.split(the_input, num_or_size_splits=[seq_len, -1], axis=0)
                        the_input=tf.concat([input1, tf.nn.embedding_lookup(self.end_of_passes_embedding, [0]), input2], axis=0)
                        return the_input, seq_len
                    self.input_module.final_output, _=tf.map_fn(insert_end_of_passes, (self.input_module.final_output, self.mid_sequence_len))
                    self.mid_sequence_len=tf.where(tf.greater(self.mid_sequence_len, 0), self.mid_sequence_len+1, self.mid_sequence_len)
                else:
                    end_of_passes_embedded=tf.expand_dims(tf.ones_like(self.input_module.final_output[:,0,:])*tf.nn.embedding_lookup(self.end_of_passes_embedding, [0]), axis=1)
                    self.input_module.final_output=tf.concat([self.input_module.final_output, end_of_passes_embedded], axis=1)
                    if self.mid_sequence_len is not None:
                        self.mid_sequence_len=tf.where(tf.greater(self.mid_sequence_len, 0), self.mid_sequence_len+1, self.mid_sequence_len)

        self.question_module=None
        if self.has_question:
            if 'question_none_embedding' in data_src_dict.keys():
                self.none_embedding_question=data_src_dict['question_none_embedding']
            else:
                self.none_embedding_question=None

            self.embedding_questions=None
            for i in range(len(data_src_dict)):
                name='question_embedding_{}'.format(i)
                if name in data_src_dict.keys():
                    if self.embedding_questions is None:
                        self.embedding_questions=[]
                    self.embedding_questions.append(data_src_dict[name])
                else:
                    break

            if self.dynamic and 'question_sequence_length' in data_src_dict.keys():
                self.question_sequence_length=data_src_dict['question_sequence_length']
            else:
                self.question_sequence_length=None

            self.question=DataPackage()
            if self.none_embedding_question is not None and self.embedding_questions is not None:
                self.question.input_data=tf.concat([self.none_embedding_question]+[tf.nn.embedding_lookup(self.embeddings[eb_question_idx], question_to_be_embedded) for eb_question_idx, question_to_be_embedded in zip(embedding_question_idx_list, self.embedding_questions)], axis=2)
            elif self.none_embedding_question is not None:
                self.question.input_data=self.none_embedding_question
            elif self.embedding_questions is not None:
                self.question.input_data=tf.concat([tf.nn.embedding_lookup(self.embeddings[eb_question_idx], question_to_be_embedded) for eb_question_idx, question_to_be_embedded in zip(embedding_question_idx_list, self.embedding_questions)], axis=2)
            if 'question_initial_states' in data_src_dict.keys():
                self.question.initial_states=data_src_dict['question_initial_states']
            else:
                self.question.initial_states=self.input_module.output.final_states if consider_input_context else None
            self.question.sequence_length=self.question_sequence_length

            scale=init_scale/self.question.input_data.shape[-1].value
            with tf.variable_scope('question_module', initializer=tf.truncated_normal_initializer(stddev=scale)):
                self.question_module=RNNNode(self.question, running_type, rnn_cell_generator, num_layers=self.num_layers
                                           , input_keep_prob=keep_prob, output_keep_prob=keep_prob, state_keep_prob=keep_prob, variational_recurrent=True)

        scale=init_scale/self.input_module.final_output.shape[-1].value
        with tf.variable_scope('memory_module', initializer=tf.truncated_normal_initializer(stddev=scale)):
            episodic=DataPackage()
            #episodic.input_data=self.input_module.final_output
            episodic.initial_states=None
            episodic.sequence_length=self.mid_sequence_len

            with tf.variable_scope('attention'):
                self.attention=Attention(self.num_layers, attention_hidden_size, init_scale, regularizer, need_end_of_passes, self.mid_sequence_len, use_softmax=use_softmax_attn)

            with tf.variable_scope('episodic'):
                if self.question_module is None:
                    if self.num_layers>1:
                        m=[tf.zeros_like(x) for x in self.input_module.output.final_states]
                    else:
                        m=tf.zeros_like(self.input_module.output.final_states)
                    q=None
                else:
                    m=self.question_module.output.final_states
                    q=m

                def cond(i, m, is_end_of_passes):
                    return tf.logical_and(i<max_episodic_num, tf.logical_or(not need_end_of_passes, tf.logical_not(tf.reduce_all(is_end_of_passes))))

                def body(i, m, is_end_of_passes):
                    curr_attn, is_end_of_passes_new=self.attention.cal_attention(self.input_module.final_output, m, q)
                    is_end_of_passes=tf.logical_or(is_end_of_passes, is_end_of_passes_new) if need_end_of_passes else is_end_of_passes
                    episodic.input_data=tf.concat([self.input_module.final_output, curr_attn], axis=-1)
                    episodic_module=RNNNode(episodic, running_type, episodic_rnn_cell_generator, num_layers=self.num_layers
                                          , input_keep_prob=keep_prob, output_keep_prob=keep_prob, state_keep_prob=keep_prob, variational_recurrent=True)
                    with tf.variable_scope('memory'):
                        if self.num_layers>1:
                            mem_cell=tf.nn.rnn_cell.MultiRNNCell([rnn_cell_generator() for _ in range(self.num_layers)])
                        else:
                            mem_cell=rnn_cell_generator()
                        _, mm=mem_cell(episodic_module.output.final_states[-1] if self.num_layers>1 else episodic_module.output.final_states, m)
                        if need_end_of_passes:
                            if self.num_layers>1:
                                for i in range(self.num_layers):
                                    m[i]=tf.where(is_end_of_passes, m[i], mm[i])
                            else:
                                m=tf.where(is_end_of_passes, m, mm)
                        else:
                            m=mm
                    return i+1, m, is_end_of_passes

                self.episodic_count, self.memory, self.end_of_passes_flag=tf.while_loop(cond
                                                                                      , body
                                                                                      , loop_vars=(tf.constant(0, dtype=tf.int32), m, tf.zeros_like(self.input_module.final_output[:,0,0], dtype=tf.bool)))
                if self.num_layers>1:
                    self.memory=tf.concat(self.memory, axis=-1)

        self.mid_result_list=[]
        self.result_list=[]

        scale=init_scale/self.memory.shape[-1].value
        with tf.variable_scope('answer_module', initializer=tf.truncated_normal_initializer(stddev=scale), regularizer=regularizer):
            for idx, target_op in enumerate(target_op_list):
                with tf.variable_scope('answer_{}'.format(idx)):
                    mid_result=target_op.build_mid(self.memory)
                    result=target_op.build_inference(mid_result)
                    self.mid_result_list.append(mid_result)
                    self.result_list.append(result)

        if running_type==ci.RunningType.train or running_type==ci.RunningType.valid or running_type==ci.RunningType.test:
            self.targets=[]
            for i in range(len(data_src_dict)):
                name='target_{}'.format(i)
                if name in data_src_dict.keys():
                    self.targets.append(data_src_dict[name])
                else:
                    break

            losses=[]
            with tf.variable_scope('answer_module', initializer=tf.truncated_normal_initializer(stddev=scale), regularizer=regularizer):
                for idx, (target_op, target, mid_result) in enumerate(zip(target_op_list, self.targets, self.mid_result_list)):
                    with tf.variable_scope('answer_training_{}'.format(idx)):
                        losses.append(target_op.build_training((target, mid_result)))
            self.loss=tf.concat(losses, axis=-1, name='loss')
            self.batch_size=tf.reduce_sum(tf.ones_like(self.input.input_data[:,0,0]))
            self.cost=tf.truediv(tf.reduce_sum(self.loss), self.batch_size*len(target_op_list), name='cost')

            if running_type==ci.RunningType.train:
                self.lr, self.optimizer, self.grads, self.train_op, self.global_step=common.add_train_op(self.cost, learning_rate, optimizer_getter, max_grad_value, max_grad_norm)

    def assign_lr(self, session, lr_value):
        session.run(tf.assign(self.lr, lr_value))

    def set_summary(self, session, log_dir, verbose):
        self.summary=tf.summary.merge_all()
        self.summary_writer=tf.summary.FileWriter(log_dir, session.graph)
        self.summary_verbose=verbose
        self.total_step=0

    def run_epoch(self, session, need_train, verbose=-1):
        start_time=time.time()
        costs=0.0
        batch_size=0
        iters=0
        try:
            while True:
                if self.summary is not None:
                    if self.total_step%self.summary_verbose==0:
                        summary_str=session.run(self.summary)
                        self.summary_writer.add_summary(summary_str, self.total_step)
                        sys.stderr.write('summary wrote at total_step={}\n'.format(self.total_step))
                    self.total_step+=1

                res=session.run([self.cost, self.batch_size, self.train_op] if need_train else [self.cost, self.batch_size])
                cost=res[0]
                batch_size+=res[1]
                costs+=cost
                iters+=1

                if verbose>=0 and iters%verbose==0:
                    sys.stderr.write('step {0} avg perplexity: {1:.3f} current perplexity: {2:.3f} speed: {3:.0f} bps\n'.format(iters, np.exp(costs/iters), np.exp(cost), batch_size/max(time.time()-start_time, 1)))
        except tf.errors.OutOfRangeError:
            pass

        return np.exp(costs/(iters if iters!=0 else 1))

    def run_training(self, session, verbose=-1):
        return self.run_epoch(session, True, verbose)

    def run_predicting(self, session):
        start_time=time.time()
        iters=0
        try:
            while True:
                result=session.run([self.result_list, self.data_src_dict])
                yield result
                sys.stderr.write('step {0} finished. time: {1:.3f} sec\n'.format(iters, time.time()-start_time))
                iters+=1
        except tf.errors.OutOfRangeError:
            pass

    def run_epoch_ph(self, session, data_iterator, need_train, verbose=-1):
        start_time=time.time()
        costs=0.0
        batch_size=0
        iters=0
        for step, feed_dict in enumerate(data_iterator):
            if self.summary is not None:
                if self.total_step%self.summary_verbose==0:
                    summary_str=session.run(self.summary, feed_dict=feed_dict)
                    self.summary_writer.add_summary(summary_str, self.total_step)
                    sys.stderr.write('summary wrote at total_step={}\n'.format(self.total_step))
                self.total_step+=1

            res=session.run([self.cost, self.batch_size, self.train_op] if need_train else [self.cost, self.batch_size], feed_dict=feed_dict)
            cost=res[0]
            batch_size+=res[1]
            costs+=cost
            iters+=1

            if verbose>=0 and step%verbose==0:
                sys.stderr.write('step {0} avg perplexity: {1:.3f} current perplexity: {2:.3f} speed: {3:.0f} bps\n'.format(step, np.exp(costs/iters), np.exp(cost), batch_size/max(time.time()-start_time, 1)))

        return np.exp(costs/(iters if iters!=0 else 1))

    def run_training_ph(self, session, data_iterator, verbose=-1):
        return self.run_epoch_ph(session, data_iterator, True, verbose)

    def run_predicting_ph(self, session, data_iterator):
        start_time=time.time()
        for step, feed_dict in enumerate(data_iterator):
            result=session.run([self.result_list], feed_dict=feed_dict)
            yield result
            sys.stderr.write('step {0} finished. time: {1:.3f} sec\n'.format(step, time.time()-start_time))
