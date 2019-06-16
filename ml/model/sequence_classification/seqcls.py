# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import time
import sys

from ...module.node import DataPackage
from ...module.rnn_node import RNNNode
from ... import config_info as ci

class SeqClassifier:
    def _optimizer_getter(lr):
        return tf.train.GradientDescentOptimizer(lr)

    def __init__(self, running_type, batch_size, data_src_dict, last_rnn_output_only, top_k
               , vocabulary_size, embedding_size
               , class_num, one_hot_label
               , init_scale
               , num_units, keep_prob, num_layers, output_hidden_units
               , learning_rate, l1_reg=-1, l2_reg=-1, max_grad_value=10000.0, max_grad_norm=10
               , dynamic=True
               , rnn_cell_generator=None
               , training_last_weight_only=False
               , optimizer_getter=_optimizer_getter):
        self.batch_size=batch_size
        self.num_units=num_units
        self.dynamic=dynamic

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

        if top_k is not None :
            self.top_k=tf.constant(top_k, dtype='int32')
        elif 'top_k' in self.data_src_dict:
            self.top_k=tf.reduce_max(self.data_src_dict['top_k'], axis=0)
        else:
            self.top_k=None

        self.embedding=tf.get_variable('embedding', [vocabulary_size, embedding_size])

        self.input=DataPackage()
        self.input.input_data=tf.squeeze(tf.nn.embedding_lookup(self.embedding, self.data_src_dict['input']), axis=2)
        self.input.initial_states=self.data_src_dict['states'] if 'states' in self.data_src_dict else None
        self.input.sequence_length=self.data_src_dict['sequence_length']

        def _rnn_cell_generator(layer_idx):
            return tf.nn.rnn_cell.LSTMCell(self.num_units)

        self.rnn_cell_generator=rnn_cell_generator if rnn_cell_generator is not None else _rnn_cell_generator

        scale=init_scale/embedding_size
        with tf.variable_scope('rnn_module', initializer=tf.random_uniform_initializer(-scale, scale)):
            self.rnn_module=RNNNode(self.input, running_type, self.rnn_cell_generator, num_layers=num_layers
                                  , input_keep_prob=keep_prob, output_keep_prob=keep_prob, state_keep_prob=keep_prob, variational_recurrent=True)

        if self.dynamic:
            unstacked_sequence_length=tf.unstack(self.input.sequence_length, num=self.batch_size, axis=0)

        scale=init_scale/self.rnn_module.output.outputs.shape[-1].value
        with tf.variable_scope('hidden_layer'):
            if last_rnn_output_only:
                if self.dynamic:
                    outputs=tf.concat([tf.reshape(one_out[one_sl-1], [1, self.num_units]) for one_sl, one_out in zip(unstacked_sequence_length, tf.unstack(self.rnn_module.output.outputs, num=self.batch_size, axis=0))], axis=0)
                else:
                    outputs=self.rnn_module.output.outputs[:,-1,:]
            else:
                outputs=tf.reshape(self.rnn_module.output.outputs, [-1, self.num_units])
            for unit in output_hidden_units:
                outputs=tf.layers.dense(outputs, unit, kernel_initializer=tf.random_uniform_initializer(-scale, scale), kernel_regularizer=regularizer)
                scale=init_scale/unit

        with tf.variable_scope('classify_layer', initializer=tf.random_uniform_initializer(-scale, scale)):
            logits=tf.reshape(tf.layers.dense(outputs, class_num, kernel_regularizer=regularizer), [self.batch_size, -1, class_num])
            if class_num>1:
                self.result=tf.nn.softmax(logits)
                if self.top_k is not None:
                    self.result=tf.nn.top_k(self.result, k=self.top_k)
            else:
                self.result=tf.nn.sigmoid(logits)

        if running_type==ci.RunningType.train or running_type==ci.RunningType.valid or running_type==ci.RunningType.test:
            self.target=self.data_src_dict['target']
            target_num=1 if one_hot_label else class_num

            if last_rnn_output_only:
                if self.dynamic:
                    self.logits=tf.concat([one_logit[:one_sl] for one_sl, one_logit in zip(unstacked_sequence_length, tf.unstack(logits, num=self.batch_size, axis=0))], axis=0)
                    self.target=tf.concat([one_target[:one_sl] for one_sl, one_target in zip(unstacked_sequence_length, tf.unstack(self.target, num=self.batch_size, axis=0))], axis=0)
                    the_div=tf.reduce_sum(tf.cast(tf.greater(self.input.sequence_length, 0), dtype='float32'))
                else:
                    self.logits=logits
                    self.target=tf.squeeze(self.target, axis=1)
                    the_div=float(self.batch_size)
            else:
                if self.dynamic:
                    self.logits=tf.concat([tf.reshape(one_logit[:one_sl], [-1, class_num]) for one_sl, one_logit in zip(unstacked_sequence_length, tf.unstack(logits, num=self.batch_size, axis=0))], axis=0)
                    self.target=tf.concat([tf.reshape(one_target[:one_sl], [-1, target_num]) for one_sl, one_target in zip(unstacked_sequence_length, tf.unstack(self.target, num=self.batch_size, axis=0))], axis=0)
                    the_div=tf.cast(tf.reduce_sum(self.input.sequence_length), dtype='float32')
                else:
                    self.logits=tf.reshape(logits, [-1, class_num])
                    self.target=tf.reshape(self.target, [-1, target_num])
                    the_div=float(self.batch_size*self.input.input_data.shape[1].value)

            if class_num==1:
                self.loss=tf.nn.sigmoid_cross_entropy_with_logits(labels=self.target, logits=self.logits, name='loss')
            else:
                if one_hot_label:
                    self.loss=tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.squeeze(self.target, 1), logits=self.logits, name='loss')
                else:
                    self.loss=tf.nn.softmax_cross_entropy_with_logits(labels=self.target, logits=self.logits, name='loss')
            self.cost=tf.truediv(tf.reduce_sum(self.loss), the_div, name='cost')

            if running_type==ci.RunningType.train:
                self.lr=tf.Variable(learning_rate, trainable=False, name='learning_rate')
                self.optimizer=optimizer_getter(self.lr)
                if training_last_weight_only:
                    variables_to_train=[]
                    for var in tf.trainable_variables():
                        if 'classify_layer' in var.name:
                            variables_to_train.append(var)
                else:
                    variables_to_train=None
                grads, vars=zip(*self.optimizer.compute_gradients(self.cost, var_list=variables_to_train))
                grads, _=tf.clip_by_global_norm([tf.clip_by_value(g, -max_grad_value, max_grad_value) for g in grads], max_grad_norm)
                self.grads=list(zip(grads, vars))
                self.train_op=self.optimizer.apply_gradients(self.grads)

    def assign_lr(self, session, lr_value):
        session.run(tf.assign(self.lr, lr_value))

    def set_summary(self, session, log_dir, verbose):
        self.summary=tf.summary.merge_all()
        self.summary_writer=tf.summary.FileWriter(log_dir, session.graph)
        self.summary_verbose=verbose
        self.total_step=0

    def run_epoch(self, session, eval_op, verbose=-1):
        start_time=time.time()
        costs=0.0
        iters=0
        try:
            while True:
                if self.summary is not None:
                    if self.total_step%self.summary_verbose==0:
                        summary_str=session.run(self.summary)
                        self.summary_writer.add_summary(summary_str, self.total_step)
                        sys.stderr.write('summary wrote at total_step={}\n'.format(self.total_step))
                    self.total_step+=1

                res=session.run([self.cost] if eval_op is None else [self.cost, eval_op])
                cost=res[0]
                costs+=cost
                iters+=1

                if verbose>=0 and iters%verbose==0:
                    sys.stderr.write('step {0} avg perplexity: {1:.3f} current perplexity: {2:.3f} speed: {3:.0f} wps\n'.format(iters, np.exp(costs/iters), np.exp(cost), iters*self.batch_size/max(time.time()-start_time, 1)))
        except tf.errors.OutOfRangeError:
            pass

        return np.exp(costs/(iters if iters!=0 else 1))

    def run_training(self, session, verbose=-1):
        return self.run_epoch(session, self.train_op, verbose)

    def run_predicting(self, session):
        start_time=time.time()
        iters=0
        try:
            while True:
                result=session.run([self.result, self.data_src_dict])
                yield result
                sys.stderr.write('step {0} finished. time: {1:.3f} sec\n'.format(iters, time.time()-start_time))
                iters+=1
        except tf.errors.OutOfRangeError:
            pass

    def run_estimating(self, session):
        start_time=time.time()
        iters=0
        try:
            while True:
                result=session.run([self.result, self.target, self.data_src_dict])
                yield result
                sys.stderr.write('step {0} finished. time: {1:.3f} sec\n'.format(iters, time.time()-start_time))
                iters+=1
        except tf.errors.OutOfRangeError:
            pass
