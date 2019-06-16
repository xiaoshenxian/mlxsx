# -*- coding: utf-8 -*-

import time
import tensorflow as tf
import numpy as np

class TimeSeriesModel:
    def _verbose_func(avg_cost):
        return np.exp(avg_cost)

    def _optimizer_getter(lr):
        return tf.train.GradientDescentOptimizer(lr)

    def __init__(self, is_training, config, result_map=None, loss_map=None, input_proj=None, use_peepholes=True, state_is_tuple=False, optimizer_getter=_optimizer_getter, verbose_func=_verbose_func):
        self.verbose_func=verbose_func
        self.batch_size=config.batch_size
        self.num_steps=config.input_shape[0]
        self.num_units=config.num_units
        self.last_units=config.last_units

        self.input_data=tf.placeholder(tf.float32, [self.batch_size]+config.input_shape)
        self.targets=tf.placeholder(tf.float32, [self.batch_size]+config.target_shape)
        self.initial_state=tf.placeholder(tf.float32, [self.batch_size, 2*self.num_units*config.num_layers])

        def lstm_cell():
            return tf.nn.rnn_cell.LSTMCell(self.num_units, use_peepholes=use_peepholes, cell_clip=0.95, state_is_tuple=state_is_tuple)
        attn_cell=lstm_cell
        if is_training and config.keep_prob<1:
            def attn_cell():
                return tf.nn.rnn_cell.DropoutWrapper(lstm_cell(), output_keep_prob=config.keep_prob)
        self.cell=tf.nn.rnn_cell.MultiRNNCell([attn_cell() for _ in range(config.num_layers)], state_is_tuple=state_is_tuple)

        if input_proj is None:
            inputs=self.input_data
        else:
            with tf.variable_scope("input_proj", reuse=True):
                inputs=input_proj(self.input_data)

        outputs, state=tf.nn.static_rnn(self.cell, tf.unstack(inputs, num=self.num_steps, axis=1), initial_state=self.initial_state)
        if is_training and loss_map is not None:
            the_inputs=tf.reshape(inputs, [-1]+inputs.get_shape()[2:].as_list())

        self.output=tf.reshape(tf.concat(values=outputs, axis=1), [-1, self.num_units])#[batch_size*self.num_steps, self.num_units]
        self.last_w=tf.get_variable("last_w", [self.num_units, self.last_units])
        self.last_b=tf.get_variable("last_b", [self.last_units])
        self.logits=tf.matmul(self.output, self.last_w)+self.last_b
        self.final_state=state

        if result_map is None:
            self.result=tf.nn.softmax(self.logits)
        else:
            with tf.variable_scope("result_map", reuse=True):
                self.result=result_map(self.logits)
        self.result=tf.reshape(self.result, [self.batch_size, -1, self.last_units])#[batch_size, self.num_steps, self.last_units]

        if is_training:
            if loss_map is None:
                self.loss=tf.contrib.seq2seq.sequence_loss([self.logits], [tf.reshape(self.targets, [-1])], [tf.ones([self.batch_size*self.num_steps])])
            else:
                with tf.variable_scope("loss_map", reuse=True):
                    self.loss=loss_map(self.logits, self.targets, the_inputs)

            self.cost=tf.reduce_sum(self.loss)/self.batch_size

            self.lr=tf.Variable(config.learning_rate, trainable=False)
            tvars=tf.trainable_variables()
            grads, _=tf.clip_by_global_norm(tf.gradients(self.cost, tvars), config.max_grad_norm)
            self.optimizer=optimizer_getter(self.lr)
            self.train_op=self.optimizer.apply_gradients(zip(grads, tvars))

    def assign_lr(self, session, lr_value):
        session.run(tf.assign(self.lr, lr_value))

    def run_epoch(self, session, data_iterator, eval_op, verbose=-1):
        start_time=time.time()
        costs=0.0
        iters=0
        state=self.cell.zero_state(self.batch_size, tf.float32).eval()
        for step, (x, y) in enumerate(data_iterator):
            cost, state, _=session.run([self.cost, self.final_state, eval_op], feed_dict={self.input_data:x, self.targets:y, self.initial_state:state})
            costs+=cost
            iters+=self.num_steps

            if verbose>=0 and step%verbose==0:
                print("step {0} avg perplexity: {1:.3f} current perplexity: {2:.3f} speed: {3:.0f} wps".format(step, self.verbose_func(costs/iters), self.verbose_func(cost), iters*self.batch_size/(time.time()-start_time)))

        return self.verbose_func(costs/(iters if iters!=0 else 1))

    def run_training(self, session, data_iterator, verbose=-1):
        start_time=time.time()
        costs=0.0
        iters=0
        for step, (x, y, state) in enumerate(data_iterator):
            cost, state, _=session.run([self.cost, self.final_state, self.train_op], feed_dict={self.input_data:x, self.targets:y, self.initial_state:state})
            costs+=cost
            iters+=self.num_steps

            if verbose>=0 and step%verbose==0:
                print("step {0} avg perplexity: {1:.3f} current perplexity: {2:.3f} speed: {3:.0f} sps".format(step, self.verbose_func(costs/iters), self.verbose_func(cost/self.num_steps), iters*self.batch_size/(time.time()-start_time)))

        return self.verbose_func(costs/(iters if iters!=0 else 1))

    def run_predicting(self, session, data_iterator):
        start_time=time.time()
        for step, (input, state) in enumerate(data_iterator):
            if state is None:
                state=self.initial_state.eval()
            result, state=session.run([self.result, self.final_state], feed_dict={self.input_data:input, self.initial_state:state})
            yield result, state
            print("step {0} finished. time: {1:.3f} sec".format(step, time.time()-start_time))
