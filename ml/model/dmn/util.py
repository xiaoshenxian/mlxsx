# -*- coding: utf-8 -*-

import tensorflow as tf
import math

class AttentionGRUWrapper(tf.nn.rnn_cell.RNNCell):
    def __init__(self, input_units, num_units, state_is_tuple=True, reuse=None):
        super().__init__(_reuse=reuse)
        self.input_units=input_units
        self._cell=tf.nn.rnn_cell.GRUCell(num_units, reuse=reuse)
        self._state_is_tuple=state_is_tuple

    @property
    def state_size(self):
        return self._cell.state_size

    @property
    def output_size(self):
        return self._cell.output_size+1

    def call(self, inputs, state):
        the_inputs, attn=tf.split(inputs, num_or_size_splits=[self.input_units, 1], axis=1)
        _, new_state=self._cell(the_inputs, state)
        new_state=attn*new_state+(1-attn)*state
        return tf.concat([new_state, attn], axis=1), new_state

class Attention:
    def __init__(self, num_layers, hidden_size, init_scale, regularizer, has_end_of_passes, sequence_length=None, use_softmax=True):
        self.num_layers=num_layers
        self.hidden_size=hidden_size
        self.init_scale=init_scale
        self.regularizer=regularizer
        self.has_end_of_passes=has_end_of_passes
        self.sequence_length=sequence_length
        if sequence_length is not None:
            self.valid_mask=tf.sequence_mask(sequence_length)
        self.use_softmax=use_softmax

    def cal_attention(self, inputs, m, q):
        mm=tf.concat(m, axis=-1) if self.num_layers>1 else m
        scale=self.init_scale/math.sqrt(inputs.shape[-1].value**2+mm.shape[-1].value**2)
        with tf.variable_scope('weight_c_m', initializer=tf.truncated_normal_initializer(stddev=scale)):
            w_c_m=tf.get_variable('w_c_m', [inputs.shape[-1].value, mm.shape[-1].value])
        if q is None:
            def mapping_func(packed):
                x, xx=packed
                return tf.concat([x
                                , mm
                                , tf.multiply(xx, mm)
                                , tf.sqrt(tf.maximum(tf.reduce_sum(tf.square(xx-mm), axis=1, keep_dims=True), 0))
                                , tf.reduce_sum(tf.multiply(tf.matmul(x, w_c_m), mm), axis=1, keep_dims=True)], axis=-1), xx
        else:
            qq=tf.concat(q, axis=-1) if self.num_layers>1 else q
            scale=self.init_scale/math.sqrt(inputs.shape[-1].value**2+qq.shape[-1].value**2)
            with tf.variable_scope('weight_c_q', initializer=tf.truncated_normal_initializer(stddev=scale)):
                w_c_q=tf.get_variable('w_c_q', [inputs.shape[-1].value, qq.shape[-1].value])
            def mapping_func(packed):
                x, xx=packed
                return tf.concat([x
                                , m
                                , tf.multiply(xx, mm)
                                , tf.sqrt(tf.maximum(tf.reduce_sum(tf.square(xx-mm), axis=1, keep_dims=True), 0))
                                , tf.reduce_sum(tf.multiply(tf.matmul(x, w_c_m), mm), axis=1, keep_dims=True)
                                , q
                                , tf.multiply(xx, qq)
                                , tf.sqrt(tf.maximum(tf.reduce_sum(tf.square(xx-qq), axis=1, keep_dims=True), 0))
                                , tf.reduce_sum(tf.multiply(tf.matmul(x, w_c_q), qq), axis=1, keep_dims=True)], axis=-1), xx

        trans_inputs=tf.transpose(inputs, perm=[1, 0, 2])
        multi_trans_inputs=tf.tile(trans_inputs, multiples=[1, 1, self.num_layers]) if self.num_layers>1 else trans_inputs
        attn, _=tf.map_fn(mapping_func, (trans_inputs, multi_trans_inputs))
        attn=tf.transpose(attn, perm=[1, 0, 2])

        scale=self.init_scale/attn.shape[-1].value
        with tf.variable_scope('layer1', initializer=tf.truncated_normal_initializer(stddev=scale)):
            attn=tf.layers.dense(attn, self.hidden_size, activation=tf.tanh, kernel_regularizer=None)
        scale=self.init_scale/self.hidden_size
        with tf.variable_scope('layer2', initializer=tf.truncated_normal_initializer(stddev=scale)):
            attn=tf.layers.dense(attn, 1, activation=None, kernel_regularizer=self.regularizer)

        with tf.variable_scope('final'):
            attn=tf.squeeze(attn, axis=2)
            if self.use_softmax:
                attn=tf.exp(attn)
                if self.sequence_length is not None:
                    attn=tf.where(self.valid_mask, attn, tf.zeros_like(attn))
                attn=tf.truediv(attn, tf.reduce_sum(attn, axis=1, keep_dims=True))
            else:
                attn=tf.sigmoid(attn)
                if self.sequence_length is not None:
                    attn=tf.where(self.valid_mask, attn, tf.zeros_like(attn))

            if self.has_end_of_passes:
                _, max_idx=tf.nn.top_k(attn, k=1)
                if self.sequence_length is not None:
                    is_end_of_passes=tf.logical_or(self.sequence_length<=0, tf.equal(max_idx[:,0], self.sequence_length-1))
                else:
                    is_end_of_passes=tf.equal(max_idx[:,0], tf.reduce_sum(tf.ones_like(attn, dtype=tf.int32), axis=1)-1)
            else:
                is_end_of_passes=None

            attn=tf.expand_dims(attn, axis=2)

        return attn, is_end_of_passes
