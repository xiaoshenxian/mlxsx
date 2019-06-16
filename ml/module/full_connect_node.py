# -*- coding: utf-8 -*-

import tensorflow as tf
from .node import *
from .. import config_info as ci

class FullConnectNode(TfNode):
    def __init__(self, the_input, running_type
               , dimensions
               , l1_reg=-1, l2_reg=-1, activation=tf.sigmoid
               , train_auto_encoder=False):
        """
        input:
                input

        output:
                output
                ae_out: not None if running_type==ci.RunningType.train and train_auto_encoder==True 
        """
        super().__init__(the_input, running_type)
        self.num_layer=len(dimensions)-1

        regularizer=None
        if running_type==ci.RunningType.train:
            if l1_reg>0 and l2_reg>0:
                regularizer=tf.contrib.layers.l1_l2_regularizer(l1_reg, l2_reg)
            elif l1_reg>0:
                regularizer=tf.contrib.layers.l1_regularizer(l1_reg)
            elif l2_reg>0:
                regularizer=tf.contrib.layers.l2_regularizer(l2_reg)

        self.weights=[]
        the_input=self.input.input
        input_size=dimensions[0]
        for idx, out_size in enumerate(dimensions[1:]):
            w=tf.get_variable('weight_{}'.format(idx), [input_size, out_size], regularizer=regularizer)
            b=tf.get_variable('bias_{}'.format(idx), [out_size])
            the_input=activation(tf.matmul(the_input, w)+b)
            self.weights.append((w, b))

        self.output=DataPackage()
        self.output.output=the_input
        self.output.ae_out=None

        if running_type==ci.RunningType.train and train_auto_encoder==True:
            self.ae_weights=[]
            for idx, ((w, b), out_size) in enumerate(zip(self.weights, dimensions[:-1])[::-1]):
                index=self.num_layer-idx-1
                w=tf.transpose(w, name='ae_weight_{}'.format(index))
                b=tf.get_variable('ae_bias_{}'.format(index), [out_size])
                the_input=activation(tf.matmul(the_input, w)+b)
                self.ae_weights.append((w, b))
            self.output.ae_out=the_input
