# -*- coding: utf-8 -*-

import tensorflow as tf
from .node import *
from .. import config_info as ci

class RNNNode(TfNode):
    def __init__(self, the_input, running_type
               , rnn_cell_generator, num_layers
               , input_keep_prob=1.0, output_keep_prob=1.0, state_keep_prob=1.0, variational_recurrent=False
               , state_is_tuple=True):
        """
        input:
                input_data
                initial_states
                sequence_length: None for static rnn

        output:
                outputs
                final_states
        """
        super().__init__(the_input, running_type)

        self.cell_list=[]
        cell_generator=rnn_cell_generator
        if running_type==ci.RunningType.train and (input_keep_prob<1 or output_keep_prob<1 or state_keep_prob<1):
            def cell_generator(layer_idx):
                in_keep_prob=input_keep_prob
                out_keep_prob=output_keep_prob
                if input_keep_prob<1 and output_keep_prob<1:
                    if layer_idx>0:
                        in_keep_prob=1
                    if layer_idx<num_layers-1:
                        out_keep_prob=min(input_keep_prob, output_keep_prob)
                return tf.nn.rnn_cell.DropoutWrapper(rnn_cell_generator(layer_idx)
                                                   , input_keep_prob=in_keep_prob, output_keep_prob=out_keep_prob, state_keep_prob=state_keep_prob
                                                   , variational_recurrent=variational_recurrent, input_size=((self.input.input_data.shape[-1].value if layer_idx==0 else self.cell_list[i-1].output_size) if (variational_recurrent and input_keep_prob<1) else None), dtype=self.input.input_data.dtype)
        if num_layers<=1:
            self.cell=cell_generator(0)
            self.cell_list.append(self.cell)
        else:
            for i in range(num_layers):
                self.cell_list.append(cell_generator(i))
            self.cell=tf.nn.rnn_cell.MultiRNNCell(self.cell_list, state_is_tuple=state_is_tuple)

        outputs, states=tf.nn.dynamic_rnn(self.cell, inputs=self.input.input_data, initial_state=self.input.initial_states, dtype=self.input.input_data.dtype, sequence_length=self.input.sequence_length)

        self.output=DataPackage()
        self.output.outputs=outputs
        self.output.final_states=states
