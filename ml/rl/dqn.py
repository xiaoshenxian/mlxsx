# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np

from .. import config_info as ci
from . import common as rl_common
from ..module import common as ml_common

class DQNComponent(rl_common.RLComponent):
    def __init__(self, running_type
               , net_generator, state, state_, discrete_action_num_list
               , use_dueling, tau):
        self.state=state
        self.state_=state_

        with tf.variable_scope('eval') as scope:
            self.discrete_critic_info_eval, self.net_info_eval=DQNComponent._build_net(net_generator, self.state, use_dueling, discrete_action_num_list)
            self.eval_variables=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope.name)

        if running_type==ci.RunningType.train or running_type==ci.RunningType.valid or running_type==ci.RunningType.test:
            with tf.variable_scope('target') as scope:
                self.discrete_critic_info_target, self.net_info_target=DQNComponent._build_net(net_generator, self.state_, use_dueling, discrete_action_num_list)
                self.target_variables=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope.name)
            if running_type==ci.RunningType.train:
                self.replace_target_op=[tf.assign(t, tau*e+(1-tau)*t) for e, t in zip(self.eval_variables, self.target_variables)]

    @staticmethod
    def _build_net(net_generator, input_state, use_dueling, discrete_action_num_list):
        discrete_critic_info=[]

        discrete_out, net_info=net_generator(input_state)

        discrete_count=len(discrete_action_num_list)

        if len(discrete_out)!=discrete_count:
            raise ValueError('DQNComponent::_build_net: Discrete output number mismatch! Expect {} received {}.'.format(discrete_count, len(discrete_out)))
        for num_actions, curr_out in zip(discrete_action_num_list, discrete_out):
            expect_last_dim=num_actions+1 if use_dueling else num_actions
            if curr_out.shape[-1].value!=expect_last_dim:
                raise ValueError('DQNComponent::_build_net: Net output last dimension mismatch! Expect {} received {}.'.format(expect_last_dim, curr_out.shape[-1].value))
            if use_dueling:
                val, adv=tf.split(curr_out, num_or_size_splits=[1, -1], axis=-1)
                adv=adv-tf.reduce_mean(adv, axis=-1, keep_dims=True)
                q=val+adv
                discrete_critic_info.append((q, val, adv))
            else:
                discrete_critic_info.append((curr_out, None, None))

        return discrete_critic_info, net_info

    def get_eval_op(self):
        return self.discrete_critic_info_eval

    def get_target_op(self):
        return self.discrete_critic_info_target

    def get_replace_target_op(self):
        return self.replace_target_op

    def get_trainable_variables(self):
        return self.eval_variables

class DeepQNetwork(rl_common.RL):
    def __init__(self, running_type
               , net_generator, state, discrete_action_num_list
               , action, state_, reward
               , use_double, use_dueling, tau, replace_target_span
               , learning_rate, gamma
               , max_grad_value, max_grad_norm, optimizer_getter
               , loss_mask=None):
        self.state=state
        self.action=action
        self.state_=state_
        self.reward=reward
        self.running_type=running_type
        self.discrete_action_num_list=discrete_action_num_list
        self.use_double=use_double
        self.replace_target_span=replace_target_span
        self.learning_rate=learning_rate
        self.gamma=gamma
        if loss_mask is not None:
            self.loss_mask=tf.expand_dims(loss_mask, axis=-1)
            self.mask_count=tf.maximum(tf.reduce_sum(loss_mask), 1)
        else:
            self.loss_mask=None
            self.mask_count=None

        self.dqn=DQNComponent(running_type
                            , net_generator, self.state, self.state_, discrete_action_num_list
                            , use_dueling, tau if use_double else 1)
        self.discrete_action_op=[q for (q, _, _) in self.dqn.get_eval_op()]
        self.continuous_action_op=[]

        if running_type==ci.RunningType.train or running_type==ci.RunningType.valid or running_type==ci.RunningType.test:
            discrete_critic_info_eval=self.dqn.get_eval_op()
            discrete_critic_info_target=self.dqn.get_target_op()
            discrete_q_loss_list=[]
            for act, num_actions, q_info_eval, q_info_target in zip(self.action, discrete_action_num_list, discrete_critic_info_eval, discrete_critic_info_target):
                action_vec=tf.one_hot(act, num_actions)
                the_q_eval=tf.reduce_sum(tf.multiply(action_vec, q_info_eval[0]), axis=-1, keep_dims=True)
                the_q_target=self.reward+self.gamma*tf.reduce_sum(tf.multiply(action_vec, q_info_target[0]), axis=-1, keep_dims=True)
                discrete_q_loss_list.append(tf.squared_difference(the_q_eval, the_q_target))
            self.loss=tf.concat(discrete_q_loss_list, axis=-1, name='loss')
            if self.loss_mask is not None:
                self.cost=tf.truediv(tf.reduce_sum(self.loss*self.loss_mask), self.mask_count, name='cost')
            else:
                self.cost=tf.reduce_mean(tf.reduce_sum(self.loss, axis=-1), name='cost')

            if running_type==ci.RunningType.train:
                self.lr, self.optimizer, self.grads, self.train_op, self.global_step=ml_common.add_train_op(cost=self.cost, learning_rate=self.learning_rate, optimizer_getter=optimizer_getter, max_grad_value=max_grad_value, max_grad_norm=max_grad_norm, global_step=None, var_list=self.dqn.get_trainable_variables())

    def replace_target(self, sess, iters):
        if self.running_type==ci.RunningType.train:
            if not self.use_double or iters%self.replace_target_span==0:
                sess.run(self.dqn.get_replace_target_op())
