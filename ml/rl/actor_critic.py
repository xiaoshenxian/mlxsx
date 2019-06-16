# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np

from .. import config_info as ci
from . import common as rl_common
from ..module import common as ml_common

class Actor(rl_common.RLComponent):
    def __init__(self, running_type
               , net_generator, state, discrete_action_num_list
               , use_double, tau
               , action_lower, action_upper, sigma_delta, use_deterministic
               , state_
               , extra_eval_vars=[], extra_target_vars=[]):
        self.state=state
        self.state_=state_

        continuous_action_config=self._check_continuous_param(action_lower, action_upper, sigma_delta, use_deterministic)

        with tf.variable_scope('eval') as scope:
            self.discrete_action_info_eval, self.continuous_action_info_eval, self.net_info_eval=Actor._build_net(net_generator, self.state, discrete_action_num_list, continuous_action_config)
            self.eval_variables=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope.name)+extra_eval_vars

        if running_type!=ci.RunningType.predict and use_double:
            if len(extra_eval_vars)!=len(extra_target_vars):
                raise ValueError('Actor::__init__: The sizes of extra_eval_vars ({}) and extra_target_vars ({}) must be equal!'.format(len(extra_eval_vars), len(extra_target_vars)))
            with tf.variable_scope('target') as scope:
                self.discrete_action_info_target, self.continuous_action_info_target, self.net_info_target=Actor._build_net(net_generator, self.state_, discrete_action_num_list, continuous_action_config)
                self.target_variables=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope.name)+extra_target_vars
            if running_type==ci.RunningType.train:
                self.replace_target_op=[tf.assign(t, tau*e+(1-tau)*t) for e, t in zip(self.eval_variables, self.target_variables)]

    def _check_continuous_param(self, action_lower, action_upper, sigma_delta, use_deterministic):
        if action_lower is not None and action_upper is not None:
            if len(action_lower.shape)!=len(action_upper.shape)!=1 or len(action_lower)!=len(action_upper) or (sigma_delta is not None and (len(action_lower.shape)!=len(sigma_delta.shape)!=1 or len(action_lower)!=len(sigma_delta))):
                raise ValueError('Actor::_check_continuous_param: Continuous action parameters must be 1-dimensional numpy array with equal length!')
            self.continuous_action_len=len(action_lower)
            self.action_lower=tf.constant(action_lower, dtype=action_lower.dtype)
            self.action_upper=tf.constant(action_upper, dtype=action_upper.dtype)
            self.sigma_delta=None if sigma_delta is None else tf.constant(sigma_delta, dtype=sigma_delta.dtype)
            self.use_deterministic=use_deterministic
            return self.continuous_action_len, self.action_lower, self.action_upper, self.sigma_delta, self.use_deterministic
        else:
            return None

    @staticmethod
    def _build_net(net_generator, input_state, discrete_action_num_list, continuous_action_config):
        discrete_action_info=[]

        out, net_info=net_generator(input_state)

        discrete_count=len(discrete_action_num_list)
        continuous_count=0 if continuous_action_config is None else 1
        discrete_out=out[:discrete_count]
        continuous_out=out[discrete_count:]

        if len(discrete_out)!=discrete_count:
            raise ValueError('Actor::_build_net: Discrete output number mismatch! Expect {} received {}.'.format(discrete_count, len(discrete_out)))
        for num_actions, curr_out in zip(discrete_action_num_list, discrete_out):
            if curr_out.shape[-1].value!=num_actions:
                raise ValueError('Actor::_build_net: Net output last dimension mismatch! Expect {} received {}.'.format(num_actions, curr_out.shape[-1].value))
            action_prob=tf.nn.softmax(curr_out)
            discrete_action_info.append(action_prob)

        if len(continuous_out)!=continuous_count:
            raise ValueError('Actor::_build_net: Continuous output number mismatch! Expect {} received {}.'.format(continuous_count, len(continuous_out)))
        if continuous_action_config is not None:
            continuous_action_len, action_lower, action_upper, sigma_delta, use_deterministic=continuous_action_config
            continuous_out=continuous_out[0]
            if use_deterministic:
                if continuous_out.shape[-1].value!=continuous_action_len:
                    raise ValueError('Actor::_build_net: Net output last dimension mismatch! Expect {} for modeling deterministic continuous valued action, but {} received!'.format(continuous_action_len, continuous_out.shape[-1].value))
                action=tf.nn.sigmoid(continuous_out)*(action_upper-action_lower)+action_lower
                continuous_action_info=(None, action)
            else:
                if continuous_out.shape[-1].value!=2 or continuous_out.shape[-2].value!=continuous_action_len:
                    raise ValueError('Actor::_build_net: Net output last dimension mismatch! Expect [{}, 2] for modeling continuous valued action, but [{}, {}] received!'.format(continuous_action_len, continuous_out.shape[-2].value, continuous_out.shape[-1].value))
                mu, sigma=tf.unstack(continuous_out, num=2, axis=-1)
                mu=tf.nn.sigmoid(mu)
                sigma=tf.nn.softplus(sigma)
                mu=mu*(action_upper-action_lower)+action_lower
                if sigma_delta is not None:
                    sigma=sigma+sigma_delta
                normal_dist=tf.distributions.Normal(mu, sigma)
                action=tf.clip_by_value(tf.squeeze(normal_dist.sample(1), axis=0), action_lower, action_upper)
                continuous_action_info=(normal_dist, action)
        else:
            continuous_action_info=None

        return discrete_action_info, continuous_action_info, net_info

    def get_eval_op(self):
        return self.discrete_action_info_eval, self.continuous_action_info_eval

    def get_target_op(self):
        return self.discrete_action_info_target, self.continuous_action_info_target

    def get_replace_target_op(self):
        return self.replace_target_op

    def get_trainable_variables(self):
        return self.eval_variables

class Critic(rl_common.RLComponent):
    def __init__(self, running_type
               , net_generator, input, input_, input_for_actor, discrete_action_num_list, continuous_action_length
               , use_dueling, tau, estimate_val_instead
               , extra_eval_vars=[], extra_target_vars=[]):
        self.input=input
        self.input_=input_
        self.input_for_actor=input_for_actor
        
        if len(extra_eval_vars)!=len(extra_target_vars):
            raise ValueError('Critic::__init__: The sizes of extra_eval_vars ({}) and extra_target_vars ({}) must be equal!'.format(len(extra_eval_vars), len(extra_target_vars)))

        with tf.variable_scope('eval') as scope:
            self.discrete_critic_info_eval, self.continuous_critic_info_eval, self.net_info_eval=Critic._build_net(net_generator, self.input, use_dueling, discrete_action_num_list, continuous_action_length, estimate_val_instead)
            self.eval_variables=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope.name)+extra_eval_vars

        with tf.variable_scope('target') as scope:
            self.discrete_critic_info_target, self.continuous_critic_info_target, self.net_info_target=Critic._build_net(net_generator, self.input_, use_dueling, discrete_action_num_list, continuous_action_length, estimate_val_instead)
            self.target_variables=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope.name)+extra_target_vars
        if running_type==ci.RunningType.train:
            self.replace_target_op=[tf.assign(t, tau*e+(1-tau)*t) for e, t in zip(self.eval_variables, self.target_variables)]

        if self.input_for_actor is not None:
            with tf.variable_scope('eval', reuse=True) as scope:
                self.discrete_critic_info_eval_for_actor, self.continuous_critic_info_eval_for_actor, self.net_info_for_actor=Critic._build_net(net_generator, self.input_for_actor, use_dueling, discrete_action_num_list, continuous_action_length, estimate_val_instead)

    @staticmethod
    def _build_net(net_generator, input_state, use_dueling, discrete_action_num_list, continuous_action_length, estimate_val_instead):
        discrete_critic_info=[]

        out, net_info=net_generator(input_state)

        discrete_count=len(discrete_action_num_list)
        continuous_count=1 if continuous_action_length>0 else 0
        discrete_out=out[:discrete_count]
        continuous_out=out[discrete_count:]

        if len(discrete_out)!=discrete_count:
            raise ValueError('Critic::_build_net: Discrete output number mismatch! Expect {} received {}.'.format(discrete_count, len(discrete_out)))
        for idx, (num_actions, curr_out) in enumerate(zip(discrete_action_num_list, discrete_out)):
            expect_last_dim=num_actions+1 if use_dueling else (1 if estimate_val_instead else num_actions)
            if curr_out.shape[-1].value!=expect_last_dim:
                raise ValueError('Critic::_build_net: Net output last dimension mismatch! Expect {}, but receive {}.'.format(expect_last_dim, curr_out.shape[-1].value))
            if use_dueling:
                val, adv=tf.split(curr_out, num_or_size_splits=[1, -1], axis=-1)
                adv=adv-tf.reduce_mean(adv, axis=-1, keep_dims=True)
                q=val+adv
                discrete_critic_info.append((q, val, adv))
            else:
                if estimate_val_instead:
                    discrete_critic_info.append((None, curr_out, None))
                else:
                    discrete_critic_info.append((curr_out, None, None))

        if len(continuous_out)!=continuous_count:
            raise ValueError('Critic::_build_net: Continuous output number mismatch! Expect {} received {}.'.format(continuous_count, len(continuous_out)))
        if continuous_action_length>0:#not able to perform dueling for only one action output
            continuous_out=continuous_out[0]
            if continuous_out.shape[-1].value!=continuous_action_length:
                raise ValueError('Critic::_build_net: Net output last dimension mismatch! Expect {} continuous actions, but receive {}.'.format(continuous_action_length, curr_out.shape[-1].value))
            continuous_critic_info=continuous_out
        else:
            continuous_critic_info=None

        return discrete_critic_info, continuous_critic_info, net_info

    def get_eval_op(self):
        return self.discrete_critic_info_eval, self.continuous_critic_info_eval

    def get_target_op(self):
        return self.discrete_critic_info_target, self.continuous_critic_info_target

    def get_replace_target_op(self):
        return self.replace_target_op

    def get_trainable_variables(self):
        return self.eval_variables

class ActorCritic(rl_common.RL):
    def __init__(self, running_type
               , actor_net_generator, state
               , discrete_action_num_list, action_lower, action_upper, sigma_delta, use_deterministic
               , critic_net_generator, action, state_, reward
               , use_double, use_dueling
               , actor_tau, replace_actor_target_span, critic_tau, replace_critic_target_span, estimate_val_instead
               , actor_learning_rate, critic_learning_rate, gamma
               , max_grad_value, max_grad_norm, optimizer_getter
               , entropy_factor=0, num_e=1e-8
               , loss_mask=None
               , extra_actor_eval_vars=[], extra_actor_target_vars=[]
               , extra_critic_eval_vars=[], extra_critic_target_vars=[]):
        self.state=state
        self.action=action
        self.state_=state_
        self.reward=reward
        self.running_type=running_type
        self.discrete_action_num_list=discrete_action_num_list
        self.use_double=use_double
        self.replace_actor_target_span=replace_actor_target_span
        self.replace_critic_target_span=replace_critic_target_span
        self.actor_learning_rate=actor_learning_rate
        self.critic_learning_rate=critic_learning_rate
        self.gamma=gamma
        self.entropy_factor=entropy_factor
        if loss_mask is not None:
            self.loss_mask=tf.expand_dims(loss_mask, axis=-1)
            self.mask_count=tf.maximum(tf.reduce_sum(loss_mask), 1)
        else:
            self.loss_mask=None
            self.mask_count=None

        with tf.variable_scope('actor'):
            self.actor=Actor(running_type
                           , actor_net_generator, self.state, discrete_action_num_list
                           , use_double, actor_tau
                           , action_lower, action_upper, sigma_delta, use_deterministic
                           , self.state_
                           , extra_actor_eval_vars, extra_actor_target_vars)
        self.discrete_action_op, self.continuous_action_op=self.actor.get_eval_op()
        if self.continuous_action_op is None:
            self.continuous_action_op=[]
        else:
            self.continuous_action_op=[self.continuous_action_op[1]]

        if running_type==ci.RunningType.train or running_type==ci.RunningType.valid or running_type==ci.RunningType.test:
            discrete_action_info_eval, continuous_action_info_eval=self.actor.get_eval_op()

            with tf.variable_scope('critic_input'):
                discrete_eval_action_vec_list=[tf.one_hot(act, num_actions) for act, num_actions in zip(self.action[:len(discrete_action_num_list)], discrete_action_num_list)]
                if use_double:
                    discrete_action_info_target, continuous_action_info_target=self.actor.get_target_op()
                    critic_input=(self.state, discrete_eval_action_vec_list, None if continuous_action_info_eval is None else self.action[len(discrete_action_num_list)])
                    critic_input_=(self.state_, discrete_action_info_target, None if continuous_action_info_target is None else continuous_action_info_target[1])
                    critic_input_for_actor=(self.state, discrete_action_info_eval, None if continuous_action_info_eval is None else continuous_action_info_eval[1])
                else:
                    critic_input=self.state
                    critic_input_=self.state_
                    critic_input_for_actor=None

            with tf.variable_scope('critic'):
                self.critic=Critic(running_type
                                 , critic_net_generator, critic_input, critic_input_, critic_input_for_actor, discrete_action_num_list, 0 if action_lower is None else len(action_lower)
                                 , use_dueling, critic_tau if use_double else 1, estimate_val_instead
                                 , extra_critic_eval_vars, extra_critic_target_vars)

            if len(discrete_action_info_eval)!=len(self.critic.get_eval_op()[0]):
                raise ValueError('ActorCritic::__init__: Discrete count mismatch in actor ({}) and critic ({})!'.format(len(self.actor.get_eval_op()[0]), len(self.critic.get_eval_op()[0])))
            if continuous_action_info_eval is not None and continuous_action_info_eval[1].shape[-1].value!=self.critic.get_eval_op()[1][0].shape[-1].value:
                raise ValueError('ActorCritic::__init__: Continuous count mismatch in actor ({}) and critic ({})!'.format(continuous_action_info_eval[1].shape[-1].value, self.critic.get_eval_op()[1][0].shape[-1].value))

            with tf.variable_scope('critic_loss'):
                discrete_critic_info_eval, continuous_critic_info_eval=self.critic.get_eval_op()
                discrete_critic_info_target, continuous_critic_info_target=self.critic.get_target_op()
                discrete_critic_td_error_list=[]
                discrete_critic_loss_list=[]
                continuous_critic_loss_list=[]
                for act_vec_eval, q_info_eval, q_info_target in zip(discrete_eval_action_vec_list, discrete_critic_info_eval, discrete_critic_info_target):
                    if estimate_val_instead:
                        the_q_eval=q_info_eval[1]
                        the_q_target=self.reward+self.gamma*q_info_target[1]
                    else:
                        the_q_eval=tf.reduce_sum(tf.multiply(act_vec_eval, q_info_eval[0]), axis=-1, keep_dims=True)
                        the_q_target=self.reward+self.gamma*tf.reduce_sum(tf.multiply(act_vec_eval, q_info_target[0]), axis=-1, keep_dims=True)
                    td_error=the_q_target-the_q_eval
                    discrete_critic_td_error_list.append(td_error)
                    discrete_critic_loss_list.append(tf.square(td_error))
                if continuous_critic_info_eval is not None:
                    the_q_eval=continuous_critic_info_eval
                    the_q_target=self.reward+self.gamma*continuous_critic_info_target
                    continuous_critic_td_error=the_q_target-the_q_eval
                    continuous_critic_loss_list.append(tf.square(continuous_critic_td_error))
                self.critic_loss=tf.concat(discrete_critic_loss_list+continuous_critic_loss_list, axis=-1, name='critic_loss')
                if self.loss_mask is not None:
                    self.critic_cost=tf.truediv(tf.reduce_sum(self.critic_loss*self.loss_mask), self.mask_count, name='critic_cost')
                else:
                    self.critic_cost=tf.reduce_mean(tf.reduce_sum(self.critic_loss, axis=-1), name='critic_cost')

            with tf.variable_scope('actor_loss'):
                if use_double:
                    if estimate_val_instead:
                        actor_loss_list=[-q_info[1] for q_info in self.critic.discrete_critic_info_eval_for_actor]
                    else:
                        actor_loss_list=[-tf.reduce_sum(tf.multiply(tf.one_hot(tf.argmax(act, axis=-1), num_actions), q_info[0]), axis=-1, keep_dims=True) for act, num_actions, q_info in zip(discrete_action_info_eval, discrete_action_num_list, self.critic.discrete_critic_info_eval_for_actor)]
                    if continuous_action_info_eval is not None:
                        actor_loss_list.append(-self.critic.continuous_critic_info_eval_for_actor)
                    self.actor_loss=tf.concat(actor_loss_list, axis=-1, name='actor_loss')
                else:
                    actor_loss_list=[]
                    for action_vec, action_prob, critic_td_error in zip(discrete_eval_action_vec_list, discrete_action_info_eval, discrete_critic_td_error_list):
                        curr_loss=-tf.log(tf.reduce_sum(tf.multiply(action_vec, action_prob), axis=-1, keep_dims=True))*critic_td_error
                        if self.entropy_factor>0:
                            curr_loss+=self.entropy_factor*tf.reduce_sum(tf.multiply(action_prob, tf.log(action_prob+num_e)), axis=-1, keep_dims=True)
                        actor_loss_list.append(curr_loss)
                    if continuous_action_info_eval is not None:
                        act=self.action[len(discrete_action_num_list)]
                        normal_dist, _=continuous_action_info_eval
                        curr_loss=-normal_dist.log_prob(act)*continuous_critic_td_error
                        if self.entropy_factor>0:
                            curr_loss-=self.entropy_factor*normal_dist.entropy()
                        actor_loss_list.append(curr_loss)
                    self.actor_loss=tf.concat(actor_loss_list, axis=-1, name='actor_loss')
                if self.loss_mask is not None:
                    self.actor_cost=tf.truediv(tf.reduce_sum(self.actor_loss*self.loss_mask), self.mask_count, name='actor_cost')
                else:
                    self.actor_cost=tf.reduce_mean(tf.reduce_sum(self.actor_loss, axis=-1), name='actor_cost')

            self.cost=self.critic_cost+self.actor_cost

            if running_type==ci.RunningType.train:
                with tf.variable_scope('training_critic'):
                    self.critic_lr, self.critic_optimizer, self.critic_grads, self.critic_train_op, self.critic_global_step=ml_common.add_train_op(cost=self.critic_cost, learning_rate=self.critic_learning_rate, optimizer_getter=optimizer_getter, max_grad_value=max_grad_value, max_grad_norm=max_grad_norm, global_step=None, var_list=self.critic.get_trainable_variables())
                with tf.variable_scope('training_actor'):
                    self.actor_lr, self.actor_optimizer, self.actor_grads, self.actor_train_op, self.actor_global_step=ml_common.add_train_op(cost=self.actor_cost, learning_rate=self.actor_learning_rate, optimizer_getter=optimizer_getter, max_grad_value=max_grad_value, max_grad_norm=max_grad_norm, global_step=None, var_list=self.actor.get_trainable_variables())
                self.train_op=[self.critic_train_op, self.actor_train_op]

    def get_continuous_lower_op(self):
        return self.actor.action_lower

    def get_continuous_upper_op(self):
        return self.actor.action_upper

    def replace_target(self, sess, iters):
        if self.running_type==ci.RunningType.train:
            if self.use_double and iters%self.replace_actor_target_span==0:
                sess.run(self.actor.get_replace_target_op())
            if not self.use_double or iters%self.replace_critic_target_span==0:
                sess.run(self.critic.get_replace_target_op())
