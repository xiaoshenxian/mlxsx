# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np

from . import actor_critic as ac
from .. import config_info as ci
from . import common as rl_common
from ..module import common as ml_common

class PolicyUpdater:
    def policy_loss(self, pi, pi_old, dist, dist_old, adv, num_actions):
        raise NotImplementedError('PolicyUpdater::policy_loss is not implemented!')

    def updating_ctrl(self):
        raise NotImplementedError('PolicyUpdater::updating_ctrl is not implemented!')

    def after_actor_update(self):
        raise NotImplementedError('PolicyUpdater::after_actor_update is not implemented!')

class KLPenaltyPolicyUpdater(PolicyUpdater):
    def __init__(self, lam, KL_target, xi, alpha, beta_high, beta_low, lam_lower, lam_higher, num_e):
        self.lam=tf.Variable(lam, trainable=False)
        self.KL_target=KL_target
        self.xi=xi
        self.alpha=alpha
        self.beta_high=beta_high
        self.beta_low=beta_low
        self.lam_lower=lam_lower
        self.lam_higher=lam_higher
        self.num_e=num_e

    def policy_loss(self, pi, pi_old, dist, dist_old, adv, continuous_dist):
        self.ratio=pi/(pi_old+self.num_e)
        self.surr=tf.multiply(self.ratio, adv)
        if continuous_dist:
            self.kl=tf.distributions.kl_divergence(dist_old, dist)
        else:
            self.kl=tf.reduce_sum(tf.multiply(dist_old, tf.log(dist_old/(dist+self.num_e)+self.num_e)), axis=-1, keep_dims=True)
        self.kl_mean=tf.reduce_mean(self.kl)
        loss=self.lam*self.kl+self.xi*tf.square(tf.maximum(0.0, self.kl-2*self.KL_target))-self.surr
        return loss

    def updating_ctrl(self):
        return tf.less_equal(self.kl_mean, 4*self.KL_target)

    def after_actor_update(self):
        return tf.cond(tf.greater(self.kl_mean, self.beta_high*self.KL_target)
                     , lambda : tf.assign(self.lam, tf.clip_by_value(self.lam*self.alpha, self.lam_lower, self.lam_higher))
                     , lambda : tf.cond(tf.less(self.kl_mean, self.beta_low*self.KL_target)
                                      , lambda : tf.assign(self.lam, tf.clip_by_value(self.lam/self.alpha, self.lam_lower, self.lam_higher))
                                      , lambda : tf.assign(self.lam, self.lam)))

class ClipPolicyUpdater(PolicyUpdater):
    def __init__(self, epsilon, c, num_e):
        self.epsilon=epsilon
        self.c=c
        self.num_e=num_e

    def policy_loss(self, pi, pi_old, dist, dist_old, adv, continuous_dist):
        self.ratio=pi/(pi_old+self.num_e)
        self.surr=tf.multiply(self.ratio, adv)
        loss=-tf.minimum(self.surr, tf.clip_by_value(self.ratio, 1-self.epsilon, 1+self.epsilon)*adv)
        if self.c>0:
            if continuous_dist:
                loss-=self.c*dist.entropy()
            else:
                loss+=self.c*tf.reduce_sum(tf.multiply(dist, tf.log(dist+self.num_e)), axis=-1, keep_dims=True)
        return loss

    def updating_ctrl(self):
        return tf.constant(True, dtype=tf.bool)

    def after_actor_update(self):
        return tf.no_op()

class ProximalPolicyOptimizer(rl_common.RL):
    def __init__(self, running_type
               , actor_net_generator, state, sequence_length
               , discrete_action_num_list, action_lower, action_upper, sigma_delta
               , critic_net_generator, action, state_, reward
               , use_double, use_dueling, add_actor_action_to_critic
               , critic_tau, replace_critic_target_span, estimate_val_instead
               , policy_updater_list, actor_update_num, critic_update_num, recursive_reward_target
               , actor_learning_rate, critic_learning_rate, gamma
               , max_grad_value, max_grad_norm, optimizer_getter
               , loss_mask=None
               , extra_actor_eval_vars=[], extra_actor_target_vars=[]
               , extra_critic_eval_vars=[], extra_critic_target_vars=[]):
        self.state=state
        self.sequence_length=sequence_length
        self.action=action
        self.state_=state_
        self.reward=reward
        self.running_type=running_type
        self.discrete_action_num_list=discrete_action_num_list
        self.use_double=use_double
        self.replace_critic_target_span=replace_critic_target_span
        self.actor_learning_rate=actor_learning_rate
        self.critic_learning_rate=critic_learning_rate
        self.gamma=gamma
        self.policy_updater_list=policy_updater_list
        self.actor_update_num=actor_update_num
        self.critic_update_num=critic_update_num
        if loss_mask is not None:
            self.loss_mask=tf.expand_dims(loss_mask, axis=-1)
            self.mask_count=tf.maximum(tf.reduce_sum(loss_mask), 1)
        else:
            self.loss_mask=None
            self.mask_count=None

        with tf.variable_scope('actor'):
            self.actor=ac.Actor(running_type
                              , actor_net_generator, self.state, discrete_action_num_list
                              , True, 1
                              , action_lower, action_upper, sigma_delta, False
                              , self.state
                              , extra_actor_eval_vars, extra_actor_target_vars)
        self.discrete_action_op, self.continuous_action_op=self.actor.get_eval_op()
        if self.continuous_action_op is None:
            self.continuous_action_op=[]
        else:
            self.continuous_action_op=[self.continuous_action_op[1]]

        if running_type==ci.RunningType.train or running_type==ci.RunningType.valid or running_type==ci.RunningType.test:
            discrete_action_info_eval, continuous_action_info_eval=self.actor.get_eval_op()
            discrete_action_info_target, continuous_action_info_target=self.actor.get_target_op()
            discrete_eval_action_vec_list=[tf.one_hot(act, num_actions) for act, num_actions in zip(self.action[:len(discrete_action_num_list)], discrete_action_num_list)]

            with tf.variable_scope('critic_input'):
                if add_actor_action_to_critic:
                    discrete_action_info_target, continuous_action_info_target=self.actor.get_target_op()
                    critic_input=(self.state, discrete_eval_action_vec_list, None if continuous_action_info_eval is None else self.action[len(discrete_action_num_list)])
                    critic_input_=(self.state_, discrete_action_info_target, None if continuous_action_info_target is None else continuous_action_info_target[1])
                    #critic_input=tf.concat([self.state]+discrete_eval_action_vec_list+([] if continuous_action_info_eval is None else [self.action[len(discrete_action_num_list)]]), axis=-1)
                    #critic_input_=tf.concat([self.state_]+discrete_action_info_target+([] if continuous_action_info_target is None else [continuous_action_info_target[1]]), axis=-1)
                else:
                    critic_input=self.state
                    critic_input_=self.state_

            with tf.variable_scope('critic'):
                self.critic=ac.Critic(running_type
                                    , critic_net_generator, critic_input, critic_input_, None, discrete_action_num_list, 0 if action_lower is None else len(action_lower)
                                    , use_dueling, critic_tau if use_double else 1, estimate_val_instead
                                    , extra_critic_eval_vars, extra_critic_target_vars)

            if len(discrete_action_info_eval)!=len(self.critic.get_eval_op()[0]):
                raise ValueError('ProximalPolicyOptimizer::__init__: Discrete count mismatch in actor ({}) and critic ({})!'.format(len(discrete_action_info_eval), len(self.critic.get_eval_op()[0])))
            if continuous_action_info_eval is not None and continuous_action_info_eval[1].shape[-1].value!=self.critic.get_eval_op()[1][0].shape[-1].value:
                raise ValueError('ProximalPolicyOptimizer::__init__: Continuous count mismatch in actor ({}) and critic ({})!'.format(continuous_action_info_eval[1].shape[-1].value, self.critic.get_eval_op()[1][0].shape[-1].value))

            with tf.variable_scope('critic_loss'):
                discrete_q_info_eval, continuous_q_info_eval=self.critic.get_eval_op()
                discrete_q_info_target, continuous_q_info_target=self.critic.get_target_op()
                loss_param_list=[]
                for idx, (action_vec, q_info_eval, q_info_target) in enumerate(zip(discrete_eval_action_vec_list, discrete_q_info_eval, discrete_q_info_target)):
                    if estimate_val_instead:
                        the_q_eval=q_info_eval[1]
                        target_op=q_info_target[1]
                    else:
                        the_q_eval=tf.reduce_sum(tf.multiply(action_vec, q_info_eval[0]), axis=-1, keep_dims=True)
                        target_op=tf.reduce_sum(tf.multiply(action_vec, q_info_target[0]), axis=-1, keep_dims=True)
                    loss_param_list.append(('discrete_{}'.format(idx), the_q_eval, target_op))
                if continuous_q_info_eval is not None:
                    loss_param_list.append(('continuous', continuous_q_info_eval, continuous_q_info_target))
                q_target_list=[]
                critic_loss_list=[]
                for (scope_name, the_q_eval, target_op) in loss_param_list:
                    with tf.variable_scope('q_target'):
                        with tf.variable_scope(scope_name):
                            if recursive_reward_target:
                                def update_q_target(i, last_target, the_q_target_array):
                                    last_target=self.reward[:, i]+self.gamma*last_target
                                    the_q_target_array=the_q_target_array.write(i, last_target)
                                    return [i-1, last_target, the_q_target_array]
                                _, _, the_q_target_array=tf.while_loop(lambda i, last_target, the_q_target_array : i>=0
                                                                     , update_q_target
                                                                     , [self.sequence_length-1, target_op[:,-1], tf.TensorArray(dtype=target_op.dtype, size=self.sequence_length)]
                                                                     , parallel_iterations=1)
                                the_q_target=tf.transpose(the_q_target_array.stack(), perm=[1, 0]+[i for i in range(2, len(target_op.shape))])
                            else:
                                the_q_target=self.reward+self.gamma*target_op
                        q_target_list.append(the_q_target)
                        critic_loss_list.append(tf.squared_difference(the_q_eval, the_q_target))
                self.critic_loss=tf.concat(critic_loss_list, axis=-1, name='critic_loss')
                if self.loss_mask is not None:
                    self.critic_cost=tf.truediv(tf.reduce_sum(self.critic_loss*self.loss_mask), self.mask_count, name='critic_cost')
                else:
                    self.critic_cost=tf.reduce_mean(tf.reduce_sum(self.critic_loss, axis=-1), name='critic_cost')

            discrete_critic_info_eval, continuous_critic_info_eval=self.critic.get_eval_op()
            discrete_critic_info_target, continuous_critic_info_target=self.critic.get_target_op()
            with tf.variable_scope('actor_loss'):
                discrete_adv_list=[]
                for q_target, (q_eval, val_eval, adv_eval) in zip(q_target_list[:len(discrete_action_num_list)], discrete_critic_info_eval):
                    if use_dueling:
                        discrete_adv_list.append(adv_eval)
                    elif estimate_val_instead:
                        discrete_adv_list.append(q_target-val_eval)
                    else:
                        discrete_adv_list.append(q_target-tf.reduce_sum(tf.multiply(action_vec, q_eval), axis=-1, keep_dims=True))
                actor_loss_list=[]
                actor_updating_ctrl_op_list=[]
                self.after_actor_update_op=[]
                for action_vec, adv, action_prob_eval, action_prob_target, policy_updater in zip(discrete_eval_action_vec_list, discrete_adv_list, discrete_action_info_eval, discrete_action_info_target, self.policy_updater_list[:len(discrete_action_num_list)]):
                    pi=tf.reduce_sum(tf.multiply(action_vec, action_prob_eval), axis=-1, keep_dims=True)
                    pi_old=tf.reduce_sum(tf.multiply(action_vec, action_prob_target), axis=-1, keep_dims=True)
                    dist=action_prob_eval
                    dist_old=action_prob_target
                    actor_loss_list.append(policy_updater.policy_loss(pi, pi_old, dist, dist_old, adv, False))
                    actor_updating_ctrl_op_list.append(policy_updater.updating_ctrl())
                    self.after_actor_update_op.append(policy_updater.after_actor_update())
                if continuous_critic_info_target is not None:
                    adv=q_target_list[len(discrete_action_num_list)]-continuous_critic_info_eval
                    act=self.action[len(discrete_action_num_list)]
                    normal_dist_eval, _=continuous_action_info_eval
                    normal_dist_target, _=continuous_action_info_target
                    policy_updater=self.policy_updater_list[len(discrete_action_num_list)]
                    pi=normal_dist_eval.prob(act)
                    pi_old=normal_dist_target.prob(act)
                    dist=normal_dist_eval
                    dist_old=normal_dist_target
                    actor_loss_list.append(policy_updater.policy_loss(pi, pi_old, dist, dist_old, adv, True))
                    actor_updating_ctrl_op_list.append(policy_updater.updating_ctrl())
                    self.after_actor_update_op.append(policy_updater.after_actor_update())
                self.actor_loss=tf.concat(actor_loss_list, axis=-1, name='actor_loss')
                if self.loss_mask is not None:
                    self.actor_cost=tf.truediv(tf.reduce_sum(self.actor_loss*self.loss_mask), self.mask_count, name='actor_cost')
                else:
                    self.actor_cost=tf.reduce_mean(tf.reduce_sum(self.actor_loss, axis=-1), name='actor_cost')
                self.actor_updating_ctrl_op=tf.reduce_all(tf.stack(actor_updating_ctrl_op_list))

            if running_type==ci.RunningType.train:
                with tf.variable_scope('training_actor'):
                    self.actor_lr, self.actor_optimizer, self.actor_grads, self.actor_train_op, self.actor_global_step=ml_common.add_train_op(cost=self.actor_cost, learning_rate=self.actor_learning_rate, optimizer_getter=optimizer_getter, max_grad_value=max_grad_value, max_grad_norm=max_grad_norm, global_step=None, var_list=self.actor.get_trainable_variables())
                with tf.variable_scope('training_critic'):
                    self.critic_lr, self.critic_optimizer, self.critic_grads, self.critic_train_op, self.critic_global_step=ml_common.add_train_op(cost=self.critic_cost, learning_rate=self.critic_learning_rate, optimizer_getter=optimizer_getter, max_grad_value=max_grad_value, max_grad_norm=max_grad_norm, global_step=None, var_list=self.critic.get_trainable_variables())
                self.replace_actor_target_op=self.actor.get_replace_target_op()
                self.replace_critic_target_op=self.critic.get_replace_target_op()

    def get_continuous_lower_op(self):
        return self.actor.action_lower

    def get_continuous_upper_op(self):
        return self.actor.action_upper

    def replace_target(self, sess, iters):
        if self.running_type==ci.RunningType.train:
            sess.run(self.replace_actor_target_op)
            if not self.use_double or iters%self.replace_critic_target_span==0:
                sess.run(self.replace_critic_target_op)

    def run_sess_and_cost(self, sess, for_training, iters):
        self.replace_target(sess, iters)
        cost=0
        if for_training:
            for i in range(self.actor_update_num):
                _, actor_cost, updating_ctrl=sess.run([self.actor_train_op, self.actor_cost, self.actor_updating_ctrl_op])
                cost+=actor_cost
                if updating_ctrl==False:
                    i+=1
                    break
            sess.run(self.after_actor_update_op)
            cost/=i
            cost_critic=0
            for i in range(self.critic_update_num):
                _, critic_cost=sess.run([self.critic_train_op, self.critic_cost])
                cost_critic+=critic_cost
            cost+=cost_critic/self.critic_update_num
        else:
            actor_cost, critic_cost=sess.run([self.actor_cost, self.critic_cost])
            cost+=actor_cost+critic_cost
        return cost

    def run_sess_and_cost_for_placeholder(self, sess, for_training, iters, feed_dict):
        self.replace_target(sess, iters)
        cost=0
        if for_training:
            for i in range(self.actor_update_num):
                _, actor_cost, updating_ctrl=sess.run([self.actor_train_op, self.actor_cost, self.actor_updating_ctrl_op], feed_dict=feed_dict)
                cost+=actor_cost
                if updating_ctrl==False:
                    break
            sess.run(self.after_actor_update_op, feed_dict=feed_dict)
            cost/=i+1
            cost_critic=0
            for i in range(self.critic_update_num):
                _, critic_cost=sess.run([self.critic_train_op, self.critic_cost], feed_dict=feed_dict)
                cost_critic+=critic_cost
            cost+=cost_critic/self.critic_update_num
        else:
            actor_cost, critic_cost=sess.run([self.actor_cost, self.critic_cost], feed_dict=feed_dict)
            cost+=actor_cost+critic_cost
        return cost
