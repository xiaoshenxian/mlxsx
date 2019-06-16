# -*- coding: utf-8 -*-

import sys
import time

import tensorflow as tf
import numpy as np

class RLComponent:
    def get_eval_op(self):
        raise NotImplementedError('RLComponent::get_eval_op is not implemented!')

    def get_target_op(self):
        raise NotImplementedError('RLComponent::get_target_op is not implemented!')

    def get_replace_target_op(self):
        raise NotImplementedError('RLComponent::get_replace_target_op is not implemented!')

    def get_trainable_variables(self):
        raise NotImplementedError('RLComponent::get_trainable_variables is not implemented!')

class RL:
    def choose_action(self, sess, random_threshold=1, prob_random_threshold=1, random_sigma=None):
        action_prob_list, continuous_action_list, action_lower, action_upper=sess.run([self.discrete_action_op, self.continuous_action_op, [], []] if random_threshold>=1 or random_sigma is None else [self.discrete_action_op, self.continuous_action_op, self.get_continuous_lower_op(), self.get_continuous_upper_op()])
        discrete_action_list=[]
        for action_prob, num_actions in zip(action_prob_list, self.discrete_action_num_list):
            shape=action_prob.shape
            action_shape=shape[:-1]
            rand_score=np.random.uniform()
            action=np.where(rand_score<random_threshold
                          , np.argmax(action_prob, axis=-1)
                          , np.where(rand_score<random_threshold+(1-random_threshold)*prob_random_threshold
                                   , np.array([np.random.choice(num_actions, p=probs) for probs in np.reshape(action_prob, (-1, num_actions))], dtype='int32').reshape(action_shape)
                                   , np.random.randint(num_actions, size=np.prod(action_shape)).reshape(action_shape)))
            discrete_action_list.append(action)
        if random_sigma is not None:
            actions=continuous_action_list[0]
            shape=actions.shape
            rand=np.random.uniform(0, 1, size=np.prod(shape)).reshape(shape)
            idx=np.where(rand>=random_threshold)
            actions[idx]=np.random.normal(actions[idx], random_sigma)
            actions=np.transpose(actions, axes=[-1]+list(range(len(shape)-1)))
            sh=actions.shape
            actions=np.clip(actions.reshape((sh[0], -1)), action_lower.reshape((sh[0], 1)), action_upper.reshape((sh[0], 1)))
            actions=np.transpose(actions.reshape(sh), axes=list(range(1, len(sh)))+[0])
            continuous_action_list=[actions]
        return discrete_action_list+continuous_action_list, action_prob_list

    def get_discrete_prob(self, sess):
        return sess.run(self.discrete_action_op)

    def choose_action_for_placeholder(self, sess, feed_dict, random_threshold=1, prob_random_threshold=1, random_sigma=None):
        action_prob_list, continuous_action_list, action_lower, action_upper=sess.run([self.discrete_action_op, self.continuous_action_op, [], []] if random_threshold>=1 or random_sigma is None else [self.discrete_action_op, self.continuous_action_op, self.get_continuous_lower_op(), self.get_continuous_upper_op()], feed_dict=feed_dict)
        discrete_action_list=[]
        for action_prob, num_actions in zip(action_prob_list, self.discrete_action_num_list):
            shape=action_prob.shape
            action_shape=shape[:-1]
            rand_score=np.random.uniform()
            action=np.where(rand_score<random_threshold
                          , np.argmax(action_prob, axis=-1)
                          , np.where(rand_score<random_threshold+(1-random_threshold)*prob_random_threshold
                                   , np.array([np.random.choice(num_actions, p=probs) for probs in np.reshape(action_prob, (-1, num_actions))], dtype='int32').reshape(action_shape)
                                   , np.random.randint(num_actions, size=np.prod(action_shape)).reshape(action_shape)))
            discrete_action_list.append(action)
        if random_sigma is not None:
            actions=continuous_action_list[0]
            shape=actions.shape
            rand=np.random.uniform(0, 1, size=np.prod(shape)).reshape(shape)
            idx=np.where(rand>=random_threshold)
            actions[idx]=np.random.normal(actions[idx], random_sigma)
            actions=np.transpose(actions, axes=[-1]+list(range(len(shape)-1)))
            sh=actions.shape
            actions=np.clip(actions.reshape((sh[0], -1)), action_lower.reshape((sh[0], 1)), action_upper.reshape((sh[0], 1)))
            actions=np.transpose(actions.reshape(sh), axes=list(range(1, len(sh)))+[0])
            continuous_action_list=[actions]
        return discrete_action_list+continuous_action_list, action_prob_list

    def get_discrete_prob_for_placeholder(self, sess, feed_dict):
        return sess.run(self.discrete_action_op, feed_dict=feed_dict)

    def get_continuous_lower_op(self):
        raise NotImplementedError('RL::get_continuous_lower_op is not implemented!')

    def get_continuous_upper_op(self):
        raise NotImplementedError('RL::get_continuous_upper_op is not implemented!')

    def set_summary(self, sess, log_dir, verbose):
        self.summary=tf.summary.merge_all()
        self.summary_writer=tf.summary.FileWriter(log_dir, sess.graph)
        self.summary_verbose=verbose
        self.total_step=0

    def run_summary(self, sess):
        if self.summary is not None:
            if self.total_step%self.summary_verbose==0:
                summary_str=sess.run(self.summary)
                self.summary_writer.add_summary(summary_str, self.total_step)
                sys.stderr.write('summary wrote at total_step={}\n'.format(self.total_step))
            self.total_step+=1

    def run_summary_for_placeholder(self, sess, feed_dict):
        if self.summary is not None:
            if self.total_step%self.summary_verbose==0:
                summary_str=sess.run(self.summary, feed_dict=feed_dict)
                self.summary_writer.add_summary(summary_str, self.total_step)
                sys.stderr.write('summary wrote at total_step={}\n'.format(self.total_step))
            self.total_step+=1

    def run_sess_and_cost(self, sess, for_training, iters):
        self.replace_target(sess, iters)
        res=sess.run([self.cost, self.train_op] if for_training else [self.cost])
        return res[0]

    def run_sess_and_cost_for_placeholder(self, sess, for_training, iters, feed_dict):
        self.replace_target(sess, iters)
        res=sess.run([self.cost, self.train_op] if for_training else [self.cost], feed_dict=feed_dict)
        return res[0]

    def replace_target(self, sess, iters):
        raise NotImplementedError('RL::replace_target is not implemented!')

    def run_epoch(self, sess, for_training, verbose=-1):
        start_time=time.time()
        costs=0.0
        iters=0
        try:
            while True:
                self.run_summary(sess)

                cost=self.run_sess_and_cost(sess, for_training, iters)
                costs+=cost
                iters+=1

                if verbose>=0 and iters%verbose==0:
                    sys.stderr.write('step {0} avg cost: {1:.3f} current cost: {2:.3f} speed: {3:.0f} sps\n'.format(iters, costs/iters, cost, iters*self.batch_size/max(time.time()-start_time, 1)))
        except tf.errors.OutOfRangeError:
            pass

        return costs/(iters if iters!=0 else 1)

    def run_training(self, sess, verbose=-1):
        return self.run_epoch(sess, self.train_op, verbose)

    def run_epoch_for_placeholder(self, sess, data_iterator, for_training, verbose=-1):
        start_time=time.time()
        costs=0.0
        iters=0
        for step, (feed_dict) in enumerate(data_iterator):
            self.run_summary(sess)

            cost=self.run_sess_and_cost_for_placeholder(sess, for_training, iters, feed_dict)
            costs+=cost
            iters+=1

            if verbose>=0 and step%verbose==0:
                print("step {0} avg cost: {1:.3f} current cost: {2:.3f} speed: {3:.0f} sps".format(step, costs/iters, cost, iters*self.batch_size/(time.time()-start_time)))

        return costs/(iters if iters!=0 else 1)

    def run_training_for_placeholder(self, sess, data_iterator, verbose=-1):
        return self.run_epoch_for_placeholder(sess, self.train_op, data_iterator, verbose)
