# -*- coding: utf-8 -*-

import tensorflow as tf

class TargetOp:
    def build_mid(self, input_data, initializer=None, regularizer=None, name=None):
        pass

    def build_training(self, input_data, initializer=None, regularizer=None, name=None):
        pass

    def build_inference(self, input_data, initializer=None, regularizer=None, name=None):
        pass

    def _get_init(self, initializer=None, regularizer=None):
        scope=tf.get_variable_scope()
        return (scope.initializer if initializer is None else initializer), (scope.regularizer if regularizer is None else regularizer)

class RegressionTargetOp(TargetOp):
    def __init__(self, build_mid_func=None):
        self.build_mid_func=build_mid_func

    def build_mid(self, input_data, initializer=None, regularizer=None, name=None):
        initializer, regularizer=self._get_init(initializer, regularizer)
        if self.build_mid_func is None:
            return tf.layers.dense(input_data, 1, kernel_initializer=initializer, kernel_regularizer=regularizer, name=name)
        else:
            return self.build_mid_func(input_data, initializer, regularizer, name=name)

    def build_training(self, input_data, initializer=None, regularizer=None, name=None):
        targets, predicts=input_data
        return tf.square(predicts-targets, name=name)

    def build_inference(self, input_data, initializer=None, regularizer=None, name=None):
        return input_data

class LrTargetOp(TargetOp):
    def __init__(self, build_mid_func=None):
        self.build_mid_func=build_mid_func

    def build_mid(self, input_data, initializer=None, regularizer=None, name=None):
        initializer, regularizer=self._get_init(initializer, regularizer)
        if self.build_mid_func is None:
            return tf.layers.dense(input_data, 1, kernel_initializer=initializer, kernel_regularizer=regularizer, name=name)
        else:
            return self.build_mid_func(input_data, initializer, regularizer, name=name)

    def build_training(self, input_data, initializer=None, regularizer=None, name=None):
        targets, logits=input_data
        return tf.nn.sigmoid_cross_entropy_with_logits(labels=targets, logits=logits, name=name)

    def build_inference(self, input_data, initializer=None, regularizer=None, name=None):
        return tf.nn.sigmoid(input_data, name=name)

class SoftmaxTargetOp(TargetOp):
    def __init__(self, result_shape, top_k, sparse_target=False, build_mid_func=None):
        self.result_shape=result_shape
        self.top_k=top_k
        self.sparse_target=sparse_target
        self.build_mid_func=build_mid_func

    def build_mid(self, input_data, initializer=None, regularizer=None, name=None):
        initializer, regularizer=self._get_init(initializer, regularizer)
        if self.build_mid_func is None:
            return tf.layers.dense(input_data, self.result_shape, kernel_initializer=initializer, kernel_regularizer=regularizer, name=name)
        else:
            return self.build_mid_func(input_data, initializer, regularizer, name=name)

    def build_training(self, input_data, initializer=None, regularizer=None, name=None):
        targets, logits=input_data
        if self.sparse_target:
            return tf.nn.sparse_softmax_cross_entropy_with_logits(labels=targets, logits=logits, name=name)
        else:
            return tf.nn.softmax_cross_entropy_with_logits(labels=targets, logits=logits, name=name)

    def build_inference(self, input_data, initializer=None, regularizer=None, name=None):
        val, idx=tf.nn.top_k(tf.nn.softmax(input_data, name=name), k=self.top_k)
        return val, idx

class RankNetTargetOp(TargetOp):
    def __init__(self, build_mid_func=None):
        self.build_mid_func=build_mid_func

    def build_mid(self, input_data, initializer=None, regularizer=None, name=None):
        initializer, regularizer=self._get_init(initializer, regularizer)
        if self.build_mid_func is None:
            return tf.layers.dense(input_data, 1, kernel_initializer=initializer, kernel_regularizer=regularizer, name=name)
        else:
            return self.build_mid_func(input_data, initializer, regularizer, name=name)

    def build_training(self, input_data, initializer=None, regularizer=None, name=None):
        targets, logits=input_data
        self.targets=tf.subtract(targets[:-1], targets[1:])
        self.targets=tf.truediv(1.0, 1.0+tf.exp(-self.targets), name='targets')
        self.logits=tf.subtract(logits[:-1], logits[1:], name='logits')
        return tf.nn.sigmoid_cross_entropy_with_logits(labels=self.targets, logits=self.logits, name=name)

    def build_inference(self, input_data, initializer=None, regularizer=None, name=None):
        return tf.sigmoid(input_data, name=name)
