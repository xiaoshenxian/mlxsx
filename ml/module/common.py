# -*- coding: utf-8 -*-

import tensorflow as tf

def add_gradient_op(cost, learning_rate, optimizer_getter=lambda lr : tf.train.GradientDescentOptimizer(lr), max_grad_value=-1, max_grad_norm=-1, var_list=None):
    lr=tf.get_variable('learning_rate', shape=[], dtype=tf.float32, initializer=tf.constant_initializer(learning_rate), trainable=False)
    optimizer=optimizer_getter(lr)
    grads, vars=zip(*optimizer.compute_gradients(cost, var_list=var_list))
    if max_grad_value>0:
        grads=[tf.IndexedSlices(tf.clip_by_value(g.values, -max_grad_value, max_grad_value), g.indices, g.dense_shape) if isinstance(g, tf.IndexedSlices) else tf.clip_by_value(g, -max_grad_value, max_grad_value) for g in grads]
    if max_grad_norm>0:
        grads, _=tf.clip_by_global_norm(grads, max_grad_norm)
    grads=list(zip(grads, vars))
    return lr, optimizer, grads

def add_apply_gradient_op(optimizer, grads, global_step=None):
    if global_step is None:
        global_step=tf.get_variable('global_step', shape=[], dtype=tf.int32, initializer=tf.constant_initializer(0), trainable=False)
    train_op=optimizer.apply_gradients(grads, global_step=global_step)
    return train_op, global_step

def add_train_op(cost, learning_rate, optimizer_getter=lambda lr : tf.train.GradientDescentOptimizer(lr), max_grad_value=-1, max_grad_norm=-1, global_step=None, var_list=None):
    lr, optimizer, grads=add_gradient_op(cost=cost, learning_rate=learning_rate, optimizer_getter=optimizer_getter, max_grad_value=max_grad_value, max_grad_norm=max_grad_norm, var_list=var_list)
    train_op, global_step=add_apply_gradient_op(optimizer=optimizer, grads=grads, global_step=global_step)
    return lr, optimizer, grads, train_op, global_step
