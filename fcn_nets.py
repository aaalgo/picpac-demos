#!/usr/bin/env python
from __future__ import absolute_import, division, print_function
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim.nets import resnet_v1

def woshialex (X, is_training, num_classes):
    # Data Science Bowl 2016 winning solution
    net = X
    with slim.arg_scope([slim.conv2d, slim.conv2d_transpose, slim.max_pool2d], padding='SAME'):
        with arg_scope([layers.batch_norm], is_training=is_training):
            net = slim.batch_norm(slim.conv2d(net, 8, 5, 1))
            net = slim.batch_norm(slim.conv2d(net, 16, 3, 1))
            net = slim.max_pool2d(net, 2, 2)
            net = slim.batch_norm(slim.conv2d(net, 32, 4, 1))
            net = slim.max_pool2d(net, 2, 2)
            net = slim.batch_norm(slim.conv2d(net, 64, 4, 1))
            net = slim.max_pool2d(net, 2, 2)
            net = slim.batch_norm(slim.conv2d(net, 128, 5, 1))
            net = slim.batch_norm(slim.conv2d(net, 64, 5, 1))
            net = slim.batch_norm(slim.conv2d_transpose(net, 32, 5, 2))
            net = slim.batch_norm(slim.conv2d_transpose(net, 16, 5, 2))
            net = slim.batch_norm(slim.conv2d_transpose(net, 8, 5, 2))
            net = slim.conv2d(net, num_classes, 5, 1, activation_fn=None) 
    return tf.identity(net, 'logits'), 8

def  resnet_v1_50 (X, is_training, num_classes):
    with slim.arg_scope([slim.conv2d, slim.conv2d_transpose, slim.max_pool2d], padding='SAME'):
            net, _ = resnet_v1.resnet_v1_50(X,
                                    num_classes=num_classes,
                                    is_training=is_training,
                                    global_pool = False,
                                    output_stride = 16)
            net = slim.conv2d_transpose(net, num_classes, 31, 16, scope='upscale')
    net = tf.identity(net, 'logits')
    return net, 16

def myunet (X, is_training, num_classes):
    net = X
    stack = []
    with slim.arg_scope([slim.conv2d, slim.conv2d_transpose, slim.max_pool2d], padding='SAME'):
        with arg_scope([layers.batch_norm], is_training=is_training):
            net = slim.batch_norm(slim.conv2d(net, 16, 5, 2))
            stack.append(net)       # 1/2
            net = slim.batch_norm(slim.conv2d(net, 32, 3, 1))
            net = slim.max_pool2d(net, 2, 2)
            stack.append(net)       # 1/4
            net = slim.batch_norm(slim.conv2d(net, 64, 4, 1))
            net = slim.max_pool2d(net, 2, 2)
            stack.append(net)       # 1/8
            net = slim.batch_norm(slim.conv2d(net, 128, 4, 1))
            net = slim.max_pool2d(net, 2, 2)
                                    # 1/16
            net = slim.batch_norm(slim.conv2d(net, 256, 5, 1))
            net = slim.batch_norm(slim.conv2d(net, 128, 5, 1))
            net = slim.batch_norm(slim.conv2d_transpose(net, 64, 5, 2))
                                    # 1/8
            net = tf.concat([net, stack.pop()], 3)
            net = slim.batch_norm(slim.conv2d_transpose(net, 32, 5, 2))
                                    # 1/4
            net = tf.concat([net, stack.pop()], 3)
            net = slim.batch_norm(slim.conv2d_transpose(net, 16, 5, 2))
            net = tf.concat([net, stack.pop()], 3)
            net = slim.batch_norm(slim.conv2d_transpose(net, 8, 5, 2))
            net = slim.conv2d(net, num_classes, 5, 1, activation_fn=None) 
            assert len(stack) == 0
    return tf.identity(net, 'logits'), 16

