#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim.nets import resnet_v1
from tensorflow.contrib.slim.nets import resnet_utils

def tiny (X, num_classes=2):
    # stride is  2 * 2 * 2 * 2 = 16
    net = X
    layers = [X]
    with tf.name_scope('simple'):
        # slim.arg_scope([slim.conv2d]):
        # slim.conv2d defaults:
        #   padding = 'SAME'
        #   activation_fn = nn.relu
        # parameters: net, out_channels, kernel_size, stride
        net = slim.conv2d(net, 64, 3, 2, scope='conv1')
        net = slim.max_pool2d(net, 2, 2, scope='pool1')
        net = slim.conv2d(net, 128, 3, 1, scope='conv2_1')
        net = slim.conv2d(net, 128, 3, 1, scope='conv2_2')
        net = slim.max_pool2d(net, 2, 2, scope='pool2')
        net = slim.conv2d(net, 256, 3, 1, scope='conv3_1')
        net = slim.conv2d(net, 256, 3, 1, scope='conv3_2')
        net = slim.conv2d(net, 128, 1, 1, scope='conv5')
        #net = slim.dropout(net, keep_prob=0.9, scope='dropout')
        net = slim.conv2d(net, 32, 1, 1, scope='conv6',
                            activation_fn=None,
                            normalizer_fn=None,
                         )
        net = slim.conv2d_transpose(net, num_classes, 17, 8, scope='upscale')
    net = tf.identity(net, 'logits')
    return net, 8

def tiny2 (X, num_classes=2):
    # stride is  2 * 2 * 2 * 2 = 16
    net = X
    layers = [X]
    with tf.name_scope('simple'):
        # slim.arg_scope([slim.conv2d]):
        # slim.conv2d defaults:
        #   padding = 'SAME'
        #   activation_fn = nn.relu
        # parameters: net, out_channels, kernel_size, stride
        net = slim.batch_norm(slim.conv2d(net, 64, 3, 2, scope='conv1'))
        net = slim.max_pool2d(net, 2, 2, scope='pool1')
        net = slim.batch_norm(slim.conv2d(net, 128, 3, 1, scope='conv2_1'))
        net = slim.batch_norm(slim.conv2d(net, 128, 3, 1, scope='conv2_2'))
        net = slim.max_pool2d(net, 2, 2, scope='pool2')
        net = slim.batch_norm(slim.conv2d(net, 256, 3, 1, scope='conv3_1'))
        net = slim.batch_norm(slim.conv2d(net, 256, 3, 1, scope='conv3_2'))
        net = slim.batch_norm(slim.conv2d(net, 128, 1, 1, scope='conv5'))
        #net = slim.dropout(net, keep_prob=0.9, scope='dropout')
        net = slim.batch_norm(slim.conv2d(net, 32, 1, 1, scope='conv6'))
        net = slim.batch_norm(slim.conv2d_transpose(net, 16, 9, 4, scope='upscale1'))
        net = slim.batch_norm(slim.conv2d_transpose(net, 8, 5, 2, scope='upscale2'))
        net = slim.conv2d(net, num_classes, 3, 1, scope='final',
                            activation_fn=None,
                            normalizer_fn=None,
                         )
    net = tf.identity(net, 'logits')
    return net, 8

def woshialex (X, num_classes=2):
    net = X
    with tf.name_scope('woshialex'):
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

def superalex (X, num_classes=2):
    net = X
    with tf.name_scope('superalex'):
        net = slim.batch_norm(slim.conv2d(net, 16, 5, 2))
        net = slim.batch_norm(slim.conv2d(net, 32, 3, 1))
        net = slim.max_pool2d(net, 2, 2)
        net = slim.batch_norm(slim.conv2d(net, 64, 4, 1))
        net = slim.max_pool2d(net, 2, 2)
        net = slim.batch_norm(slim.conv2d(net, 128, 4, 1))
        net = slim.max_pool2d(net, 2, 2)
        net = slim.batch_norm(slim.conv2d(net, 256, 5, 1))
        net = slim.batch_norm(slim.conv2d(net, 128, 5, 1))
        net = slim.batch_norm(slim.conv2d_transpose(net, 64, 5, 2))
        net = slim.batch_norm(slim.conv2d_transpose(net, 32, 5, 2))
        net = slim.batch_norm(slim.conv2d_transpose(net, 16, 5, 2))
        net = slim.batch_norm(slim.conv2d_transpose(net, 8, 5, 2))
        net = slim.conv2d(net, num_classes, 5, 1, activation_fn=None) 
    return tf.identity(net, 'logits'), 16


# conv2d and conv2d_transpose

# conv2d output size if padding = 'SAME':   W <- (W + S -1)/S 
#                                 'VALID':  W <- (W - F + S)/S
def simple (X, num_classes=2):
    # stride is  2 * 2 * 2 * 2 = 16
    net = X
    layers = [X]
    with tf.name_scope('simple'):
        # slim.arg_scope([slim.conv2d]):
        # slim.conv2d defaults:
        #   padding = 'SAME'
        #   activation_fn = nn.relu
        # parameters: net, out_channels, kernel_size, stride
        net = slim.conv2d(net, 100, 5, 2, scope='conv1')
        net = slim.max_pool2d(net, 2, 2, scope='pool1')
        net = slim.conv2d(net, 200, 5, 2, scope='conv2')
        net = slim.max_pool2d(net, 2, 2, scope='pool2')
        net = slim.conv2d(net, 300, 3, 1, scope='conv3')
        net = slim.conv2d(net, 300, 3, 1, scope='conv4')
        net = slim.dropout(net, keep_prob=0.9, scope='dropout')
        net = slim.conv2d(net, 20, 1, 1, scope='layer5')
        net = slim.conv2d_transpose(net, num_classes, 31, 16, scope='upscale')
    net = tf.identity(net, 'logits')
    return net, 16

def  resnet_v1_50 (X, num_classes=2):
    with slim.arg_scope([slim.conv2d, slim.conv2d_transpose, slim.max_pool2d], padding='SAME'):
        net, _ = resnet_v1.resnet_v1_50(X,
                                num_classes=num_classes,
                                global_pool = False,
                                output_stride = 16)
        net = slim.conv2d_transpose(net, num_classes, 31, 16, scope='upscale')
    net = tf.identity(net, 'logits')
    return net, 16

def myunet (X, num_classes=2):
    net = X
    stack = []
    with tf.name_scope('myunet'):
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

def mytinyunet (X, num_classes=2):
    net = X
    stack = []
    with tf.name_scope('myunet'):
        net = slim.batch_norm(slim.conv2d(net, 8, 5, 2))
        stack.append(net)       # 1/2
        net = slim.batch_norm(slim.conv2d(net, 16, 3, 1))
        net = slim.max_pool2d(net, 2, 2)
        stack.append(net)       # 1/4
        net = slim.batch_norm(slim.conv2d(net, 32, 4, 1))
        net = slim.max_pool2d(net, 2, 2)
        stack.append(net)       # 1/8
        net = slim.batch_norm(slim.conv2d(net, 64, 4, 1))
        net = slim.max_pool2d(net, 2, 2)
                                # 1/16
        net = slim.batch_norm(slim.conv2d(net, 128, 5, 1))
        net = slim.batch_norm(slim.conv2d(net, 128, 5, 1))
        net = slim.batch_norm(slim.conv2d_transpose(net, 64, 5, 2))
                                # 1/8
        net = tf.concat(3, [net, stack.pop()])
        net = slim.batch_norm(slim.conv2d_transpose(net, 32, 5, 2))
                                # 1/4
        net = tf.concat(3, [net, stack.pop()])
        net = slim.batch_norm(slim.conv2d_transpose(net, 16, 5, 2))
        net = tf.concat(3, [net, stack.pop()])
        net = slim.batch_norm(slim.conv2d_transpose(net, 8, 5, 2))
        net = slim.conv2d(net, num_classes, 5, 1, activation_fn=None) 
        assert len(stack) == 0
    return tf.identity(net, 'logits'), 16

