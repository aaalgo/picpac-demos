#!/usr/bin/env python
from __future__ import absolute_import, division, print_function
import tensorflow as tf
from tensorflow.contrib.slim.nets import resnet_v1

def resnet_v1_50 (inputs, is_training, num_classes):
    logits, _ = resnet_v1.resnet_v1_50(inputs, num_classes=num_classes, is_training=is_training)
    logits = tf.squeeze(logits, [1,2]) # resnet output is (N,1,1,C, remove the 
    return tf.identity(logits, name='logits')

def resnet_v1_101 (inputs, is_training, num_classes):
    logits, _ = resnet_v1.resnet_v1_101(inputs, num_classes)
    logits = tf.squeeze(logits, [1,2]) # resnet output is (N,1,1,C, remove the 
    return tf.identity(logits, name='logits')

'''
def inception (inputs, num_classes):
    from tensorflow.contrib.slim.nets import inception_v3
    logits, _ = inception_v3.inception_v3(inputs, num_classes)
    return tf.identity(logits, name='logits')
'''

def vgg16 (inputs, is_training, num_classes):
    from tensorflow.contrib.slim.nets import vgg
    logits, _ = vgg.vgg16(inputs, num_classes)
    return tf.identity(logits, name='logits')
