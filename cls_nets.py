#!/usr/bin/env python
from __future__ import absolute_import, division, print_function
import tensorflow as tf
from tensorflow.contrib.slim.nets import resnet_v2
#import resnet_v2

def resnet_v2_18_impl (inputs,
                 num_classes=None,
                 is_training=True,
                 global_pool=True,
                 output_stride=None,
                 reuse=None,
                 scope='resnet_v2_18'):
  resnet_v2_block = resnet_v2.resnet_v2_block
  blocks = [
      resnet_v2_block('block1', base_depth=64, num_units=2, stride=2),
      resnet_v2_block('block2', base_depth=128, num_units=2, stride=2),
      resnet_v2_block('block3', base_depth=256, num_units=2, stride=2),
      resnet_v2_block('block4', base_depth=512, num_units=2, stride=1),
  ]
  return resnet_v2.resnet_v2(
      inputs,
      blocks,
      num_classes,
      is_training,
      global_pool,
      output_stride,
      include_root_block=False,
      reuse=reuse,
      scope=scope)

def resnet_18 (inputs, is_training, num_classes):
    logits, _ = resnet_v2_18_impl(inputs, num_classes=num_classes, is_training=is_training)
    logits = tf.squeeze(logits, [1,2]) # resnet output is (N,1,1,C, remove the 
    return tf.identity(logits, name='logits')

def resnet_50 (inputs, is_training, num_classes):
    logits, _ = resnet_v2.resnet_v2_50(inputs, num_classes=num_classes, is_training=is_training)
    logits = tf.squeeze(logits, [1,2]) # resnet output is (N,1,1,C, remove the 
    return tf.identity(logits, name='logits')

def resnet_101 (inputs, is_training, num_classes):
    logits, _ = resnet_v2.resnet_v2_101(inputs, num_classes)
    logits = tf.squeeze(logits, [1,2]) # resnet output is (N,1,1,C, remove the 
    return tf.identity(logits, name='logits')

'''
def inception (inputs, num_classes):
    from tensorflow.contrib.slim.nets import inception_v2
    logits, _ = inception_v2.inception_v2(inputs, num_classes)
    return tf.identity(logits, name='logits')
'''

def vgg16 (inputs, is_training, num_classes):
    from tensorflow.contrib.slim.nets import vgg
    print(vgg.__file__)
    logits, _ = vgg.vgg_d(inputs, num_classes)
    return tf.identity(logits, name='logits')
