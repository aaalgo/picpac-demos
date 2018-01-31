#!/usr/bin/env python
from __future__ import absolute_import, division, print_function
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import time
import logging
from tqdm import tqdm
import numpy as np
import cv2
import tensorflow as tf
import tensorflow.contrib.slim as slim
import picpac
import fcn_nets

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('db', 'db', '')
flags.DEFINE_string('mixin', None, '')
flags.DEFINE_string('net', 'resnet_v1_50', '')
flags.DEFINE_string('model', 'model', 'Directory to put the training data.')

flags.DEFINE_integer('channels', 3, '')
flags.DEFINE_integer('verbose', logging.INFO, '')
# optimizer settings
flags.DEFINE_float('learning_rate', 0.0001, 'Initial learning rate.')
flags.DEFINE_float('decay_rate', 0.9, '')
flags.DEFINE_float('decay_steps', 10000, '')
#
flags.DEFINE_string('resume', None, '')
flags.DEFINE_integer('max_steps', 200000, '')
flags.DEFINE_integer('epoch_steps', 100, '')
flags.DEFINE_integer('ckpt_epochs', 200, '')
flags.DEFINE_integer('max_size', None, '')
flags.DEFINE_integer('max_to_keep', 100, '')

def fcn_loss (logits, labels):
    # to HWC
    logits = tf.reshape(logits, (-1, 2))
    labels = tf.reshape(labels, (-1,))
    xe = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=tf.to_int32(labels))
    xe = tf.reduce_mean(xe, name='xe')
    return xe, [xe]

def main (_):
    logging.basicConfig(level=FLAGS.verbose)
    try:
        os.makedirs(FLAGS.model)
    except:
        pass

    X = tf.placeholder(tf.float32, shape=(None, None, None, FLAGS.channels), name="images")
    Y = tf.placeholder(tf.float32, shape=(None, None, None, 1), name="labels")

    logits, stride = getattr(fcn_nets, FLAGS.net)(X)

    loss, metrics = fcn_loss(logits, Y)

    #tf.summary.scalar("loss", loss)
    metric_names = [x.name[:-2] for x in metrics]

    global_step = tf.Variable(0, name='global_step', trainable=False)
    rate = FLAGS.learning_rate
    rate = tf.train.exponential_decay(rate, global_step, FLAGS.decay_steps, FLAGS.decay_rate, staircase=True)
    optimizer = tf.train.AdamOptimizer(rate)

    train_op = optimizer.minimize(loss, global_step=global_step)

    picpac_config = dict(seed=2016,
                max_size=320,
                shuffle=True,
                reshuffle=True,
                batch=1,
                split=1,
                split_fold=0,
                round_div=stride,
                annotate='json',
                channels=FLAGS.channels,
                stratify=True,
                pert_color1=20,
                pert_color2=20,
                pert_color3=20,
                pert_angle=20,
                pert_min_scale=0.9,
                pert_max_scale=1.5,
                pert_hflip=True,
                pert_vflip=True,
                channel_first=False # this is tensorflow specific
                                    # Caffe's dimension order is different.
                )
    assert FLAGS.db and os.path.exists(FLAGS.db)
    if not FLAGS.mixin is None:
        assert os.path.exists(FLAGS.mixin)
        picpac_config['mixin'] = FLAGS.mixin
        picpac_config['mixin_group_delta'] = 1
        pass

    stream = picpac.ImageStream(FLAGS.db, perturb=True, loop=True, **picpac_config)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver(max_to_keep=FLAGS.max_to_keep)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True

    with tf.Session(config=config) as sess:
        sess.run(init)
        if FLAGS.resume:
            saver.restore(sess, FLAGS.resume)
        step = 0
        epoch = 0
        global_start_time = time.time()
        while step < FLAGS.max_steps:
            start_time = time.time()
            avg = np.array([0] * len(metrics), dtype=np.float32)
            for _ in tqdm(range(FLAGS.epoch_steps), leave=False):
                images, labels, _ = stream.next()
                feed_dict = {X: images, Y: labels}
                mm, _, = sess.run([metrics, train_op, ], feed_dict=feed_dict)
                avg += np.array(mm)
                step += 1
                pass
            avg /= FLAGS.epoch_steps
            stop_time = time.time()
            txt = ', '.join(['%s=%.4f' % (a, b) for a, b in zip(metric_names, list(avg))])
            print('step %d: elapsed=%.4f time=%.4f, %s'
                    % (step, (stop_time - global_start_time), (stop_time - start_time), txt))
            epoch += 1
            if epoch and (epoch % FLAGS.ckpt_epochs == 0):
                ckpt_path = '%s/%d' % (FLAGS.model, step)
                saver.save(sess, ckpt_path)
                print('epoch %d step %d, saving to %s.' % (epoch, step, ckpt_path))
            pass
        pass
    pass

if __name__ == '__main__':
    tf.app.run()

