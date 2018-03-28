#!/usr/bin/env python3
import os
import sys
sys.path.append('install/picpac/build/lib.linux-x86_64-%d.%d' % sys.version_info[:2])
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import time
from tqdm import tqdm
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.contrib import layers
import tensorflow.contrib.slim as slim
from tensorflow.python.ops import control_flow_ops

import picpac
import cls_nets as nets
print(picpac.__file__)

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('db', 'cifar10-train.picpac', 'training db')
flags.DEFINE_string('val_db', 'cifar10-test.picpac', 'validation db')
flags.DEFINE_integer('classes', 2, 'number of classes')
flags.DEFINE_string('mixin', None, 'mix-in training db')
flags.DEFINE_integer('channels', 3, '')

flags.DEFINE_integer('resize', 224, '') 
flags.DEFINE_integer('shift', 4, '')
flags.DEFINE_integer('batch', 128, 'Batch size.  ')

flags.DEFINE_string('net', 'resnet_v1_50', 'architecture')
flags.DEFINE_string('model', 'cls_model', 'model directory')
flags.DEFINE_string('resume', None, 'resume training from this model')
flags.DEFINE_integer('max_to_keep', 100, '')

# optimizer settings
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_float('decay_rate', 0.95, '')
flags.DEFINE_float('decay_steps', 500, '')
#
flags.DEFINE_integer('max_epochs', 200, '')
flags.DEFINE_integer('ckpt_epochs', 10, '')
flags.DEFINE_integer('val_epochs', 1, '')


def cls_loss (logits, labels):

    # cross-entropy
    xe = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
    xe = tf.reduce_mean(xe, name='xe')
    # accuracy
    acc = tf.cast(tf.nn.in_top_k(logits, labels, 1), tf.float32)
    acc = tf.reduce_mean(acc, name='acc')
    # regularization
    reg = tf.reduce_sum(tf.losses.get_regularization_losses())
    reg = tf.identity(reg, name='re')
    # loss
    loss = tf.identity(xe + reg, name='lo')
    return loss, [acc, xe, reg, loss]

def main (_):

    try:
        os.makedirs(FLAGS.model)
    except:
        pass

    X = tf.placeholder(tf.float32, shape=(None, None, None, FLAGS.channels), name="images")
    # ground truth labels
    Y = tf.placeholder(tf.int32, shape=(None, ), name="labels")
    is_training = tf.placeholder(tf.bool, name="is_training")

    # load network
    with slim.arg_scope([slim.conv2d], weights_regularizer=slim.l2_regularizer(2.5e-4)), \
         slim.arg_scope([slim.batch_norm], decay=0.9, epsilon=5e-4): 
        logits = getattr(nets, FLAGS.net)(X, is_training, FLAGS.classes)

    loss, metrics = cls_loss(logits, Y)

    metric_names = [x.name[:-2] for x in metrics]

    def format_metrics (avg):
        return ' '.join(['%s=%.3f' % (a, b) for a, b in zip(metric_names, list(avg))])

    global_step = tf.train.create_global_step()
    rate = FLAGS.learning_rate
    rate = tf.train.exponential_decay(rate, global_step, FLAGS.decay_steps, FLAGS.decay_rate, staircase=True)
    optimizer = tf.train.MomentumOptimizer(learning_rate=rate, momentum=0.9)
    #optimizer = tf.train.AdamOptimizer(0.0001)

    #train_op = optimizer.minimize(loss, global_step=global_step)
    train_op = slim.learning.create_train_op(loss, optimizer, global_step=global_step)
    #init = tf.initialize_all_variables()
    saver = tf.train.Saver(max_to_keep=FLAGS.max_to_keep)

    picpac_config = {"db": FLAGS.db,
              "loop": True,
              "shuffle": True,
              "reshuffle": True,
              "annotate": False,
              "channels": FLAGS.channels,
              "stratify": True,
              "dtype": "float32",
              "batch": FLAGS.batch,
              "transforms": [
                  {"type": "augment.flip", "horizontal": True, "vertical": False},
                  {"type": "normalize", "mean": 127, "std": 127},
                  #{"type": "augment.rotate", "min":-10, "max":10},
                  #{"type": "augment.scale", "min":0.7, "max":1.2},
                  {"type": "resize", "size": FLAGS.resize},
                  {"type": "clip", "size": FLAGS.resize, "shift": FLAGS.shift, "border_type": "replicate"},
                  #{"type": "colorspace", "code": "BGR2HSV", "mul0": 1.0/255},
                  #{"type": "augment.add", "range":20},
                  #{"type": "colorspace", "code": "HSV2BGR", "mul1": 255.0},
              ]
             }

    if not FLAGS.mixin is None:
        print("mixin support is incomplete in new picpac.")
    #    assert os.path.exists(FLAGS.mixin)
    #    picpac_config['mixin'] = FLAGS.mixin
    #    picpac_config['mixin_group_delta'] = 1
    #    pass

    # do we want to apply below to validation images?
    # load training db
    assert FLAGS.db and os.path.exists(FLAGS.db)
    stream = picpac.ImageStream(picpac_config)

    # load validation db
    val_stream = None
    if FLAGS.val_db:
        val_config = {"db": FLAGS.val_db,
                  "loop": False,
                  "channels": FLAGS.channels,
                  "dtype": "float32",
                  "batch": FLAGS.batch,
                  "transforms": [
                        {"type": "normalize", "mean": 127, "std": 127},
                        {"type": "resize", "size": FLAGS.resize}
                  ]
                 }
        assert os.path.exists(FLAGS.val_db)
        val_stream = picpac.ImageStream(val_config)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True

    epoch_steps = (stream.size() + FLAGS.batch-1) // FLAGS.batch
    best = 0
    with tf.Session(config=config) as sess, open('cls-train-cifar10.log', 'w') as log:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        if FLAGS.resume:
            saver.restore(sess, FLAGS.resume)
        step = 0
        epoch = 0
        global_start_time = time.time()

        while epoch < FLAGS.max_epochs:
            start_time = time.time()
            avg = np.array([0] * len(metrics), dtype=np.float32)
            cnt = 0
            progress = tqdm(range(epoch_steps), leave=False)
            for _ in progress:
                meta, image = stream.next()
                feed_dict = {X: image, Y: meta.labels, is_training: True}
                mm, _, = sess.run([metrics, train_op, ], feed_dict=feed_dict)
                avg += np.array(mm) * image.shape[0]
                cnt += image.shape[0]
                progress.set_description(format_metrics(avg/cnt))
                step += 1
                pass
            avg /= cnt
            stop_time = time.time()
            print('epoch=%d step=%d elapsed=%.4f time=%.4f %s'
                    % (epoch, step, (stop_time - global_start_time), (stop_time - start_time),
                       format_metrics(avg)))
            epoch += 1

            # validation
            std = None
            if epoch and (epoch % FLAGS.val_epochs == 0) and not val_stream is None:
                lr = sess.run(rate)
                # evaluation
                val_stream.reset()
                avg = np.array([0] * len(metrics), dtype=np.float32)
                cnt = 0
                checksum = []
                for meta, image in val_stream:
                    feed_dict = {X: image, Y: meta.labels, is_training: False}
                    mm = sess.run(metrics, feed_dict=feed_dict)
                    avg += np.array(mm) * image.shape[0]
                    cnt += image.shape[0]
                    checksum.append(np.sum(image))
                    pass
                avg /= cnt
                checksum = np.array(checksum)
                if std is None:
                    std = checksum

                if avg[0] > best:
                    best = avg[0]
                print('validation %s' % format_metrics(avg), "best", best, 'lr', lr)
                log.write('%d\t%s\t%.4f\n' % (epoch, '\t'.join(['%.4f' % x for x in avg]), best))

            # model saving
            if epoch and (epoch % FLAGS.ckpt_epochs == 0):
                ckpt_path = '%s/%d' % (FLAGS.model, epoch)
                saver.save(sess, ckpt_path)
                print('step %d, saving to %s.' % (step, ckpt_path))
            pass
        pass
    pass

if __name__ == '__main__':
    tf.app.run()

