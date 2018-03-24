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
import picpac
import cls_nets as nets
print(picpac.__file__)

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('db', 'db', 'training db')
flags.DEFINE_string('val_db', None, 'validation db')
flags.DEFINE_integer('classes', 2, 'number of classes')
flags.DEFINE_string('mixin', None, 'mix-in training db')
flags.DEFINE_integer('channels', 3, '')
flags.DEFINE_integer('max_size', None, '')

flags.DEFINE_integer('resize_width', None, '')
flags.DEFINE_integer('resize_height', None, '')
flags.DEFINE_integer('batch', 1, 'Batch size.  ')

flags.DEFINE_string('net', 'resnet_v1_50', 'architecture')
flags.DEFINE_string('model', 'cls_model', 'model directory')
flags.DEFINE_string('resume', None, 'resume training from this model')
flags.DEFINE_integer('max_to_keep', 100, '')

# optimizer settings
flags.DEFINE_float('learning_rate', 0.0001, 'Initial learning rate.')
flags.DEFINE_float('decay_rate', 0.9, '')
flags.DEFINE_float('decay_steps', 10000, '')
#
flags.DEFINE_integer('max_steps', 200000, '')
flags.DEFINE_integer('epoch_steps', 100, '')
flags.DEFINE_integer('ckpt_epochs', 100, '')
flags.DEFINE_integer('val_epochs', 100, '')


def cls_loss (logits, labels):

    labels = tf.to_int32(labels)
    logits = tf.reshape(logits, (-1, FLAGS.classes))
    labels = tf.reshape(labels, (-1,))
    xe = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
    xe = tf.reduce_mean(xe, name='xe')

    acc = tf.cast(tf.nn.in_top_k(logits, labels, 1), tf.float32)
    acc = tf.reduce_mean(acc, name='acc')
    # optimize for cross-entropy
    # display cross entropy and accuracy
    return xe, [xe, acc]

def main (_):

    try:
        os.makedirs(FLAGS.model)
    except:
        pass

    X = tf.placeholder(tf.float32, shape=(None, None, None, FLAGS.channels), name="images")
    # ground truth labels
    Y = tf.placeholder(tf.float32, shape=(None, ), name="labels")
    is_training = tf.placeholder(tf.bool, name="is_training")


    # load network
    logits = getattr(nets, FLAGS.net)(X, is_training, FLAGS.classes)

    loss, metrics = cls_loss(logits, Y)

    metric_names = [x.name[:-2] for x in metrics]

    global_step = tf.Variable(0, name='global_step', trainable=False)
    rate = FLAGS.learning_rate
    rate = tf.train.exponential_decay(rate, global_step, FLAGS.decay_steps, FLAGS.decay_rate, staircase=True)
    optimizer = tf.train.AdamOptimizer(rate)

    train_op = optimizer.minimize(loss, global_step=global_step)
    init = tf.global_variables_initializer()
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
                  {"type": "normalize", "mean": [0,1,2], "std": [3,4,5]},
                  #{"type": "augment.rotate", "min":-10, "max":10},
                  #{"type": "augment.scale", "min":0.7, "max":1.2},
                  {"type": "clip", "size": 32, "shift": 4},
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
    '''
    if not FLAGS.resize_width is None:
        config['resize_width'] = FLAGS.resize_width
    if not FLAGS.resize_height is None:
        config['resize_height'] = FLAGS.resize_height
    if not FLAGS.max_size is None:
        config['max_size'] = FLAGS.max_size
    '''

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
                 }
        assert os.path.exists(FLAGS.val_db)
        val_stream = picpac.ImageStream(val_config)


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
                meta, image = stream.next()
                feed_dict = {X: image, Y: meta.labels, is_training: True}
                mm, _, = sess.run([metrics, train_op, ], feed_dict=feed_dict)
                avg += np.array(mm)
                pass
            step += FLAGS.epoch_steps
            avg /= FLAGS.epoch_steps
            stop_time = time.time()
            txt = ', '.join(['%s=%.4f' % (a, b) for a, b in zip(metric_names, list(avg))])
            print('step %d: elapsed=%.4f time=%.4f, %s'
                    % (step, (stop_time - global_start_time), (stop_time - start_time), txt))
            epoch += 1

            # validation
            if epoch and (epoch % FLAGS.val_epochs == 0) and not val_stream is None:
                # evaluation
                val_stream.reset()
                avg = np.array([0] * len(metrics), dtype=np.float32)
                C = 0
                for meta, image in val_stream:
                    feed_dict = {X: image, Y: meta.labels, is_training: False}
                    mm = sess.run(metrics, feed_dict=feed_dict)
                    avg += np.array(mm)
                    C += 1
                    pass
                avg /= C
                txt = ', '.join(['%s=%.4f' % (a, b) for a, b in zip(metric_names, list(avg))])
                print('step %d, validation: %s' % (step, txt))

            # model saving
            if epoch and (epoch % FLAGS.ckpt_epochs == 0):
                ckpt_path = '%s/%d' % (FLAGS.model, step)
                saver.save(sess, ckpt_path)
                print('step %d, saving to %s.' % (step, ckpt_path))
            pass
        pass
    pass

if __name__ == '__main__':
    tf.app.run()

