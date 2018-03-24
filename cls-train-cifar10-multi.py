#!/usr/bin/env python3
import os
import sys
sys.path.append('install/picpac/build/lib.linux-x86_64-%d.%d' % sys.version_info[:2])
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import time
import threading
from tqdm import tqdm
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.contrib import layers
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim.nets import resnet_v1
#import resnet_v1
import picpac
#import cls_nets as nets
print(picpac.__file__)

def resnet_v1_18_impl (inputs,
                 num_classes=None,
                 is_training=True,
                 global_pool=True,
                 output_stride=None,
                 reuse=tf.AUTO_REUSE,
                 scope='resnet_v1_18'):
  resnet_v1_block = resnet_v1.resnet_v1_block
  blocks = [
      resnet_v1_block('block1', base_depth=64, num_units=2, stride=2),
      resnet_v1_block('block2', base_depth=128, num_units=2, stride=2),
      resnet_v1_block('block3', base_depth=256, num_units=2, stride=2),
      resnet_v1_block('block4', base_depth=512, num_units=2, stride=1),
  ]
  return resnet_v1.resnet_v1(
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
    logits, _ = resnet_v1_18_impl(inputs, num_classes=num_classes, is_training=is_training)
    logits = tf.squeeze(logits, [1,2]) # resnet output is (N,1,1,C, remove the 
    return tf.identity(logits, name='logits')


flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('db', 'cifar10-train.picpac', 'training db')
flags.DEFINE_string('val_db', 'cifar10-test.picpac', 'validation db')
flags.DEFINE_integer('classes', 10, 'number of classes')
flags.DEFINE_string('mixin', None, 'mix-in training db')
flags.DEFINE_integer('channels', 3, '')
flags.DEFINE_integer('size', 32, '')

flags.DEFINE_integer('batch', 256, 'Batch size.  ')

flags.DEFINE_string('net', 'resnet_18', 'architecture')
flags.DEFINE_string('model', 'cls_model', 'model directory')
flags.DEFINE_string('resume', None, 'resume training from this model')
flags.DEFINE_integer('max_to_keep', 100, '')

# optimizer settings
flags.DEFINE_float('learning_rate', 0.02, 'Initial learning rate.')
flags.DEFINE_float('decay_rate', 0.95, '')
flags.DEFINE_float('decay_steps', 500, '')
#
flags.DEFINE_integer('max_steps', 200000, '')
flags.DEFINE_integer('ckpt_epochs', 10, '')
flags.DEFINE_integer('val_epochs', 1, '')
flags.DEFINE_integer('num_gpus', 2, '')


def cls_loss (logits, labels):

    cnt = tf.identity(tf.to_float(tf.shape(logits)[0]), name='ct')
    #labels = tf.to_int32(labels)
    logits = tf.reshape(logits, (-1, FLAGS.classes))
    labels = tf.reshape(labels, (-1,))
    xe = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
    xe = tf.reduce_sum(xe, name='xe')

    with tf.device('/cpu:0'):
        acc = tf.cast(tf.nn.in_top_k(logits, labels, 1), tf.float32)
        acc = tf.reduce_sum(acc, name='acc')

    reg = tf.reduce_sum(tf.losses.get_regularization_losses())
    reg = tf.identity(reg, name='re')
    # optimize for cross-entropy
    # display cross entropy and accuracy
    #reg = tf.identity(0, name='reg')
    loss = tf.identity(xe + reg, name='lo')
    return loss/cnt, [cnt, xe, acc, reg, loss]

def average_gradients(tower_grads):
  #[ [grad, v] ]
  average_grads = []
  for grad_and_vars in zip(*tower_grads):
    grads = []
    for g, nn in grad_and_vars:
      #print(nn, g.dtype)
      expanded_g = tf.expand_dims(g, 0)
      grads.append(expanded_g)
    grad = tf.concat(grads, 0)
    grad = tf.reduce_mean(grad, 0)
    v = grad_and_vars[0][1]
    grad_and_var = (grad, v)
    average_grads.append(grad_and_var)
  return average_grads

def main (_):

    try:
        os.makedirs(FLAGS.model)
    except:
        pass

    with tf.device('/cpu:0'):
        X = tf.placeholder(tf.float32, shape=(None, None, None, FLAGS.channels), name="images")
        # ground truth labels
        Y = tf.placeholder(tf.int32, shape=(None, ), name="labels")
        queue = tf.FIFOQueue(32, (tf.float32, tf.int32))
        enc = queue.enqueue((X, Y))

        is_training = tf.placeholder(tf.bool, name="is_training")

        global_step = tf.train.create_global_step()
        rate = FLAGS.learning_rate
        rate = tf.train.exponential_decay(rate, global_step, FLAGS.decay_steps, FLAGS.decay_rate, staircase=True)
        optimizer = tf.train.MomentumOptimizer(learning_rate=rate, momentum=0.9)

        with slim.arg_scope([slim.conv2d],
                weights_regularizer=slim.l2_regularizer(2.5e-4)), \
             slim.arg_scope([layers.batch_norm], decay=0.9, epsilon=5e-4): 

            # CPU variables
            #_ = nets.resnet_18(X, is_training, FLAGS.classes)

            all_grads = []
            all_metrics = []

            refs = []
            for i in range(FLAGS.num_gpus):
                with tf.device('/gpu:%d' % i):
                    with tf.name_scope('w%d' % i) as scope:
                        decX, decY = queue.dequeue()
                        decX.set_shape(X.get_shape())
                        decY.set_shape(Y.get_shape())
                        with tf.variable_scope('netnet'):
                            logits = resnet_18(decX, is_training, FLAGS.classes)
                        #tf.get_variable_scope().reuse_variables()
                        loss, metrics = cls_loss(logits, decY)
                        grads = optimizer.compute_gradients(loss)
                        all_grads.append(grads)
                        all_metrics.append(metrics)
                        refs.append((decX, decY, loss, metrics))
                        pass
                    pass
                pass
        grads = average_gradients(all_grads)
        train_op = optimizer.apply_gradients(grads, global_step=global_step)
        pass

    metric_names = [x.name.split('/')[1][:-2] for x in all_metrics[0]]

    def format_metrics (avg):
        return ' '.join(['%s=%.3f' % (a, b) for a, b in zip(metric_names, list(avg))])

    #global_step = tf.Variable(0, name='global_step', trainable=False)
    #optimizer = tf.train.AdamOptimizer(0.0001)

    #train_op = optimizer.minimize(loss, global_step=global_step)
    #train_op = slim.learning.create_train_op(loss, optimizer, global_step=global_step)
    #init = tf.initialize_all_variables()
    #saver = tf.train.Saver(max_to_keep=FLAGS.max_to_keep)

    picpac_config = {"db": FLAGS.db,
              "loop": True,
              "shuffle": True,
              "reshuffle": True,
              "annotate": False,
              "channels": FLAGS.channels,
              "stratify": True,
              "dtype": "float32",
              "batch": FLAGS.batch//2,
              "threads": 6,
              "transforms": [
                  {"type": "augment.flip", "horizontal": True, "vertical": False},
                  {"type": "normalize", "mean": 127, "std": 127},
                  #{"type": "augment.rotate", "min":-10, "max":10},
                  #{"type": "augment.scale", "min":0.7, "max":1.2},
                  {"type": "clip", "size": FLAGS.size, "shift": 4},
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
                        {"type": "clip", "size": FLAGS.size, 'border_type': 'replicate'},
                  ]
                 }
        assert os.path.exists(FLAGS.val_db)
        val_stream = picpac.ImageStream(val_config)


    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True

    with tf.Session(config=config) as sess, open('cls-train-cifar10.log', 'w') as log:
        coord = tf.train.Coordinator()
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        def enqueue_sample (): 
            while not coord.should_stop():
                meta, images = stream.next()
                sess.run(enc, feed_dict={X: images, Y: meta.labels.astype(np.int32)})
            pass

        # create encoding threads
        thread = threading.Thread(target=enqueue_sample, args=())
        thread.start()

        #if FLAGS.resume:
        #    saver.restore(sess, FLAGS.resume)
        epoch_steps = (stream.size() + FLAGS.batch-1) // FLAGS.batch
        best = 0

        step = 0
        epoch = 0
        global_start_time = time.time()

        while step < FLAGS.max_steps:
            start_time = time.time()
            avg = np.array([0] * len(metrics), dtype=np.float32)
            progress = tqdm(range(epoch_steps), leave=False)
            for _ in progress:
                mms, _, = sess.run([all_metrics, train_op], feed_dict={is_training:True})
                for mm in mms:
                    avg += np.array(mm)
                progress.set_description(format_metrics(avg/avg[0]))
                step += 1
                pass
            avg /= avg[0]
            stop_time = time.time()
            print('epoch=%d step=%d elapsed=%.4f time=%.4f %s'
                    % (epoch, step, (stop_time - global_start_time), (stop_time - start_time),
                       format_metrics(avg)))
            epoch += 1

            # validation

            X1, Y1, loss1, metrics1 = refs[0]

            if epoch and (epoch % FLAGS.val_epochs == 0) and not val_stream is None:
                lr = sess.run(rate)
                # evaluation
                val_stream.reset()
                avg = np.array([0] * len(metrics), dtype=np.float32)
                for meta, image in val_stream:
                    feed_dict = {X1: image, Y1: meta.labels, is_training: False}
                    mm = sess.run(metrics1, feed_dict=feed_dict)
                    avg += np.array(mm)
                    pass
                avg /= avg[0]

                if avg[2] > best:
                    best = avg[2]
                print('validation %s' % format_metrics(avg), "best", best, 'lr', lr)
                log.write('%d\t%s\t%.4f\n' % (epoch, '\t'.join(['%.4f' % x for x in avg]), best))

            '''
            # model saving
            if epoch and (epoch % FLAGS.ckpt_epochs == 0):
                ckpt_path = '%s/%d' % (FLAGS.model, epoch)
                saver.save(sess, ckpt_path)
                print('step %d, saving to %s.' % (step, ckpt_path))
            pass
            '''
        pass
    pass

if __name__ == '__main__':
    tf.app.run()

