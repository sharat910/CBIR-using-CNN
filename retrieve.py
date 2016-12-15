from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import time

import numpy as np
import tensorflow as tf
import datetime
from tensorflow.models.image.cifar10 import cifar10
import time

def get_formatted_image(image):
    raw = np.array(image)
    raw2 = tf.reshape(raw, [3, 32, 32])
    raw3 = tf.transpose(raw2, [1, 2, 0])
    raw4 = tf.cast(raw3, tf.float32)
    resized_image = tf.image.resize_image_with_crop_or_pad(raw4, 24, 24)
    # Subtract off the mean and divide by the variance of the pixels.
    float_image = tf.image.per_image_whitening(resized_image)
    raw5 = tf.expand_dims(float_image, 0)# 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    return raw5


def getvector(image):
    with tf.Graph().as_default() as g:
        fimage = get_formatted_image(image)
        local4 = cifar10.inference(fimage)
        # Restore the moving average version of the learned variables for eval.
        variable_averages = tf.train.ExponentialMovingAverage(cifar10.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)
        #print ("Loaded saver"), datetime.datetime.now().time()
        # print time.time()
        # print datetime.datetime.now()
        with tf.Session() as sess:
          ckpt = tf.train.get_checkpoint_state("./train_log")
          #print ("Loaded ckpt"), datetime.datetime.now().time()
          if ckpt and ckpt.model_checkpoint_path:
            # Restores from checkpoint
            saver.restore(sess, ckpt.model_checkpoint_path)
            #print ("Restored sess"), datetime.datetime.now().time()
            # Assuming model_checkpoint_path looks something like:
            #   /my-favorite-path/cifar10_train/model.ckpt-0,
            # extract global_step from it.
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
          else:
            print('No checkpoint file found'), datetime.datetime.now().time()
            return

          # Start the queue runners.
          coord = tf.train.Coordinator()
          try:
            threads = []
            for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
              threads.extend(qr.create_threads(sess, coord=coord, daemon=True,start=True))
            #print ("Before run"), datetime.datetime.now().time()
            vector = sess.run([local4])
            #print ("After run"), datetime.datetime.now().time()
          except Exception as e:
              coord.request_stop(e)

          coord.request_stop()
          coord.join(threads, stop_grace_period_secs=10)
          return vector

if __name__ == '__main__':
    raw_image = unpickle("test_batch")['data'][0]
    print (getvector(raw_image))
