import tensorflow as tf
from tensorflow.models.image.cifar10 import cifar10
import numpy as np
# Extract the CIFAR-10 data from test_batch.bin
def unpickle(file):
    import cPickle
    fo = open(file, 'rb')
    dic = cPickle.load(fo)
    fo.close()
    return dic

# def oneimage():
#     # One example taken as 1 x 3072 array
#     arr1x3072= unpickle("test_batch")['data'][0]
#     ## convert from int8 to int32
#     reshaped_image = tf.cast(arr1x3072, tf.float32)
#     # reshape to array of 3 x 32 x 32
#     arr3x32x32 = tf.reshape(reshaped_image, [3, 32, 32])
#     resized_image = tf.image.resize_image_with_crop_or_pad(arr3x32x32,24, 24)
#
#     # Subtract off the mean and divide by the variance of the pixels.
#     float_image = tf.image.per_image_whitening(resized_image)
#
#     # images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
#     ## Add extra dimension as the first row to make compatible with the input images to function inference
#     arr1x3x32x32 = tf.expand_dims(float_image, 0)
#     ## Reorder the array to required dimensions
#     arr1x32x32x3 = tf.transpose(arr1x3x32x32, [0, 2 , 3, 1])
#
#     return arr1x32x32x3

# def secondimage():
raw = np.array(unpickle("test_batch")['data'][0])
raw2 = tf.reshape(raw, [3, 32, 32])
raw3 = tf.transpose(raw2, [1, 2, 0])
raw4 = tf.cast(raw3, tf.float32)
resized_image = tf.image.resize_image_with_crop_or_pad(raw4, 24, 24)
# Subtract off the mean and divide by the variance of the pixels.
float_image = tf.image.per_image_whitening(resized_image)
raw5 = tf.expand_dims(float_image, 0)

# Test whether compatible with inference function
with tf.Session() as sess:
	#print sess.run([oneimage()])
	print (sess.run([raw5]))
