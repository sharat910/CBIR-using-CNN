# pass the path of the file as argument that is to be formatted for input
import os, sys
# from pillow import Image
# import pillow
from PIL import Image

from resizeimage import resizeimage

import tensorflow as tf
size = 32, 32
from PIL import Image, ImageOps

def create_thumbnail(infile):
    outfile = "new"+infile
    if infile != outfile:
        try:
            with open(infile, 'r+b') as f:
                with Image.open(f) as image:
                    cover = resizeimage.resize_cover(image, [32, 32], validate=False)
                    cover.save(outfile, image.format)
        except IOError:
            print "cannot create thumbnail for '%s'" % infile
	return outfile

# create 3D tensor out of the image [height, width, channels]
def create_array(x):
	input_img = tf.image.decode_jpeg(tf.read_file(x), channels=3)
	transposed_img = tf.transpose(input_img, [2, 0, 1])
	reshaped_img = tf.reshape(transposed_img,[-1])
	return reshaped_img

def getarray(infile):
	outfile= create_thumbnail(infile)
	# 2. TENSORFLOW SESSION
	with tf.Session() as sess:
		array = sess.run([create_array(outfile)])
        return array
