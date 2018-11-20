'''
Helper class for processing image inputs
'''

import os
import tensorflow as tf
import numpy as np

from tensorflow.contrib.data import Iterator
from tensorflow.python.framework import dtypes
from tensorflow.python.framework.ops import convert_to_tensor

IMAGENET_MEAN = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32)

class imageDataGenerator(object):

    def __init__(self, txt_file, batch_size, sub_img_folder):
        self.txt_file = txt_file
        self.sub_img_folder = sub_img_folder

        self._read_txt_file()

        self.data_size = len(self.labels)

        # Convert lits to tensor
        self.img_paths = convert_to_tensor(self.img_paths, dtype=dtypes.string)
        self.labels = convert_to_tensor(self.labels, dtype=dtypes.int32)

        # Create dataset
        data = tf.data.Dataset.from_tensor_slices((self.img_paths, self.labels))
        data = data.map(self._parse_function)
        self.wholeset = data
        data = data.batch(batch_size)
        self.data = data

    def _read_txt_file(self):
        self.img_paths = []
        self.labels = []
        with open(self.txt_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                items = line.split(' ')
                temp = self.sub_img_folder
                self.img_paths.append(temp+items[0])
                if items[1][-2] == 'A':
                    self.labels.append(int(items[1][:-2] + '12'))
                elif items[1][-2] == 'B':
                    self.labels.append(int(items[1][:-2] + '34'))
                elif items[1][:3] == 'out':
                    self.labels.append(int('9999' + items[1][3:]))
                elif items[1][-4:-1] == 'out':
                    self.labels.append(int(items[1][:-4] + '9999'))  
                else:
                    self.labels.append(int(items[1]))

    def _parse_function(self, filename, label):
        '''Preprocessing of images'''

        img_string = tf.read_file(filename)
        img_decoded = tf.image.decode_png(img_string, channels=0)
        img_resized = tf.image.resize_images(img_decoded, [227,227])
        img_decoded = tf.to_float(img_resized)
        img_centered = tf.subtract(img_decoded, IMAGENET_MEAN)
        # RGB -> BGR
        img_bgr = img_centered[:, :, ::-1]
        return img_bgr, label

