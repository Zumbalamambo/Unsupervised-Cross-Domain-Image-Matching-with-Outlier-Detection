'''
Data generator for Evaluation
'''
#import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from tensorflow.contrib.data import Iterator
from image_data import imageDataGenerator

def eval_data_generator(file_path, batch_size, sub_img_folder):
    temp = imageDataGenerator(file_path, batch_size,sub_img_folder)
    iterator = Iterator.from_structure(temp.data.output_types,
                                       temp.data.output_shapes)
    next_batch = iterator.get_next()
    dataset_init_op = iterator.make_initializer(temp.data)
    with tf.Session() as sess:
        sess.run(dataset_init_op)
        if sub_img_folder == '/test/':
            img, label = sess.run(next_batch)
            img, label = sess.run(next_batch)
        else:
            img, label = sess.run(next_batch)
        # print(img.shape, label)
        # imgplot = plt.imshow(img[0])
        # plt.show()
    return img, label

# file_path = "image_path.txt"
# batch_size = 1
# img = eval_data_generator(file_path, batch_size)
# print(type(img))
