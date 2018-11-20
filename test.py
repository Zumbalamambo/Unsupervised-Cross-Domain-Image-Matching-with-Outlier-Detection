import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim
import os

from util.image_data import imageDataGenerator
from util.eval_data import eval_data_generator
from model.alexNet_outlier import *

from scipy.spatial.distance import cdist

if __name__ == '__main__':
    opt = TestOptions().initialize()
    # set up the database and query-images
    filewriter_path = "tensorboard_eval"
    file_path = opt.databaseSetPath #'pits_database_path.txt'
    batch_size = 7000
    # img, label = eval_data_generator(file_path, batch_size, '/home/nfs/xliu7/CycleGAN-tensorflow/test/')
    # opt.databaseImagePath = '/tudelft.net/staff-bulk/ewi/insy/VisionLab/xinliu/pitsn/'
    img, label = eval_data_generator(file_path, batch_size, opt.databaseImagePath)
    query_img_path = opt.querySetPath #'queryPitsOut.txt'
    query_size = 310
    # query_img, query_label = eval_data_generator(query_img_path, query_size, '/tudelft.net/staff-bulk/ewi/insy/VisionLab/xinliu/pitsn/')
    # opt.queryImagePath = '/home/nfs/xliu7/CycleGAN-tensorflow/test/'
    query_img, query_label = eval_data_generator(query_img_path, query_size, opt.queryImagePath)

    for epoch in range(19, 17, -2):
        tf.reset_default_graph()
        img_placeholder = tf.placeholder(tf.float32, [None, 227, 227, 3], name='img')
        keep_prob = 1.0
        train_layers = ['fc8', 'fc7', 'fc6','conv5','conv4']
        net = AlexNet(img_placeholder, keep_prob, train_layers)
        net_out = net.fc8

        saver = tf.train.Saver()

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            # opt.checkpointModel = "/tudelft.net/staff-bulk/ewi/insy/VisionLab/xinliu/checkpoints_weightedPits/model_epoch"

            saver.restore(sess, opt.checkpointModel + str(epoch)+ ".ckpt")


            train_feat = sess.run(net_out, feed_dict={img_placeholder:img})
            search_feat = sess.run(net_out, feed_dict={img_placeholder:query_img})
            # print("Model_epoch11, got features...")

        # calculate the cosine similarity and sort
        # MAP result
        retr_placeholder = tf.placeholder(tf.float32, [None, 227, 227, 3], name='retr')
        retrieval =tf.summary.image('retrieval PitsDAOut', retr_placeholder, 6)
        writer = tf.summary.FileWriter(filewriter_path)
        total = []
        avP = np.array([0.0]*50)
        dist = cdist(train_feat, search_feat, 'euclidean')
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(query_size):
                rank = np.argsort(dist[:,i].ravel())
                feed_img = query_img[i]
                # feed_img = tf.reshape(feed_img,[1,227,227,3])
                # recall = []
                precision = [0.0]*50
                preR = [0.0]*50
                count = 0
                label_x = query_label[i]
                for index, k in enumerate(rank[:50]):
                    if index < 5:
                        feed_img = np.vstack((feed_img, img[k]))
                    # print(label_x, label[k])
                    if label[k] == label_x:
                        count += 1
                        precision[index]=(count / (index + 1))
                    preR[index] = count / (index + 1)

                    # recall.append(count/251)
                # prValue = 0
                # maxS = 0
                # for i, v in enumerate(recall):
                #     while v >= prValue and prVaue <= 1:
                #         maxS += max(precision[i:])
                #         prValue += (1/251)
                avP += np.array(preR)
                total.append(sum(precision)/5)
                feed_img = tf.reshape(feed_img,[6,227,227,3])
                feed_img = sess.run(feed_img)
                sum1 = sess.run(retrieval, feed_dict={retr_placeholder: feed_img})
                writer.add_summary(sum1, i+1)
            MAP = sum(total) / query_size
            P_R = avP / query_size
            print('Accuracy of Pittsout model {}: {}'.format(epoch, MAP))
            # print("average recal_precision: {}".format(P_R))
            with open('Pittsout.txt', 'w') as f:
                for x in P_R:
                    f.write(str(x) + '\n')
