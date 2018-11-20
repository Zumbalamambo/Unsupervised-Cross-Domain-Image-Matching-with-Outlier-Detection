"""Train the model using Tensorflow.

The script is implemented based on the original code of finetuning AlexNet at:

https://kratzert.github.io/2017/02/24/finetuning-alexnet-with-tensorflow.html

@author: Xin Liu (contact: lolmesx(at)gmail.com)
"""

import os

import numpy as np
import tensorflow as tf

from model.alexNet_outlier import AlexNet, contrastive_loss, weighted_MK_MMD_loss, entropy_loss
from util.image_data import imageDataGenerator
from options.train_options import TrainOptions
from datetime import datetime
from numpy.random import choice
from tensorflow.contrib.data import Iterator
from tensorflow.python.ops import math_ops


if __name__ == '__main__':
    opt = TrainOptions().initialize()
    """
    Configuration Part.
    """
    np.random.seed(0)
    # Path to the textfiles for the trainings set
    train_file = opt.train_DomainS_path #'pits_pathS.txt' 
    domainB_file = opt.train_DomainT_path#'imgTargetPS.txt'

    # Learning params
    learning_rate1 = 0.001
    learning_rate2 = 0.0001
    num_epochs = 40
    batch_size = 64

    # Network params
    dropout_rate = 0.5
    # num_classes = 2
    train_layers = ['fc8', 'fc7', 'fc6', 'conv5', 'conv4']

    # Frequency of writing the tf.summary data to disk
    display_step = 20

    # Path for tf.summary.FileWriter and to store model checkpoints
    filewriter_path = "tensorboard"
    checkpoint_path = opt.checkpointPath # "/tudelft.net/staff-bulk/ewi/insy/VisionLab/xinliu/checkpoints_weightedPits"

    """
    Main part of the training process.
    """
    # opt.tr_DA_data = '/tudelft.net/staff-bulk/ewi/insy/VisionLab/xinliu/pitsn/'
    # opt.tr_DB_data = '/home/nfs/xliu7/CycleGAN-tensorflow/test/'

    # Load the training data
    tr_data = imageDataGenerator(train_file, batch_size, opt.tr_DS_data)
    DB_data = imageDataGenerator(domainB_file, batch_size, opt.tr_DT_data)

    # target sample probability initialization
    sample_p = []
    for i in range(0, 60032, 64):
        for _ in range(32):
            sample_p.append([0.7])
        for _ in range(32):
            sample_p.append([0.3])

    # Create an reinitializable iterator given the dataset structure
    iterator = Iterator.from_structure(tr_data.data.output_types,
                                       tr_data.data.output_shapes)

    iterator1 = Iterator.from_structure(DB_data.data.output_types,
                                       DB_data.data.output_shapes)
    next_batch = iterator.get_next()
    next_batch1 = iterator1.get_next()

    # Ops for initializing the two different iterators
    training_init_op = iterator.make_initializer(tr_data.data)
    DB_init_op = iterator1.make_initializer(DB_data.data)

    # TF placeholder for graph input and output
    left = tf.placeholder(tf.float32, [batch_size, 227, 227, 3], name='left')
    right = tf.placeholder(tf.float32, [None, 227, 227, 3], name='right')
    db = tf.placeholder(tf.float32, [batch_size, 227, 227, 3], name='domainB')
    sample_prob = tf.placeholder(tf.float32, [None, 1], name='sample_probability')
    with tf.name_scope("similarity"):
        label = tf.placeholder(tf.int32, [batch_size, 1], name='label')
        label = tf.to_float(label)
    margin = 1.0
    keep_prob = tf.placeholder(tf.float32)

    # Initialize model
    model_left = AlexNet(left, keep_prob, train_layers, reuse=False)
    model_right = AlexNet(right, keep_prob, train_layers, reuse=True)
    model_db = AlexNet(db, keep_prob, train_layers, reuse=True)

    # Variables to model outputs

    # weighted MK-MMD loss on fc6, fc7, fc8 outputs
    left6 = model_left.dropout6
    db6 = model_db.dropout6
    loss0 = weighted_MK_MMD_loss(left6, db6, sample_prob)

    left7 = model_left.dropout7
    db7 = model_db.dropout7
    loss1 = weighted_MK_MMD_loss(left7, db7, sample_prob)

    left8 = model_left.fullc8
    db8 = model_db.fullc8
    loss2 = weighted_MK_MMD_loss(left8, db8, sample_prob)

    left_output = model_left.fc8
    right_output = model_right.fc8
    db_output = model_db.fc8

    # Contrastive loss
    loss = contrastive_loss(left_output, right_output, label, margin)

    # Entropy loss
    entropy_loss = entropy_loss(left_output, db_output, batch_size)

    # Total Loss
    total_loss = (loss0 + loss1 + loss2) + loss + 1.5*entropy_loss
    total_loss = tf.reshape(total_loss, [])


    # List of trainable variables of the layers we want to train
    var_list1 = [v for v in tf.trainable_variables() if v.name.split('/')[0] == train_layers[0]]
    var_list2 = [v for v in tf.trainable_variables() if v.name.split('/')[0] in train_layers[1:]]


    # Train op
    with tf.name_scope("train"):
        # Get gradients of all trainable variables
        gradients = tf.gradients(total_loss, var_list1 + var_list2)
        grads1 = gradients[:len(var_list1)]
        grads2 = gradients[len(var_list1):]

        # Create optimizer and apply to the trainable variables
        optimizer1 = tf.train.AdamOptimizer(learning_rate=learning_rate1)
        optimizer2 = tf.train.AdamOptimizer(learning_rate=learning_rate2)
        grads1_list = list(zip(grads1, var_list1))
        grads2_list = list(zip(grads2, var_list2))
        train_op1 = optimizer1.apply_gradients(grads1_list)
        train_op2 = optimizer2.apply_gradients(grads2_list)
        train_op = tf.group(train_op1, train_op2)

    # Add gradients to summary
    for gradient, var in grads1_list + grads2_list:
        tf.summary.histogram(var.name + '/gradient', gradient)

    # Add the variables we train to the summary
    for var in var_list1 + var_list2:
        tf.summary.histogram(var.name, var)

    # Add the loss to summary
    tf.summary.scalar('totalloss', total_loss)

    # Merge all summaries together
    merged_summary = tf.summary.merge_all()

    # Initialize the FileWriter
    writer = tf.summary.FileWriter(filewriter_path)

    # store the original training target domain data for target sample probability update in each epoch
    target = []
    source = []
    orig_target = []
    orig_prob = sample_p
    probability = tf.get_variable("probability", shape=[60032,1], dtype=tf.float32)
    update_prob = probability.assign(sample_prob)

    # Initialize an saver for store model checkpoints
    saver = tf.train.Saver(max_to_keep=20)

    # Get the number of training/validation steps per epoch
    train_batches_per_epoch = int(np.floor(tr_data.data_size/(batch_size*2)))
    #val_batches_per_epoch = int(np.floor(val_data.data_size / batch_size))

    # GPU config
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True

    # Start Tensorflow session
    with tf.Session() as sess:

        # Initialize all variables
        sess.run(tf.global_variables_initializer())

        # Add the model graph to TensorBoard
        writer.add_graph(sess.graph)

        # Load the pretrained weights into the non-trainable layer
        model_left.load_initial_weights(sess)
        model_right.load_initial_weights(sess)
        model_db.load_initial_weights(sess)

        print("{} Start training...".format(datetime.now()))
        print("{} Open Tensorboard at --logdir {}".format(datetime.now(),
                                                          filewriter_path))

        # Loop over epochs
        for epoch in range(num_epochs):

            print("{} Epoch number: {}".format(datetime.now(), epoch+1))

            # Initialize iterator with the training dataset
            sess.run(training_init_op)
            if epoch == 0:
                sess.run(DB_init_op)

            for step in range(train_batches_per_epoch):

                # get next batch of data
                sim1 = [[1]] * 32   
                sim2 = [[0]] * 32
                sim = np.vstack((sim1, sim2))

                img_batch_left, label_batch_left = sess.run(next_batch)
                img_batch_right, label_batch_right = sess.run(next_batch)

                if epoch == 0:
                    img_batch_db, label_batch_db = sess.run(next_batch1)
                    if step == 0:
                        source = img_batch_right
                        target = img_batch_db
                    else:
                        source = np.concatenate((source, img_batch_right), axis=0)
                        target = np.concatenate((target, img_batch_db), axis=0)
                else:
                    img_batch_db = target[step*64:step*64+64]
                target_prob = sample_p[step*64:step*64+64]

                # Run the training op
                sess.run(train_op, feed_dict={left: img_batch_left,
                                              right: img_batch_right,
                                              db: img_batch_db,
                                              label: sim,
                                              sample_prob: target_prob,
                                              keep_prob: dropout_rate})


                # Generate summary with the current batch of data and write to file
                if step % display_step == 0:
                    s = sess.run(merged_summary, feed_dict={left: img_batch_left,
                                                            right: img_batch_right,
                                                            db: img_batch_db,
                                                            label: sim,
                                                            sample_prob: target_prob,
                                                            keep_prob: 1.})

                    writer.add_summary(s, epoch*train_batches_per_epoch + step)

            # Update sample probability
            if epoch == 0:
                orig_target = target
            for i in range(train_batches_per_epoch):
                source_tmp = sess.run(right_output, feed_dict={right: source[64*i:64*i+64], keep_prob: 1.0})
                target_tmp = sess.run(right_output, feed_dict={right: target[64*i:64*i+64], keep_prob: 1.0})
                orig_target_tmp = sess.run(right_output, feed_dict={right: orig_target[64*i:64*i+64], keep_prob: 1.0})
                if i == 0:
                    source_rep = source_tmp
                    target_rep = target_tmp
                    orig_target_rep = orig_target_tmp
                else:
                    source_rep = tf.concat([source_rep, source_tmp], 0)
                    target_rep = tf.concat([target_rep, target_tmp], 0)
                    orig_target_rep = tf.concat([orig_target_rep, orig_target_tmp], 0)

            sizet = len(target)
            sizes = len(source)
            half = sizes // 2
            target1_rep = target_rep[:32]
            target0_rep = target_rep[32:64]
            for k in range(64,sizet,64):
                target1_rep = tf.concat([target1_rep, target_rep[k:k+32]], 0)
                target0_rep = tf.concat([target0_rep, target_rep[k+32:k+64]], 0)

            p_s = tf.reduce_sum(tf.exp(tf.matmul(orig_target_rep,tf.transpose(source_rep[:half]))), 1)
            p_t1 = tf.reduce_sum(tf.exp(tf.matmul(orig_target_rep,tf.transpose(target1_rep))), 1)
            p_t0 = tf.reduce_sum(tf.exp(tf.matmul(orig_target_rep,tf.transpose(target0_rep))), 1)
            denominator = p_s + p_t1 + p_t0
            p_s, p_t1, p_t0 = tf.divide(p_s, denominator), tf.divide(p_t1, denominator), tf.divide(p_t0, denominator)
            p_s, p_t1, p_t0 = sess.run([p_s, p_t1, p_t0])
            target1_ = []
            target0_ = []
            possible = {}
            for i in range(sizet):
                t_label = max(p_s[i], p_t1[i], p_t0[i])
                if t_label == p_s[i]:
                    target1_.append(i)
                    orig_prob[i][0] = (p_t1[i]+p_s[i])/(p_s[i]+p_t1[i]+p_t0[i])
                elif t_label == p_t1[i]:
                    orig_prob[i][0] = p_t1[i]/(p_s[i]+p_t1[i]+p_t0[i])
                    possible[i] = orig_prob[i][0]
                else:
                    target0_.append(i)
                    orig_prob[i][0] = p_t1[i]/(p_s[i]+p_t1[i]+p_t0[i])

            # Make new target dataset according to the sample numbers classified as sudo-inliers and sudo-outliers
            if possible:
                sorted_possible = sorted(possible.items(), key=lambda k: k[1], reverse=True)
                size = len(sorted_possible)
                for i in range((size//3)):
                    target1_.append(sorted_possible[i][0])
                if not target0_:
                    for j in range((size//6)*5, size):
                        target0_.append(sorted_possible[j][0])

            new_target = []
            new_sample_p = []
            for _ in range(train_batches_per_epoch):
                for _ in range(32):
                    t1 = choice(target1_)
                    new_target.append(orig_target[t1])
                    new_sample_p.append(orig_prob[t1])
                for _ in range(32):
                    t2 = choice(target0_)
                    new_target.append(orig_target[t2])
                    new_sample_p.append(orig_prob[t2])
            target = new_target
            sample_p = new_sample_p

            print("{} Saving checkpoint of model...".format(datetime.now()))

            # Save checkpoint of the model
            if epoch % 2 == 0:
                sess.run(update_prob, feed_dict={sample_prob: orig_prob})
                checkpoint_name = os.path.join(checkpoint_path,
                                               'model_epoch'+str(epoch+1)+'.ckpt')
                save_path = saver.save(sess, checkpoint_name, write_meta_graph=False)

                print("{} Model checkpoint saved at {}".format(datetime.now(),
                                                               checkpoint_name))
