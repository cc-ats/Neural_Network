#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 14:38:58 2017

@author: johnalberse
"""
import numpy as np
import tensorflow as tf
import nn_input_handler as nni

#size of the input 
#TODO: Keep updated
X_SIZE = 2645

def read_and_decode_single_example(filename):
    """
    Decodes a single example from a .tfrecords file for training
    """
    #creates a filename queue, splits up data and keeps size down
    filename_queue = tf.train.string_input_producer([filename], 
                                                    num_epochs=None)
    reader = tf.TFRecordReader()
    #note that _ refers to the previous symbol
    #reader converts back into actual example
    #will go to the next file in the queue if necessary
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
            serialized_example,
            #here we tell it what features to pull
            features={
                    'bc': tf.FixedLenFeature([X_SIZE], tf.float32),
                    'e': tf.FixedLenFeature([], tf.float32)
            })
    bc = features['bc']
    e = features['e']
    #input == bc, label == e
    return bc, e

#TODO: Assemble a larger database and test this
def simple_model():
    try:
        # gets examples
        bc, e = read_and_decode_single_example("training.tfrecords")
        # create batches
        #TODO: Tweak values
        bcs_batch, es_batch = tf.train.shuffle_batch(
                [bc, e], 
                batch_size = 25,
                capacity=1000,
                min_after_dequeue=500)
        
        #construct the model here
        w = tf.get_variable('w1', [X_SIZE, 1])
        y_pred = tf.matmul(bcs_batch, w)
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(y_pred, es_batch)
        
        #monitoring
        loss_mean = tf.reduce_mean(loss)
        
        #use predefined training algorithm
        train_op = tf.rain.AdamOptimizer().minimize(loss)
        
        #tell TensorFlow to set itself up to evaluate the model
        sess = tf.Session()
        init = tf.initialize_all_variables()
        sess.run(init)
        tf.train.start_queue_runners(sess=sess)
        
        while True:
            _, loss_val = sess.run([train_op, loss_mean])
            print (loss_val)
        
    except FileNotFoundError:
        print('training.tfrecords not found. Please assemble dataset using'
              ' input handler')
    

    