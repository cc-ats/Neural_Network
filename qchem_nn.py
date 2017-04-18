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
X_SIZE = nni.COULOMB_DIMENSION * nni.COULOMB_DIMENSION

def read_and_decode_single_example(filename):
    """
    Decodes a single example from a .tfrecords file for training
    """
    #filename queue, splits up data and keeps size down
    filename_queue = tf.train.string_input_producer([filename], 
                                                    num_epochs=None)
    reader = tf.TFRecordReader()
    #note that _ refers to the previous symbol
    #reader converts back into actual example
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
            serialized_example,
            #here we tell it what features to pull
            features={
                    'bc': tf.FixedLenFeature([2654], tf.float32),
                    'e': tf.FixedLenFeature([], tf.float32)
            })
    bc = features['bc']
    e = features['e']
    
    return bc, e

#TODO: I definitely need a thread coordinator somewhere
def simple_model():
    """
    A simple model using high level tf functions. only input and output layers
    """
    #through some backend magic, batches can keep using this
    bc, e = read_and_decode_single_example("ges.tfrecords")
    
    #creates places for input
    x = tf.placeholder(tf.float32, [None, X_SIZE])
    
    #currently not deep, just input and output layers
    # [input size, output size]
    W = tf.Variable(tf.zeros([X_SIZE, 1]))
    #sze of 1, to add to output
    b = tf.Variable(tf.zeros([1]))
    
    #define the model
    y = tf.nn.softmax(tf.matmul(x, W) + b)
    #cross entropy prep
    y_ = tf.placeholder(tf.float32, [None, 1])
    
    #calculate cross entropy
    cross_entropy = cross_entropy = tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits(labels=y_, 
                                                            logits=y))
    #define one step of training procedure
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
    
    #create batches
    #TODO tweak values
    bcs_batch, es_batch = tf.train.shuffle_batch(
            [bc, e], 
            batch_size=25,
            capacity=2000,
            min_after_dequeue=1000)
    
    #TODO: Figure out how to merge input flow like this with project
    
    sess = tf.Session()
    init = tf.initialize_all_variables()
    sess.run(init)
    tf.train.start_queue_runners(sess=sess)
    #es, bcs = sess.run([es_batch, bcs_batch])