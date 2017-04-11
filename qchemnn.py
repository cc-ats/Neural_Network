#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 14:38:58 2017

@author: johnalberse
"""
import numpy as np
import tensorflow as tf
import NN_Input_Handler as nn

def read_and_decode_single_example(filename):
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
    
    #TODO try writing bc, e to a file to really test (make sure we print contents)
    
    #for now,
    return bc, e