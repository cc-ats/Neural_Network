#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 14:38:58 2017

@author: johnalberse
"""
import tensorflow as tf
import matplotlib.pyplot as plt

#TODO: Keep updated
#Size of input
INPUT_SIZE = 2645

#parameters
learning_rate = .0001
n_epochs = 500
batch_size = 25
display_step = 1
#total number of mols in training set. Keep updated if change data.
n_training_items = 8000
#number of molecules in testing and validation (not combined, invdividually)
n_test_and_valid_items = 1000

#network architecture
n_hidden_1 = 400 #number of features in first hidden layer
n_hidden_2 = 100 #number of features in second hidden layer
n_output = 1 #number of features in output layer (will be rescaled)

#tf graph input - sets up input laayer and final layer
# Note that "None" allows us to feed an arbitrary batch size
x = tf.placeholder(dtype=tf.float32, shape=[None, INPUT_SIZE])
y = tf.placeholder(dtype=tf.float32, shape=[None, 1])

#stores values for weights, bias, including initial values
# TODO experiment with rescale values as variables vs immutable
weights = {
        'w1' : tf.Variable(tf.random_normal([INPUT_SIZE, n_hidden_1])),
        'w2' : tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
        'out' : tf.Variable(tf.random_normal([n_hidden_2, n_output])),
        'rescale_weight' : tf.Variable(initial_value=-1185.)
}
biases = {
        'b1' : tf.Variable(tf.random_normal([n_hidden_1])),
        'b2' : tf.Variable(tf.random_normal([n_hidden_2])),
        'out' : tf.Variable(tf.random_normal([n_output])),
        'rescale_bias' : tf.Variable(initial_value=-301.)
}

#TODO: Function to load checkpoint and evaluate new molecules

def model(x, weights, biases):
    """
    Basic model using tanh activation function. Note that weights, biases are
    passed; this may be useful in the future if different weight, bias
    initial values are desired.
    
    x: input, geometry of a single molecule
    """
    #Hidden layer 1
    hidden_1 = tf.tanh(tf.add(tf.matmul (x, weights['w1']), biases['b1']))
    #Hidden layer 2
    hidden_2 = tf.tanh(tf.add(tf.matmul(hidden_1, weights['w2']), 
                                             biases['b2']))
    #Output layer
    out_layer = tf.tanh(tf.add(tf.matmul(hidden_2, weights['out']), 
                                              biases['out']))
    #rescale output to be actual energy
    rescaled_out_layer = tf.add(tf.multiply(out_layer, 
                                            weights['rescale_weight']),
                                            biases['rescale_bias'])
    return rescaled_out_layer

def train_model():
    """
    Trains the model
    """
    try:
        #gets examples
        bc, e = read_and_decode_single_example("training.tfrecords")
        #create batches
        bcs_batch, es_batch = tf.train.shuffle_batch(
                [bc, e], 
                batch_size = batch_size,
                capacity = 1000, 
                min_after_dequeue = 500)
        val_bc, val_e = read_and_decode_single_example('validation.tfrecords')
        val_bcs_batch, val_es_batch = tf.train.shuffle_batch(
                [val_bc, val_e], 
                batch_size = batch_size,
                capacity = 500, 
                min_after_dequeue = 200)
        
        #construct training model
        training_model = model(x, weights, biases)
        
        #Define the cost function and how we will reduce it
        #Reducing mean squared error for now
        cost = tf.reduce_mean(tf.squared_difference(es_batch, training_model))
        optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
        
        saver = tf.train.Saver()
        
        #TODO: Add dropout
        #TODO: Add other strategies for improvement after dropout
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            improving = True
            accuracy_over_time = []
            for i in range(n_epochs):
                if (improving == True):
                    # Each pass through this loop is ONE epoch
                    epoch_loss = 0
                    #Within for loop is 1 epoch
                    for j in range(int(n_training_items/batch_size)):
                        #Each pass through this loop is one batch
                        _, c = sess.run([optimizer, cost], feed_dict={
                                        x: bcs_batch.eval(),
                                        y: es_batch.eval()})
                        epoch_loss += c
                    #End of epoch operations below
                    
                    #prints epoch loss after each epoch
                    print('Epoch', i+1, 'completed out of', n_epochs,
                          'epoch loss:', epoch_loss)
                
                    
                    #Calculate accuracy
                    #DO NOT put in own function for abstraction; breaks threads
                    total_percent_error = 0.
                    for _ in range (int(n_test_and_valid_items / batch_size)):
                        e_estimates = sess.run(model(x, weights, biases), 
                                           feed_dict = {x: bcs_batch.eval()})
                        pe = tf.reduce_mean(tf.abs(tf.div(
                                tf.subtract(es_batch, e_estimates), es_batch)))
                        #total percent error over batch
                        total_percent_error += pe * 100
                    #Avg percent error for batch
                    curr_acc = (total_percent_error / (
                            int(n_test_and_valid_items / batch_size))).eval()
                    
                    #Display and record accuracy for this epoch
                    print('Current Percent Error: ', curr_acc)
                    accuracy_over_time.append(curr_acc)
                    
                    #early stopping implementation
                    improving = is_improving(10, accuracy_over_time, i)
                    
            #shows graph of accuracy over time on validation set
            plt.plot(accuracy_over_time)
            
            save_path = saver.save(sess, 'checkpoints/model.ckpt')
            print('Model saved in file: %s' % save_path)
            
            #TODO: Utilize testing data here
            #TODO: Graph training, validation, and testing together
            
            coord.request_stop()
            coord.join(threads)
            sess.close()
            
        
    except FileNotFoundError:
        print('file note found error')

def is_improving(patience, accuracy_over_time, epoch):
    """
    Checks if the network has improved in most recent window
    of length patience
    
    patience: Number of most recent accuracies to check.
    accuracy_over_time: list of all accuracies
    epoch: number of epochs completed, from 0
    """
    if (epoch > patience):
        past_epoch = epoch - patience
        if (accuracy_over_time[past_epoch] < accuracy_over_time[epoch]):
            return False
    return True
    

def read_and_decode_single_example(filename):
    """
    Decodes a single example from a .tfrecords file for training
    """
    #creates a filename queue, splits up data and keeps size down
    #right now only using one file as standard, but good for future
    filename_queue = tf.train.string_input_producer([filename], 
                                                    num_epochs=None)
    reader = tf.TFRecordReader()
    #reader converts back into actual example
    #will go to the next file in the queue if necessary
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
            serialized_example,
            #here we tell it what features to pull
            features={
                    'bc': tf.FixedLenFeature([INPUT_SIZE], tf.float32),
                    'e': tf.FixedLenFeature([1], tf.float32)
            })
    #Note we're using float, not int, because of later multiplication
    bc = tf.cast(features['bc'], tf.float32)
    e = tf.cast(features['e'], tf.float32)
    return bc, e
    
