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
epochs = 20
batch_size = 25
display_step = 1
#total number of mols in training set. Keep updated if change data.
n_training_items = 8000

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

#TODO: A function which lets us make predictions with the model (use it)

def model(x, weights, biases):
    """
    Basic model using tanh activation function
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


#TODO: Utilize validation, testing data
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
        
        #do the same for training, validation for later accuracy checks
        bc_validation, e_validation = read_and_decode_single_example(
                'validation.tfrecords')
        bc_testing, e_testing = read_and_decode_single_example(
                'testing.tfrecords')
        bcs_validation_batch, es_validation_batch = tf.train.shuffle_batch(
                [bc_validation, e_validation],
                batch_size = batch_size,
                capacity = 1000,
                min_after_dequeue = 500)
        bcs_testing_batch, es_testing_batch = tf.train.shuffle_batch(
                [bc_testing, e_testing],
                batch_size = batch_size,
                capacity = 1000,
                min_after_dequeue = 500)
        
        
        #construct training model
        training_model = model(x, weights, biases)
        
        #Define the cost function and how we will reduce it
        #Reducing mean squared error for now
        cost = tf.reduce_mean(tf.squared_difference(es_batch, training_model))
        optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
        
        #number of epochs to train on
        n_epochs = 3000
        
        saver = tf.train.Saver()
        
        #TODO: Launch graph and run model
        #TODO: Add early stopping
        #TODO: Add dropout
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            for i in range(n_epochs):
                epoch_loss = 0
                costs_list = []
                #go through all training samples, batch_size at a time
                for _ in range(int(n_training_items/batch_size)):
                    _, c = sess.run([optimizer, cost], feed_dict={
                                    x: bcs_batch.eval(),
                                    y: es_batch.eval()})
                    epoch_loss += c
                costs_list.append(epoch_loss)
                #prints evaluations of model at end of each epoch
                print('Epoch', i+1, 'completed out of', n_epochs,
                      'epoch loss:', epoch_loss)
                #print('Avg % error, validation: ' +
                #      compute_accuracy(sess, 
                #                       bcs_validation_batch, 
                #                       es_validation_batch,
                #                       weights,
                #                       biases))
                save_path = saver.save(sess, 'checkpoints/model.ckpt')
                print('Model saved in file: %s' % save_path)
            #shows graph of epoch loss over time
            plt.plot(costs_list)
            coord.request_stop()
            coord.join(threads)
            
        
    except FileNotFoundError:
        print('file note found error')

def compute_accuracy(sess, bcs_batch, es_batch, weights, biases):
    """
    This should compute and return accuracy (in form of avg % error over batch
    of model
    
    args:
        sess: the session the model is trained in
        bcs_batch, es_batch: the data to be tested
        weights: the weights
        biases: the biases
    """
    
    #TODO: This
    
    

def read_and_decode_single_example(filename):
    """
    Decodes a single example from a .tfrecords file for training
    """
    #creates a filename queue, splits up data and keeps size down
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
    
