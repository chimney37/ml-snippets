#! /usr/bin/env python
# -*- coding:utf-8
# Predicting sentiments using a deep neural network with tensorflow. 
# Special thanks: Harisson@pythonprogramming.net

'''
    File name: sentiment_predict_nn.py
    Author: chimney37
    Date created: 11/12/2017
    Python Version: 3.62
'''
'''
This is applying the neural network created for predicting digits, for
predicting sentiments. It works on the output of create_sentiment_features.py

We wil import the create_sentiment_feature_sets_and_labels from the
create_sentiment_features.py

Steps: input data will send to hidden layer 1, that is weighted. It will undergo an
activation function, so neuron can decided to fire and output data to either output
layer, or another hidden layer. We will have 3 hidden layers. We will use a cost
function (loss function), to determine how wrong we are. Lastly, we will use an
optimizer function: Adam optimizer, to minimize the cost. Cost is minimized by
tinkering with weights. How quickly we want to lower the cost is determined by
learning rate. The lower the value for learning rate, the slower we will learn, and
more likely we'd get better results. 

The act of sending the data straight through our network means we're operating a feed
forward neural network. The adjustments of weights backwards is our back propagation.
We do feeding forward and back propagation however many times we want. The cycle is
called an Epoch. We can pick any number for number of epochs. After each epoch, we've
hopefully further fine-tuned our weights lowering our cost and improving accuracy.

'''
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from create_sentiment_featuresets import create_feature_sets_and_labels
import pickle
import numpy as np

# one_hot means one eleent out of others is literally "hot" or on. This is useful for a
# in the case of sentiment prediction, it's either positive or negative
# sentiments, so we will model output as
# positive = [1,0]
# negative = [0,1]

train_x,train_y,test_x,test_y = \
create_feature_sets_and_labels('ident_nn_pos.txt','ident_nn_neg.txt')

'''
in building the model, we consider the number of nodes each hidden layer will have.
Nodes in each layer need not be identical, but it can be tweaked, depending on what
we are trying to model (TBD).

Batches are used to control how many features we are going to optimize at once, as computers
are limited by memory.

'''
n_nodes_hl1 = 1500
n_nodes_hl2 = 1500
n_nodes_hl3 = 1500
n_classes = 2
batch_size = 100
hm_epochs = 14

# specifying input and output without shape
x = tf.placeholder('float')
y = tf.placeholder('float')

def neural_network_model(data):
    # Defining the layers and the output
    # weights defined are a giant tensorflow variable, we specify the shape of the variable
    # biases: added after the weights
    # what happens here is we want (input data * weight) + biases
    # biases will make sure that a neuron may still fire even if all inputs are 0
    # tf.random_normal outputs random values for the shape we want
    # here no flow happens yet, this is just definition
    hidden_1_layer = {'f_fum':n_nodes_hl1,
                    'weight': tf.Variable(tf.random_normal([len(train_x[0]), n_nodes_hl1])),
                    'bias': tf.Variable(tf.random_normal([n_nodes_hl1]))}

    hidden_2_layer = {'f_fum':n_nodes_hl2,
                    'weight': tf.Variable(tf.random_normal([n_nodes_hl1,n_nodes_hl2])),
                    'bias': tf.Variable(tf.random_normal([n_nodes_hl2]))}

    hidden_3_layer = {'f_fum':n_nodes_hl3,
                    'weight':tf.Variable(tf.random_normal([n_nodes_hl2,n_nodes_hl3])),
                    'bias':tf.Variable(tf.random_normal([n_nodes_hl3]))}

    output_layer = {'f_fum':None,
                    'weight':tf.Variable(tf.random_normal([n_nodes_hl3,n_classes])),
                    'bias':tf.Variable(tf.random_normal([n_classes]))}

    # input_data * weights + biases (matrix multiplication)
    # relu: recified linear: this is the activation function (threshold)
    l1 = tf.add(tf.matmul(data, hidden_1_layer['weight']), hidden_1_layer['bias'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1, hidden_2_layer['weight']), hidden_2_layer['bias'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2, hidden_3_layer['weight']), hidden_3_layer['bias'])
    l3 = tf.nn.relu(l3)

    #output layer has no activation
    output = tf.matmul(l3,output_layer['weight']) + output_layer['bias']

    return output

def train_neural_network(x):
    # we produce a prediction based on the neural network model
    prediction = neural_network_model(x)
    # cost function : loss function to optimize the cost. We will
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,
                                                                  labels=y))
    #Adam optimizer is a optimizer along with others like SGD (stochastic graient
    #descent and AdaGrad
    #Adam optimizer can specifiy learning_rate, default is 0.001
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range (hm_epochs):
            epoch_loss = 0
            i=0
            while i < len(train_x):
                start = i
                end = i+batch_size
                batch_x = np.array(train_x[start:end])
                batch_y = np.array(train_y[start:end])

                _, c = sess.run([optimizer,cost],feed_dict={x:batch_x,y:batch_y})
                epoch_loss += c
                i += batch_size

            print('Epoch', epoch+1, 'completed out of', hm_epochs, 'loss:',epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y,1))
        #tell us how many predictions we made that were perfect matches to their labels
        accuracy = tf.reduce_mean(tf.cast(correct,'float'))
        print('Accuracy:', accuracy.eval({x:test_x, y:test_y}))


train_neural_network(x)
