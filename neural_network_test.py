#! /usr/bin/env python
# -*- coding:utf-8
# Testing a deep neural network (multi layered perceptron model) with tensorflow.
# Special thanks: Harisson@pythonprogramming.net

'''
    File name: neural_network_test.py
    Author: chimney37
    Date created: 11/03/2017
    Python Version: 3.62
'''
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

'''
We will use the MNIST data, using 60000 training samples and 10000 testing samples of
handwritten and labeled digits, 0 through 9, i.e. 10 total "classes". In actual deep
learning requires half a billion samples for accuracy. It's small enough to work on
any computers. MNIST dataset of images 28x28:784 pixels. Either pixel is "blank" i.e.
0 or there is something there : 0. We will predict the number we're looking at
(0,1,2,...8 or 9).

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
# one_hot means one eleent out of others is literally "hot" or on. This is useful for a
# multi-class classification, from 0,1,...to 9. So we want the output to be like
#
# 0 = [1,0,0,0,0,0,0,0,0]
# 1 = [0,1,0,0,0,0,0,0,0]
# ...
# 9 = [0,0,0,0,0,0,0,0,1]
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

'''
in building the model, we consider the number of nodes each hidden layer will have.
Nodes in each layer need not be identical, but it can be tweaked, depending on what
we are trying to model (TBD).

Batches are used to control how many features we are going to optimize at once, as computers
are limited by memory.

'''
n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500
n_classes = 10
batch_size = 100

# input: 784 is pixels. the matrix is 1 x 2 because we flatten the image: 28x28 to a
# 784 values. This is also known as the shape. If an input data is out of place that doen't
# fit the shape, this specification will ignore the data, without throwing an error.
x = tf.placeholder('float',  [None, 784])

# output
y = tf.placeholder('float')


def neural_network_model(data):
    # Defining the layers and the output
    # weights defined are a giant tensorflow variable, we specify the shape of the variable
    # biases: added after the weights
    # what happens here is we want (input data * weight) + biases
    # biases will make sure that a neuron may still fire even if all inputs are 0
    # tf.random_normal outputs random values for the shape we want
    # here no flow happens yet, this is just definition
    hidden_1_layer = {'weights': tf.Variable(tf.random_normal([784, n_nodes_hl1])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl1]))}

    hidden_2_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl2]))}

    hidden_3_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl3]))}

    output_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])), 'biases': tf.Variable(tf.random_normal([n_classes]))}

    # input_data * weights + biases (matrix multiplication)
    # relu: recified linear: this is the activation function (threshold)
    l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']), hidden_3_layer['biases'])
    l3 = tf.nn.relu(l3)

    # output layer has no activation
    output = tf.matmul(l3, output_layer['weights']) + output_layer['biases']

    return output


def train_neural_network(x):
    # we produce a prediction based on the neural network model
    prediction = neural_network_model(x)
    # cost function : loss function to optimize the cost. We will
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,
                                                                  labels=y))
    # Adam optimizer is a optimizer along with others like SGD (stochastic graient
    # descent and AdaGrad
    # Adam optimizer can specifiy learning_rate, default is 0.001
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    # number of cycles of feedforward and backward propagation
    hm_epochs = 14
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(hm_epochs):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples/batch_size)):
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c

            print('Epoch', epoch, 'completed out of', hm_epochs, 'loss:', epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        # tell us how many predictions we made that were perfect matches to their labels
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:', accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))

# 10 and 20 epochs should give ~95% accuracy, but 95% is actually nothing
# great consider other methods. It's amazing that using just pixels can
# actually achieve 95%. state of art is 99%


train_neural_network(x)
