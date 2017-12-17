#! /usr/bin/env python
# -*- coding:utf-8
# Recurrent Neural network with LSTM in Tensorflow
# Special thanks: Harisson@pythonprogramming.net

'''
    File name:lstm_rnn.py
    Author: chimney37
    Date created: 12/09/2017
    Python Version: 3.62
'''
'''
We will use the MNIST data, using 60000 training samples and 10000 testing samples of
handwritten and labeled digits, 0 through 9, i.e. 10 total "classes". Actual deep
learning requires half a billion samples for accuracy. It's small enough to work on
any computers. MNIST dataset of images 28x28:784 pixels. Either pixel is "blank" i.e.
0 or there is something there : 0. We will predict the number we're looking at
(0,1,2,...8 or 9).

In a traditional single layer neural network: input data will send to hidden layer, that is weighted. It will undergo an
activation function, so neuron can decided to fire and output data to either output
layer, or another hidden layer, in the case of multi-layered neural network.

In Recurrent neural networks (RNN), the data is passed into a cell. Along with
outputting the activation function's output, we take the output and include it as an input back
to a cell. The problem is 1) how should we weight new incoming data vs. the
recurring data. How much of this should we continue down the line, as the
initial signal can dominate everything downt the line.

LSTM: Long short term memory is a form of RNN to decide what we do with the reurring
data, i.e 1) what to forget or keep for the recurring data;2) what new to add
based on what we keep; 3) what to output as a result of 1) and 2).

For more information on what is so facinating about RNN, read: http://karpathy.github.io/2015/05/21/rnn-effectiveness/

We will use a cost function (loss function), to determine how wrong we are. Lastly, we will use an
optimizer function: Adam optimizer, to minimize the cost. Cost is minimized by
tinkering with weights. How quickly we want to lower the cost is determined by
learning rate. The lower the value for learning rate, the slower we will learn, and
more likely we'd get better results.

One training cycle involving all data is called an Epoch. We can pick any number for number of epochs. After each epoch, we've
hopefully further fine-tuned our weights lowering our cost and improving accuracy.
'''
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.ops import rnn, rnn_cell

# one_hot means one eleent out of others is literally "hot" or on. This is useful for a
# multi-class classification, from 0,1,...to 9. So we want the output to be like
#
# 0 = [1,0,0,0,0,0,0,0,0]
# 1 = [0,1,0,0,0,0,0,0,0]
#...
# 9 = [0,0,0,0,0,0,0,0,1]
mnist = input_data.read_data_sets("/tmp/data/", one_hot = True)

'''
We are establishing the shape of the input (x) to include chunks. In a basic
neural network, we are sending the entire image of pixel data at once. In a
recurrent neural network, we're treating inputs now as sequential inputs of
chunks instead.Batches are used to control how many features we are going to optimize at once, as computers
are limited by memory.
'''
hm_epochs = 10
n_classes = 10
batch_size = 128
chunk_size = 28
n_chunks = 28
rnn_size = 128

#input: the image: 28x28 to a 28 chunks each consisting of a size of 28.
#If an input data is out of place that doen't
#fit the shape, this specification will ignore the data, without throwing an error.
x = tf.placeholder('float', [None,n_chunks,chunk_size])
#output
y = tf.placeholder('float')

def recurrent_neural_network(x):
    # Defining the layers and the output
    # weights: weights of each input going into a layer
    # biases: added after weights. what happens is we have
    # (input*weight)+biases. Biases will make sure a neuron may still fire if
    # all inputs are 0.
    # rnn_size:
    layer = {'weights': tf.Variable(tf.random_normal([rnn_size,n_classes])),
             'biases': tf.Variable(tf.random_normal([n_classes]))}

    # checknumpy's tranpose for details. Basically changes the shape (from e.g.
    # (1,2,3) to (2,1,3). debug print to figure out 
    x = tf.transpose(x, [1,0,2])
    x = tf.reshape(x, [-1, chunk_size])
    x = tf.split(x, n_chunks, 0)

    #rnn_cell is defined by tensorflow
    lstm_cell = rnn_cell.BasicLSTMCell(rnn_size,state_is_tuple=True)
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

    #output layer has no activation
    output = tf.matmul(outputs[-1],layer['weights']) + layer['biases']

    return output

def train_neural_network(x):
    # we produce a prediction based on the neural network model
    prediction = recurrent_neural_network(x)
    # cost function : loss function to optimize the cost. We will
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,
                                                                  labels=y))
    #Adam optimizer is a optimizer along with others like SGD (stochastic graient
    #descent and AdaGrad
    #Adam optimizer can specifiy learning_rate, default is 0.001
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    #number of cycles of feedforward and backward propagation
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range (hm_epochs):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples/batch_size)):
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                epoch_x = epoch_x.reshape((batch_size,n_chunks,chunk_size))
                _, c = sess.run([optimizer, cost], feed_dict={x:epoch_x, y: epoch_y})
                epoch_loss += c

            print('Epoch', epoch, 'completed out of', hm_epochs, 'loss:', epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y,1))
        #tell us how many predictions we made that were perfect matches to their labels
        accuracy = tf.reduce_mean(tf.cast(correct,'float'))
        print('Accuracy:',
              accuracy.eval({x:mnist.test.images.reshape((-1,n_chunks,chunk_size)), y:mnist.test.labels}))

#train neural network
train_neural_network(x)
