#! /usr/bin/env python
# -*- coding:utf-8
# Convolutional Neural Network (CNN) for recognizing handwritten digits
# Special thanks: Harisson@pythonprogramming.net

'''
    File name: cnn_test.py
    Author: chimney37
    Date created: 12/09/2017
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

In a traditional neural network, input data will send to hidden layer 1, that is weighted. It will undergo an
activation function, so neuron can decided to fire and output data to either output
layer, or another hidden layer.

In a Convolutional Neural Network (CNN), the processes go by: Convolution =>
Pooling => Convolution => Pooling => Fully connected layer => Output. Each term
between the arrows ia a layer. Convolution is the act of taking original data, and creating feature maps from
it. Pooling is down-sampling, most often in the form of "max-pooling", where we
select a region, and then take the maximum value in that region, and that
becomes the new value for the region. Each convolution and pooling step is a
hidden layer. Fully connected layer are typically neural
networks (multilayer perceptron), where all nodes are "fully connected". Convolutional layers are not
fully connected like a traditional neural network.
Reference: https://pythonprogramming.net/convolutional-neural-network-cnn-machine-learning-tutorial/?completed=/rnn-tensorflow-python-machine-learning-tutorial/

Are convolution networks related to convolution of 2 functions in mathematics.
For people who come from signal processing, image processing, computer vision background,
convolution may be very familiar. In fact, CNN utilitizes the convolution
operation. The only difference is that in convolutional (filtering and encoding
by transformation) neural network (CNN), each network layer acts as a
detection filter for the presence of specfic features or patterns present in
the data. First layers in CNN detect large features that can be recognized
easily, then later layers detect increasingly larger features thare more
abstract and present in many of the larger feature present by earlier layers.
In a way, the CNN detects a specific feature present in input data of the
layer, by convoluting the filter with the input data. (which is why
mathematical convolution and CNN is similar)
Reference: https://www.quora.com/Are-convolutional-neural-networks-related-to-the-convolution-of-two-functions-in-mathematics

for more understanding on convolutional_neural_network and the act of
convolution, refer to:
https://www.analyticsvidhya.com/blog/2017/06/architecture-of-convolutional-neural-networks-simplified-demystified/


We will use a cost function (loss function), to determine how wrong we are. We
will use an optimizer function: Adam optimizer, to minimize the cost. Cost is
minimized by tinkering with weights. How quickly we want to lower the cost is determined by
learning rate. The lower the value for learning rate, the slower we will learn, and
more likely we'd get better results.

One training cycle over the entire piece of data is called an Epoch. We can pick any number for number of epochs. After each epoch, we've
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
n_classes = 10
batch_size = 128
# number of cycles of feedforward and backward propagation
hm_epochs = 9

# input: 784 is pixels. The matrix is 1 x 2 because we flatten the image: 28x28 to a
# 784 values. This is also known as the shape. If an input data is out of place that doen't
# fit the shape, this specification will ignore the data, without throwing an error.
x = tf.placeholder('float', [None, 784])
y = tf.placeholder('float')

# Functions here are the same from official tensorflow CNN tutorial.
# strides indicate movement of window. 1 means we just move 1 and a time, in
# conv2d, and 2 at a time in maxpool2d. ksize is size of pooling window. In
# this case we are using 2x2 pixels for pooling.
# padding is SAME means the output image has same size as input


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def maxpool2d(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def convolutional_neural_network(x):
    # weights defined are tensorflow variables, we specify the shape of the variable
    # biases: added after the weights
    # what happens here is we want (input data * weight) + biases
    # biases will make sure that a neuron may still fire even if all inputs are 0
    # tf.random_normal outputs random values for the shape we want

    weights = {
        # 5x5 convolution, 1 input image, 32 outputs
        'W_conv1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
        # 5x5 conv, 32 inputs, 64 outputs
        'W_conv2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
        # fully connected, 7x7x64 inputs, 1024 outputs
        'W_fc': tf.Variable(tf.random_normal([7*7*64, 1024])),
        # 1024 inputs, 10 outputs (class prediction)
        'out': tf.Variable(tf.random_normal([1024, n_classes]))
    }

    biases = {
        'b_conv1': tf.Variable(tf.random_normal([32])),
        'b_conv2': tf.Variable(tf.random_normal([64])),
        'b_fc': tf.Variable(tf.random_normal([1024])),
        'out': tf.Variable(tf.random_normal([n_classes]))
    }

    # Reshape input to 4D tensor
    x = tf.reshape(x, shape=[-1, 28, 28, 1])
    # Convolution layer, using our function
    # relu: recified linear: this is the activation function (threshold)
    conv1 = tf.nn.relu(conv2d(x, weights['W_conv1'])+biases['b_conv1'])
    # Max pooling (down-sampling)
    conv1 = maxpool2d(conv1)
    # Convolution layer
    conv2 = tf.nn.relu(conv2d(conv1, weights['W_conv2'])+biases['b_conv2'])
    # Max pooling (down-sampling)
    conv2 = maxpool2d(conv2)

    # Fully connected Layer
    # Reshape conv2 output to fit fully connected layer
    fc = tf.reshape(conv2, [-1, 7*7*64])
    fc = tf.nn.relu(tf.matmul(fc, weights['W_fc'])+biases['b_fc'])

    # output layer has no activation
    output = tf.matmul(fc, weights['out'])+biases['out']

    return output


def train_neural_network(x):
    # we produce a prediction based on the neural network model
    prediction = convolutional_neural_network(x)
    # cost function : loss function to optimize the cost. We will
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,
                                                                  labels=y))
    # Adam optimizer is an optimizer along with others like SGD (stochastic gradient)
    # descent and AdaGrad
    # Adam optimizer can specifiy learning_rate, default is 0.001
    optimizer = tf.train.AdamOptimizer().minimize(cost)

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
