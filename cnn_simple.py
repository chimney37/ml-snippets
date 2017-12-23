#! /usr/bin/env python
# -*- coding:utf-8
# Convolutional Neural Network (CNN) for recognizing handwritten digits, using tflearn
# Special thanks: Harisson@pythonprogramming.net

'''
    File name: cnn_simple.py
    Author: chimney37
    Date created: 12/09/2017
    Python Version: 3.62
'''
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import tflearn.datasets.mnist as mnist
import argparse
import numpy as np
import os.path
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

Another concept introduced is "dropout". Idea is to mimic dead neuraons in the brain. Actual impact
is that it seems to decrease the chance of over-weighted or otherwise biasing neurons in the
artificial neural network.

We will use a cost function (loss function), to determine how wrong we are. We
will use an optimizer function: Adam optimizer, to minimize the cost. Cost is
minimized by tinkering with weights. How quickly we want to lower the cost is determined by
learning rate. The lower the value for learning rate, the slower we will learn, and
more likely we'd get better results.

One training cycle over the entire piece of data is called an Epoch. We can pick any number for number of epochs. After each epoch, we've
hopefully further fine-tuned our weights lowering our cost and improving accuracy.

This implementation is based on TFLearn, which is a higher level library, and less prone to make mistakes.
'''

model_file = 'quicktest.model'

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a neural network of predicting digits from images')
    parser.add_argument('-e',  '--epochs', default=10, help='an integer specifying\
                        number of epochs for training.',
                        type=int)
    parser.add_argument('-t', '--train', action='store_true', help='training mode')
    parser.add_argument('-p', '--predict', action='store_true', help='prediction mode')
    args = parser.parse_args()

# one_hot means one eleent out of others is literally "hot" or on. This is useful for a
# multi-class classification, from 0,1,...to 9. So we want the output to be like
#
# 0 = [1,0,0,0,0,0,0,0,0]
# 1 = [0,1,0,0,0,0,0,0,0]
# ...
# 9 = [0,0,0,0,0,0,0,0,1]
X, Y, test_x, test_y = mnist.load_data(one_hot=True)

# input: 784 is pixels. The matrix is 1 x 2 because we flatten the image: 28x28 to a
# 784 values. This is also known as the shape. If an input data is out of place that doen't
X = X.reshape([-1, 28, 28, 1])
test_x = test_x.reshape([-1, 28, 28, 1])

# input layer
# fit the shape, this specification will ignore the data, without throwing an error.
convnet = input_data(shape=[None, 28, 28, 1], name='input')

# 2 layers of convolution and pooling
convnet = conv_2d(convnet, 32, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 64, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

# add a fully connected layer and dropout
convnet = fully_connected(convnet, 1024, activation='relu')
convnet = dropout(convnet, 0.8)

# output layer
convnet = fully_connected(convnet, 10, activation='softmax')
convnet = regression(convnet, optimizer='adam',
                     learning_rate=0.01,
                     loss='categorical_crossentropy',
                     name='targets')

# create the model
model = tflearn.DNN(convnet)


if args.train is True and args.epochs is not None:
    # train the model
    model.fit({'input': X},
              {'targets': Y},
              n_epoch=10,
              validation_set=({'input': test_x},
                              {'targets': test_y}),
              snapshot_step=500,
              show_metric=True,
              run_id='mnist')

    # save the model
    model.save(model_file)

if os.path.isfile(model_file):
    model.load(model_file)
    print("loaded model:", model_file)
else:
    print('cannot load model. train it first')

if args.predict is not None:
    print(np.round(model.predict([test_x[1]])[0]))
    print(test_y[1])
