#! /usr/bin/env python
# -*- coding:utf-8
# Using Large Data from Stanford, Predicting sentiments using a deep neural network with tensorflow. 
# Special thanks: Harisson@pythonprogramming.net

'''
    File name: sentiment_predict_large_nn.py
    Author: chimney37
    Date created: 11/23/2017
    Python Version: 3.62
'''
'''
This is applying the neural network created for predicting sentiments. It works
on the output of data_preprocessing.py.

Note: for program to work correctly, delete tf.log and all model.ckpt files in
directory, if you re-training the model.

Steps: input data will send to hidden layer 1, that is weighted. It will undergo an
activation function, so neuron can decided to fire and output data to either output
layer, or another hidden layer. We will have 3 hidden layers. We will use a cost
function (loss function), to determine how wrong we are. It indicates how
different the output is from the training data output. Lastly, we will use an
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
from os import path
from os import getcwd
import tensorflow as tf
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import pickle
import numpy as np
import argparse

lemmatizer = WordNetLemmatizer()

# one_hot means one eleent out of others is literally "hot" or on. This is useful for a
# in the case of sentiment prediction, it's either positive or negative
# sentiments, so we will model output as
# positive = [1,0]
# negative = [0,1]

'''
in building the model, we consider the number of nodes each hidden layer will have.
Nodes in each layer need not be identical, but it can be tweaked, depending on what
we are trying to model (TBD).

Batches are used to control how many features we are going to optimize at once, as computers
are limited by memory.

'''
max_training_size=1600000

def check_range(value):
    ivalue=int(value)
    if (ivalue <= 0) or (ivalue > max_training_size):
        raise argparse.ArgumentTypeError("%s is an invalid value" % value)
    return ivalue

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a neural network of\
                                     predicting sentiments')
    parser.add_argument('-n', "--num_training_size",
                        default=max_training_size, help='an integer\
                        specifying training data size. Maximum is ' +
                        str(max_training_size),
                        required=True,
                        type=check_range)
    parser.add_argument('-e', "--epochs",default=10, help='an integer specifying\
                        number of epochs for training.',
                        type=int)
    parser.add_argument('-u',action='store_true',help='only use neural network\
                        without training step')
    args = parser.parse_args()

#important parameters
n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_classes = 2
batch_size = 32
training_size = args.num_training_size
total_batches = int(training_size/batch_size)
hm_epochs = args.epochs
data_width = 2378

print(getcwd())

# specifying input and output without shape
x = tf.placeholder('float')
y = tf.placeholder('float')

# Defining the layers and the output
# weights defined are a giant tensorflow variable, we specify the shape of the variable
# biases: added after the weights
# what happens here is we want (input data * weight) + biases
# biases will make sure that a neuron may still fire even if all inputs are 0
# tf.random_normal outputs random values for the shape we want
# here no flow happens yet, this is just definition
hidden_1_layer = {'f_fum':n_nodes_hl1,
                    'weight': tf.Variable(tf.random_normal([data_width, n_nodes_hl1])),
                    'bias': tf.Variable(tf.random_normal([n_nodes_hl1]))}

hidden_2_layer = {'f_fum':n_nodes_hl2,
                    'weight': tf.Variable(tf.random_normal([n_nodes_hl1,n_nodes_hl2])),
                    'bias': tf.Variable(tf.random_normal([n_nodes_hl2]))}

output_layer = {'f_fum':None,
                    'weight':tf.Variable(tf.random_normal([n_nodes_hl2,n_classes])),
                    'bias':tf.Variable(tf.random_normal([n_classes]))}


def neural_network_model(data):
    # input_data * weights + biases (matrix multiplication)
    # relu: recified linear: this is the activation function (threshold)
    l1 = tf.add(tf.matmul(data, hidden_1_layer['weight']), hidden_1_layer['bias'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1, hidden_2_layer['weight']), hidden_2_layer['bias'])
    l2 = tf.nn.relu(l2)

    #output layer has no activation
    output = tf.matmul(l2,output_layer['weight']) + output_layer['bias']
    return output

#Save our model in form of checkpoints as time goes on
saver = tf.train.Saver()
tf_log = 'tf.log'

def extract_features(input_data,lexicon):
    current_words = word_tokenize(input_data.lower())
    current_words = [lemmatizer.lemmatize(i).lower() for i in current_words]
    features = np.zeros(len(lexicon))

    for word in current_words:
        if word in lexicon:
            index_value = lexicon.index(word)
            features[index_value]+=1

    return features

def train_neural_network(x):
    # we produce a prediction based on the neural network model
    prediction = neural_network_model(x)
    # cost function : loss function to optimize the cost. 
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,
                                                                  labels=y))
    #Adam optimizer is a optimizer along with others like SGD (stochastic graient
    #descent and AdaGrad
    #Adam optimizer can specifiy learning_rate, default is 0.001
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        try:
            #track what epoch we are using a log file
            epoch = int(open(tf_log,'r').read().split('\n')[-2])+1
            print('STARTING:' ,epoch)
        except:
            epoch = 1

        while epoch <= hm_epochs:
            if epoch != 1:
                print("Loading saved trained model.")
                saver.restore(sess,"model.ckpt")
            epoch_loss = 1
            with open('lexicon.pickle','rb') as f:
                lexicon = pickle.load(f)
            with open('train_set_shuffled.csv',buffering=20000,encoding='latin-1') as f:
                batch_x=[]
                batch_y=[]
                batches_run=0
                line_counter=0
                for line in f:
                    line_counter+=1
                    if line_counter >= training_size:
                        break

                    label = line.split(':::')[0]
                    tweet = line.split(':::')[1]
                    features=extract_features(tweet,lexicon)

                    line_x = list(features)
                    line_y = eval(label)
                    batch_x.append(line_x)
                    batch_y.append(line_y)
                    if len(batch_x) >= batch_size:
                        _, c = sess.run([optimizer,cost],
                                 feed_dict={x:np.array(batch_x),y:np.array(batch_y)})
                        epoch_loss += c
                        batch_x=[]
                        batch_y=[]
                        batches_run+=1
                        print('Batch run:',batches_run,'/',total_batches,'| \
                              Epoch:',epoch,'| Batch Loss:',c,)
            
            saver.save(sess,path.join(getcwd(),"model.ckpt"))
            print('Epoch', epoch, 'completed out of', hm_epochs, 'loss:',epoch_loss)
            with open(tf_log,'a') as f:
                f.write(str(epoch)+'\n')
            epoch+=1
        else:
            print("Satisfied target epochs:", str(hm_epochs))
        print("Training Complete.")

if args.u is False:
    train_neural_network(x)

def test_neural_network():
    prediction = neural_network_model(x)
    with tf.Session() as sess:
        for epoch in range(hm_epochs):
            try:
                saver.restore(sess,"model.ckpt")
            except Exception as e:
                print(str(e))
            epoch_loss=0

        correct = tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))
        accuracy = tf.reduce_mean(tf.cast(correct,'float'))

        feature_sets = []
        labels=[]
        counter =0
        with open('processed-test-set.csv', buffering=20000) as f:
            for line in f:
                try:
                    features=list(eval(line.split('::')[0]))
                    label=list(eval(line.split('::')[1]))

                    feature_sets.append(features)
                    labels.append(label)
                    counter+=1
                except:
                   pass
        print('Tested',counter,'samples.')
        
        test_x = np.array(feature_sets)
        test_y = np.array(labels)

        print('Accuracy:',accuracy.eval({x:test_x,y:test_y}))

test_neural_network()

def use_neural_network(input_data):
    prediction=neural_network_model(x)
    with open('lexicon.pickle','rb') as f:
        lexicon=pickle.load(f)

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        saver.restore(sess,"model.ckpt")
        features=extract_features(input_data,lexicon)

        features=np.array(list(features))
        #pos: [1,0], argmax:0
        #neg: [0,1], argmax:1
        result=(sess.run(tf.argmax(prediction.eval(feed_dict={x:[features]}),1)))
        if result[0] == 0:
            print('Positive:',input_data)
        elif result[0] == 1:
            print('Negative:',input_data)

use_neural_network("He's an idiot and a jerk")
use_neural_network("This was the best store I've ever seen.")
