#! /usr/bin/env python
# -*- coding:utf-8
# K nearest neighbor
# Sample Data : https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Original%29
# Special thanks: Harisson@pythonprogramming.net

'''
    File name: knn.py
    Author: chimney37
    Date created: 8/11/2017
    Python Version: 3.62
'''
import sys
import numpy as np
from sklearn import preprocessing, cross_validation, neighbors
import pandas as pd

# Column attributes (id,clump_thickness,uniform_cell_size,
# uniform_cell_shape,marginal_adhesion,
# single_epi_cell_size,bare_nuclei,bland_chromation,
# normal_nucleoli,mitoses,class)
# For more info, see Attribute information:
# https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.names
df = pd.read_csv('breast-cancer-wisconsin.data.txt')
df.replace('?',-sys.maxsize, inplace=True)
df.drop(['id'], 1, inplace=True)

# drop the column(s) not used for training
X = np.array(df.drop(['class'],1))

# labels (for prediction)
Y = np.array(df['class'])

#create training and testing samples
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, Y,test_size=0.2)

#Train the classifier
clf = neighbors.KNeighborsClassifier()
clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)
print('accuracy=', accuracy)

# crete example data for prediction
example_measures = np.array([[4,2,1,1,1,2,3,2,1],[4,2,1,1,1,2,3,2,1]])
example_measures = example_measures.reshape(len(example_measures),-1)

prediction = clf.predict(example_measures)
print(prediction)
