#! /usr/bin/env python
# -*- coding:utf-8
# K nearest neighbor custom function compared with scikit knn
# the problem with the algorithm is performance, since KNN 
# simply loads points into memory, compare the distance of 
# sample point to every point to find the closest one. 
# Special thanks: Harisson@pythonprogramming.net

'''
    File name: cust_knn_compare.py
    Author: chimney37
    Date created: 8/13/2017
    Python Version: 3.62
'''
import sys
import warnings
from math import sqrt
from collections import Counter
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style

def k_nearest_neighbors(data, predict, k=3):
    if len(data) >= k:
        warnings.warn('K is set to a value less than total voting groups!')
    distances = []
    for group in data:
        for features in data[group]:
            # eulidean Norm: measure magnitude of vector
            euclidean_dist = np.linalg.norm(np.array(features)-np.array(predict))
            #generate a distances list
            distances.append([euclidean_dist,group])

    # generate a top k distances of sorted list by distance
    votes = [i[1] for i in sorted(distances)[:k]]
    #find the most common voted class. [0][0] is the 1st element of tuple 
    vote_result = Counter(votes).most_common(1)[0][0]
    return vote_result

# read in data, replace ? with maxsize, drop id and replace 
# dataframe with float values
df = pd.read_csv('breast-cancer-wisconsin.data.txt')
df.replace('?',-sys.maxsize, inplace=True)
df.drop(['id'], 1, inplace=True)
full_data = df.astype(float).values.tolist()

#shuffle data, prep dictionaries to populate train and test sets
# select first 80% as train data and test_data as slicing last 
# 20%. ditionary: 2 is benign, 4 is malignant tumors
random.shuffle(full_data)
test_size=0.2
train_set = {2:[], 4:[]}
test_set = {2:[], 4:[]}
train_data = full_data[:-int(test_size*len(full_data))]
test_data = full_data[-int(test_size*len(full_data)):]

# populate dictionries
[train_set[i[-1]].append(i[:-1]) for i in train_data]
[test_set[i[-1]].append(i[:-1]) for i in test_data]

correct = 0
total = 0

for group in test_set:
    for data in test_set[group]:
        vote = k_nearest_neighbors(train_set, data, k=5)
        if group == vote:
            correct += 1
        total += 1

print('Accuracy:', correct/total)

