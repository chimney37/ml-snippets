#! /usr/bin/env python
# -*- coding:utf-8
# K nearest neighbor custom function
# Special thanks: Harisson@pythonprogramming.net

'''
    File name: cust_knn.py
    Author: chimney37
    Date created: 8/12/2017
    Python Version: 3.62
'''
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import warnings
from math import sqrt
from collections import Counter

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

style.use('fivethirtyeight')

# dict with keys(the class) and points which are attributed with class
dataset = {'k':[[1,2],[2,3],[3,1]], 'r':[[6,5],[7,7],[8,6]]}
new_features = [5,7]

#predict for sample the class
result = k_nearest_neighbors(dataset, new_features)
print('predicted class for sample is ', result)

#scatter plot by iterating through each class of points
[[plt.scatter(ii[0], ii[1],s=100,color=i) for ii in dataset[i]] for i in dataset]

#scatter plot the test set
plt.scatter(new_features[0], new_features[1], s=100, color=result)

plt.show()

