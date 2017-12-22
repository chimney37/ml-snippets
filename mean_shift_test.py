#! /usr/bin/env python
# -*- coding:utf-8
# Illustrating the use of Mean Shift
# Special thanks: Harisson@pythonprogramming.net

'''
    File name: mean_shift_test.py
    Author: chimney37
    Date created: 09/22/2017
    Python Version: 3.62
'''

import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
from sklearn.cluster import MeanShift
from sklearn.datasets.samples_generator import make_blobs
from mpl_toolkits.mplot3d import Axes3D
style.use('ggplot')

centers = [[1, 1, 1], [5, 5, 5], [3, 10, 10]]

X, _ = make_blobs(n_samples=100, centers=centers, cluster_std=1.5)

'''
Quick explanation: Mean Shift works by radius bandwidths. For every featureset, there is going to be considered as part of a cluster.
We keep adding a feature set and we get a new cluster center. When the cluster center does not move, it is optimized by convergence(we get a cluster)
'''
ms = MeanShift()
ms.fit(X)
labels = ms.labels_
cluster_centers = ms.cluster_centers_

print(cluster_centers)
n_clusters_ = len(np.unique(labels))
print("Number of estimated clusters:", n_clusters_)

colors = 10*['r', 'g', 'b', 'c', 'k', 'y', 'm']
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for i in range(len(X)):
    ax.scatter(X[i][0], X[i][1], X[i][2], c=colors[labels[i]], marker='o')

ax.scatter(cluster_centers[:, 0],
           cluster_centers[:, 1], cluster_centers[:, 2], marker="x", color='k', s=150, linewidths=5, zorder=10)

plt.show()
