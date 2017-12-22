#! /usr/bin/env python
# -*- coding:utf-8
# Custom Implementation of Kmeans
# Special thanks: Harisson@pythonprogramming.net

'''
    File name: kmeans_test.py
    Author: chimney37
    Date created: 09/15/2017
    Python Version: 3.62
'''
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
style.use('ggplot')

# fitment data
X = np.array([[1, 2],
              [1.5, 1.8],
              [5, 8],
              [8, 8],
              [1, 0.6],
              [9, 11],
              [1, 3],
              [8, 9],
              [0, 3],
              [5, 4],
              [6, 4], ])

# plt.scatter(X[:,0],X[:,1], s=150, linewidths=5, zorder=10)
# plt.show()

colors = 10*["g", "r", "c", "b", "k"]


class K_Means:
    def __init__(self, k=2, tol=0.001, max_iter=300):
        self.k = k
        self.tol = tol
        self.max_iter = max_iter

    def fit(self, data):
        self.centroids = {}

        # initialize centroids. If k=2, there are 2 centroids
        for i in range(self.k):
            self.centroids[i] = data[i]

        for i in range(self.max_iter):
            self.classifications = {}

            # initialize classifications as empty list. If k=2, there are 2 classifications
            for i in range(self.k):
                self.classifications[i] = []

            # for each featureset, measure a distance from centroid
            # classify each featureset based on the minimum distance index
            # e.g. if k==2, distances is a 2 dimensional array
            # [distance to centroid1, distance to centroid2]
            # add the featureset to a "classification" that picks
            # the shorter distance
            for featureset in data:
                distances = [np.linalg.norm(featureset-self.centroids[centroid]) for centroid in self.centroids]
                print("distances:", distances)
                classification = distances.index(min(distances))
                # print(classification)
                self.classifications[classification].append(featureset)
            # print(self.classifications)

            prev_centroids = dict(self.centroids)
            # compute the centroid for an iteration through an average of the featuresets added to a "classification"
            for classification in self.classifications:
                self.centroids[classification] = np.average(self.classifications[classification], axis=0)

            # print(self.centroids)

            optimized = True
            # compute movement of centroids, if they are bigger than tolerance
            for c in self.centroids:
                original_centroid = prev_centroids[c]
                current_centroid = self.centroids[c]
                movement = np.sum((current_centroid-original_centroid)/original_centroid*100.0)

                if movement > self.tol:
                    print("movement of centroid:", movement)
                    optimized = False

            print("movement of centroid:", movement)
            if optimized:
                break

    def predict(self, data):
        distances = [np.linalg.norm(data-self.centroids[centroid]) for centroid in self.centroids]
        classification = distances.index(min(distances))
        return classification


# fit data on fitment data
clf = K_Means()
clf.fit(X)

# plot the centroids and the featuresets that were classified
for centroid in clf.centroids:
    plt.scatter(clf.centroids[centroid][0], clf.centroids[centroid][1], marker="o",
                color="k", s=150, linewidths=2)

for classification in clf.classifications:
    color = colors[classification]
    for featureset in clf.classifications[classification]:
        plt.scatter(featureset[0], featureset[1], marker="x", color=color, s=150, linewidths=2)
'''
#unknowns=np.array([[1,3],
#                   [8,9],
#                   [0,3],
#                   [5,4],
#                   [6,4],])

#for unknown in unknowns:
#    classification=clf.predict(unknown)
#    plt.scatter(unknown[0],unknown[1],marker="*",color=colors[classification],s=150,linewidths=5)
'''
plt.show()
