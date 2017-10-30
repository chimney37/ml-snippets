#! /usr/bin/env python
# -*- coding:utf-8
# Custom Kmeans Shift clustering, with dynamic radius
# Special thanks: Harisson@pythonprogramming.net

'''
    File name: cust_kmeans_shift_dynamic.py
    Author: chimney37
    Date created: 10/30/2017
    Python Version: 3.62
'''
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
from sklearn.datasets.samples_generator import make_blobs
style.use('ggplot')

X, y= make_blobs(n_samples=50, centers=3, n_features=2)
"""X = np.array([[1,2],
              [1.5,1.8],
              [5,8],
              [8,8],
              [1,0.6],
              [9,11],
              [8,2],
              [10,2],
              [9,3],])
"""
colors=10*["g","r","c","b","k"]

class Mean_Shift:
    #plan is to crete a massive radius, but let it go in steps
    def __init__(self, radius=None, radius_norm_step=100):
        self.radius = radius
        self.radius_norm_step = radius_norm_step

    def fit(self, data):
        if self.radius == None:
            #find the "center" of all data
            all_data_centroid = np.average(data, axis=0)
            # take the norm of the data (maginitude of data from origin)
            all_data_norm = np.linalg.norm(all_data_centroid)
            #start with a radius (all data norm divided by step
            self.radius = all_data_norm / self.radius_norm_step

        centroids = {}

        #Make all datapoints centroids
        for i in range(len(data)):
            centroids[i] = data[i]

        #get a list from 0 to 99, reversed
        weights = [i for i in range(self.radius_norm_step)][::-1]

        while True:
            new_centroids=[]
            for i in centroids:
                in_bandwidth=[]
                centroid = centroids[i]

                for featureset in data:
                    #calculate the full distance between featurset and centroid
                    distance = np.linalg.norm(featureset-centroid)
                    #solve an initialization problem. featureset is compared to itself
                    if distance ==0:
                        distance = 0.00000000001

                    #weight index is the index computed by entire distance divided by the radial step length (the bigger the distance, the larger the index,)hence towards 99
                    weight_index=int(distance/self.radius)

                    #if weight index is beyond maximum, all of the bounds
                    if weight_index > self.radius_norm_step-1:
                        weight_index = self.radius_norm_step-1

                    #add the "weighted" number of centroids to the in_bandwidth
                    to_add = (weights[weight_index]**2)*[featureset]
                    in_bandwidth += to_add
                new_centroid=np.average(in_bandwidth,axis=0)
                new_centroids.append(tuple(new_centroid))

            uniques = sorted(list(set(new_centroids)))

            to_pop=[]

            #remove uniques where the difference is less than the radial step distance
            for i in uniques:
                for ii in [i for i in uniques]:
                    if i == ii:
                        pass
                    elif np.linalg.norm(np.array(i)-np.array(ii)) < self.radius:
                        to_pop.append(ii)
                        break
            
            for i in to_pop:
                try:
                    uniques.remove(i)
                except:
                    pass

            prev_centroids = dict(centroids)
            centroids={}
            for i in range(len(uniques)):
                centroids[i]=np.array(uniques[i])

            optimized = True

            #compare previous centroids to the previous ones, and measure movement.
            for i in centroids:
                #if centroid moved, not converged
                if not np.array_equal(centroids[i], prev_centroids[i]):
                    optimized=False
                if not optimized:
                    break

            #get out of loop when converged
            if optimized:
                break

        #we expect fit to also classify the existing featureset
        self.centroids=centroids
        self.classifications = {}

        for i in range(len(self.centroids)):
            self.classifications[i]=[]

        for featureset in data:
            #compare data to either centroid
            distances = [np.linalg.norm(featureset-self.centroids[centroid]) for centroid in self.centroids]
            classification = (distances.index(min(distances)))
            #featureset that belongs to the cluster
            self.classifications[classification].append(featureset)

    def predict(self,data):
        #compare data to either centroid
        distances = [np.linalg.norm(featuset-self.centroids[centroid]) for centroid in self.centroids]
        classification = (distances.index(min(distances)))
        return classification


clf = Mean_Shift()
clf.fit(X)

centroids = clf.centroids
print(centroids)

for classification in clf.classifications:
    color = colors[classification]
    for featureset in clf.classifications[classification]:
        plt.scatter(featureset[0],featureset[1],marker="x",color=color,s=150,linewidths=5,zorder=10)

for c in centroids:
    plt.scatter(centroids[c][0],centroids[c][1],color='k',
                marker="*",s=150,linewidths=5)

plt.show()
