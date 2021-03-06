#! /usr/bin/env python
# -*- coding:utf-8
# Introduction to SVM. Build SVM from scratch
# Special thanks: Harisson@pythonprogramming.net

'''
    File name: svm_cust.py
    Author: chimney37
    Date created: 9/01/2017
    Python Version: 3.62
'''
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
style.use('ggplot')

'''
    Class for SVM:
    We are trying to find the hyperplane that best divides a group    of data, 'r'
    and 'b'. We will determine w and b in the given equation wx + b, where w is the
    perpendiular vector to the hyperplane, and b is the bias. Given this is a
    lagrangian constraint optimization problem, we will solve this
    as a convex problem (i.e to find the global minimum) for ||w||maximize b and
    , having a constraint such that  y (class) = w.x + b >= 1
'''


class Support_Vector_Machine:
    def __init__(self, visualization=True):
        self.visualization = visualization
        self.colors = {1: 'r', -1: 'b'}
        if self.visualization:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(1, 1, 1)

    # train
    def fit(self, data):
        self.data = data
        # { ||w||: [w,b] }. Remember the purpose is to minimize ||w|| and maximize b
        # (lagrangian constraint optimization problem)
        # This is a quadratic problem, so we woud solve this as a convex problem.
        # The plan is to slowly step down the magnitude of a given vector

        # optimization dictionary
        opt_dict = {}

        # vector can have same magnitude but different dot product with featuresets
        # and so a very different resultant hyperplane
        transforms = [[1, 1],
                      [-1, 1],
                      [-1, -1],
                      [1, -1]]

        # finding values to work with for our ranges
        # yi are the keys in the dictionary, which is a featureset  representing key
        # the values of the dictionary is the featureset itself which contains an array of features
        # feature is a vector [x,y]
        # value is each number in the vector
        all_data = [values for yi in self.data for feature in self.data[yi] for values in feature]

        self.max_feature_value = max(all_data)
        self.min_feature_value = min(all_data)

        # can discard this memory
        all_data = None

        # create steps to solve the convex problem.
        # We have 3 levels of steps, that progressively gets more fine.
        step_sizes = [self.max_feature_value * 0.1,
                      self.max_feature_value * 0.01,
                      # starts getting very high cost after this
                      self.max_feature_value * 0.001]

        # Set b value modifiers: extremely expensive, so we make b larger
        # we can do this because the precision for b is not as important as w
        b_range_multiple = 2
        b_multiple = 5
        latest_optimum = self.max_feature_value*10

        # begin stepping with the lowest granularity, then progress to more finer
        # step sizes
        for step in step_sizes:
            # intialize w with values
            w = np.array([latest_optimum, latest_optimum])
            # we can do this because convex
            optimized = False
            while not optimized:
                # np.arange function creates an array range given a start (negative
                # value of multipled maximum feature value), stop(positive value) and step
                for b in np.arange(-1*(self.max_feature_value*b_range_multiple),
                                   self.max_feature_value*b_range_multiple,
                                   step*b_multiple):
                    # transform w using each transform
                    for transformation in transforms:
                        w_t = w*transformation
                        found_option = True

                        # check constraint requirement, yi(xi.w+b) >= 1
                        # can add a break if any feature does not satisfy constraint..
                        for i in self.data:
                            for xi in self.data[i]:
                                yi = i
                                if not yi*(np.dot(w_t, xi)+b) >= 1:
                                    found_option = False
                                    break
                                    # print(xi,':',yi*(np.dot(w_t,xi)+b))

                        # if constraints is satisfied, register a key using matrix
                        # norm (here it simply refers to ||w_t|| the magnitude) and the values as a possible w(transformed) and b
                        if found_option:
                            opt_dict[np.linalg.norm(w_t)] = [w_t, b]
                            # print(opt_dict)

                # once we pass 0, no reason to continue
                # reason is that we find optimal values of the transforms (the
                # negatives), so past 0 means we are complete in terms of
                # optimization using a particular step
                if w[0] < 0:
                    optimized = True
                    print('Optimized a step:' + str(step))
                else:
                    w = w - step

            # Once we get all candidates for a given step, we get the sorted keys which are the norm
            # magnitude of w, or ||w||.  We want the value of w which corresponds to
            # the smallest ||w||,so we use the smallest value of the norm,
            # norms[0], and its corresponding b. Also the latest_optimum in the convex
            # curve is updated to take a value x in the w vector.
            # comment: what is the purpose of +step*2?
            norms = sorted([n for n in opt_dict])
            opt_choice = opt_dict[norms[0]]
            self.w = opt_choice[0]
            self.b = opt_choice[1]
            latest_optimum = opt_choice[0][0]+step*2

        # see what the feature corresponds in terms of the constraint
        # this print out is to illustrate that the training data met the constraint
        # for support vectors
        for i in self.data:
            for xi in self.data[i]:
                yi = i
                print(xi, ':', yi*(np.dot(self.w, xi)+self.b))

    # prediction is simply sign(x.w + b) for SVM
    # also visualizes the points for prediction
    def predict(self, features):
        classification = np.sign(np.dot(np.array(features), self.w)+self.b)
        # if classification isn't 0 and we have visualization, we graph
        if classification != 0 and self.visualization:
            self.ax.scatter(features[0], features[1], s=100, marker='*',
                            c=self.colors[classification])
        else:
            print('featureset', features, 'is on the decision boundary')
        return classification

    # draw a few things:
    # 1. positive support vector hyperplane
    # 2. negative support vector hyperplane
    # 3. decision boundary hyperplane
    # 4. featuresets data (training data)
    def visualize(self):
        # scattering known featuresets
        [[self.ax.scatter(x[0], x[1], s=50, color=self.colors[i]) for x in
          data_dict[i]] for i in data_dict]

        def hyperplane(x, w, b, v):
            # v = (w.x+b)
            return (-w[0]*x-b+v)/w[1]

        datarange = (self.min_feature_value*0.9, self.max_feature_value*1.1)
        hyp_x_min = datarange[0]
        hyp_x_max = datarange[1]

        # (w.x + b) = 1
        # positive support vector hyperplane
        psv1 = hyperplane(hyp_x_min, self.w, self.b, 1)
        psv2 = hyperplane(hyp_x_max, self.w, self.b, 1)
        self.ax.plot([hyp_x_min, hyp_x_max], [psv1, psv2], 'k')
        # (w.x + b) = -1
        # negative support vector hyperplane
        nsv1 = hyperplane(hyp_x_min, self.w, self.b, -1)
        nsv2 = hyperplane(hyp_x_max, self.w, self.b, -1)
        self.ax.plot([hyp_x_min, hyp_x_max], [nsv1, nsv2], 'k')

        # (w.x + b) = 0
        # decision boundary hyperplane
        db1 = hyperplane(hyp_x_min, self.w, self.b, 0)
        db2 = hyperplane(hyp_x_max, self.w, self.b, 0)
        self.ax.plot([hyp_x_min, hyp_x_max], [db1, db2], 'y--')

        plt.show()


# starting data
# featureset -1 and 1, being a dictionary with values as an array of feature points.
data_dict = {-1: np.array([[1, 7], [2, 8], [3, 8], ]),
             1: np.array([[5, 1], [6, -1], [7, 3], ])}

svm = Support_Vector_Machine()
svm.fit(data=data_dict)

predict_us = [[0, 10],
              [1, 3],
              [3, 4],
              [3, 5],
              [5, 5],
              [5, 6],
              [6, -5],
              [5, 8]]

for p in predict_us:
    svm.predict(p)

svm.visualize()
