#! /usr/bin/env python
# -*- coding:utf-8
# SVR and LinearRegression using quandl data.
# Special thanks: Harisson@pythonprogramming.net

'''
    File name: svr_test.py
    Author: chimney37
    Date created: 8/10/2017
    Python Version: 3.62
'''

import datetime
import sys
import time
import argparse
import numpy as np
import pandas as pd
import quandl,math
import matplotlib.pyplot as plt
import pickle

from sklearn.linear_model import LinearRegression
from sklearn import preprocessing, cross_validation, svm

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='loads price data, trains or predicts 0.01 into the future using support vector regression')
    parser.add_argument('-l','--load',dest='flag_exists', action='store_true', help='setting the flag loads a saved model instead of training the model. First time runs should not use this flag', required=False)
    args = parser.parse_args()

#use popular plotting package for R
plt.style.use('ggplot')

start = time.time()

#get quandl financial data which contain stock prices
#df is pandas dataframe
df = quandl.get("WIKI/AMZN")


#just get following column
#Adj is adjusted for corporate actions (e.g. splits, mergers, spinoffs)
#Adj. Open:opening price. opening price at which a security is traded on given trading day
#Adj. Close:closing price.final price at which a security is traded on a given trading day
#Adj. High:highest price at which a stock traded during the course of day
#Adj. Low:lowest price at which a stock traded during course of day
#Adj. Volume:number of transactions during the trading day
df = df[['Adj. Open',  'Adj. High',  'Adj. Low',  'Adj. Close', 'Adj. Volume']]

#get high low percentage change
df['HL_P_change'] = (df['Adj. High'] - df['Adj. Low']) / df['Adj. Low'] * 100.0

#get low high percentage change
df['LH_P_change'] = (df['Adj. Low'] - df['Adj. High']) / df['Adj. High'] * 100.0

#get close open percentage change
df['CO_P_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0

#get adjusted columns and calculated percentage changes
df= df[['Adj. Close','HL_P_change','LH_P_change','CO_P_change','Adj. Volume']]


#use Adj. Close as the thing we want to predict
forecast_col='Adj. Close'

#fill missing data with outliers (which should be ignored)
df.fillna(value=-sys.maxsize,inplace=True)

#set forecast out count to a ratio of length of dataframe
forecast_out=int(math.ceil(0.01*len(df)))

# set a prediction label column by shifting the entire prediction column by the
# forecast out count
df['label'] = df[forecast_col].shift(-forecast_out)

print('Loading from quandl, add columns, set up prediction columns',time.time() - start, ' secs')


# create the arrays used for training and validation
# create array by dropping the prediction column
X = np.array(df.drop(['label'],1))
# normalize array values between 0 and 1
X = preprocessing.scale(X)
# take the most latest rows (the last forecast counts) as the rows we want to
# predict
X_lately = X[-forecast_out:]
# Reassign array to represent rows from the beginning to the value excluding
# the forecast count
X = X[:-forecast_out]

# Create an array of the prediction label column up to the count excluding the
# forecast count
df.dropna(inplace=True)
y=np.array(df['label'])

start = time.time()

if(vars(args)["flag_exists"]):
    pickle_in = open('regression_model.pickle', 'rb')
    clf = pickle.load(pickle_in)
    print('Loading previously saved model', time.time() - start, ' secs')
else:
    # create the training and test(validation) sets
    # test_size is proportion of the dataset to use as test(validation) samples
    # train and calculate confidence
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.3)

    #try out kernels in SVR
    kernel_confidence = dict()
    for k in ['linear','poly','rbf','sigmoid']:
        clf = svm.SVR(kernel=k)
        clf.fit(X_train, y_train)
        confidence = clf.score(X_test, y_test)
        kernel_confidence[k]=confidence
        print(k, confidence)

    kernel_confidence_sorted = sorted(kernel_confidence.items(), key=lambda x:
                                  x[1], reverse=True)

    #Predict using the best kernel and fill the forecast set with the predicted data 
    clf = svm.SVR(kernel=kernel_confidence_sorted[0][0])
    clf.fit(X_train,y_train)

    #Save the model
    with open('regression_model.pickle','wb') as f:
        pickle.dump(clf, f)
    print('Saved model')

    print('Training, Evaluating, Saving model', time.time() - start, ' secs')


start = time.time()
#create forecast
forecast_set = clf.predict(X_lately)

df['Prediction']=np.nan

# get the next unix date for creating the prediction rows
last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day=86400
next_unix = last_unix + one_day

# Fill the prediction rows with the forecast data
for i in forecast_set:
    next_date=datetime.datetime.fromtimestamp(next_unix)
    next_unix += 86400
    df.loc[next_date]=[np.nan for _ in range(len(df.columns) -1)] + [i]

print('Prediction and filling forecast data', time.time() - start, ' secs')

print('Plotting data...See separate window')
#Plot data
df['Adj. Close'].plot()
df['Prediction'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()

