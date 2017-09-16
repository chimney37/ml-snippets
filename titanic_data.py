#! /usr/bin/env python
# -*- coding:utf-8
# Handling non-numerical data (titanic.xls) and KMeans to predict survivability 
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
import cust_kmeans as ckm 
from sklearn.cluster import KMeans
from sklearn import preprocessing, cross_validation
import pandas as pd
style.use('ggplot')


'''
Pclass Passenger Class (1 = 1st; 2 = 2nd; 3 = 3rd)
survival Survival (0 = No; 1 = Yes)
name Name
sex Sex
age Age
sibsp Number of Siblings/Spouses Aboard
parch Number of Parents/Children Aboard
ticket Ticket Number
fare Passenger Fare (British pound)
cabin Cabin
embarked Port of Embarkation (C = Cherbourg; Q = Queenstown; S = Southampton)
boat Lifeboat
body Body Identification Number
home.dest Home/Destination
'''
#read and print raw contents
df = pd.read_excel('titanic.xls')
print(df.head())
df.drop(['body','name'],1,inplace=True)
df.convert_objects(convert_numeric=True)
df.fillna(0, inplace=True)
print(df.head())

def handle_non_numerical_data(df):
    columns= df.columns.values

    for column in columns:
        text_digit_vals = {}

        #memozation of data using a dict
        def convert_to_int(val):
            return text_digit_vals[val]

        #check data type of each column
        if df[column].dtype != np.int64 and df[column].dtype != np.float64:
            column_contents = df[column].values.tolist()
            #get a set of unique elements from each column
            unique_elements = set(column_contents)
            x=0
            #assign an index for each unique element not already tracked in dictionary
            for unique in unique_elements:
                if unique not in text_digit_vals:
                    text_digit_vals[unique]=x
                    x+=1

            df[column] = list(map(convert_to_int,df[column]))

    return df

# do non-numerical processing and assign numericals to non-numerical values
df = handle_non_numerical_data(df)
print(df.head())

#drop features to see impact
df.drop(['ticket','home.dest'],1,inplace=True)

#training data, take out the survived since that is what we want predicted
X=np.array(df.drop(['survived'],1).astype(float))
X=preprocessing.scale(X)
y=np.array(df['survived'])

print(y)

#clf = ckm.K_Means()
clf = KMeans(n_clusters=2)
clf.fit(X)

# a bit strange as the predictio can be 0 or 1 (switched?)
correct = 0
for i in range(len(X)):
    predict_me=np.array(X[i].astype(float))
    predict_me=predict_me.reshape(-1,len(predict_me))
    prediction=clf.predict(predict_me)
    if prediction==y[i]:
        correct+=1

print(correct/len(X))
