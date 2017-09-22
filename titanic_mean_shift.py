#! /usr/bin/env python
# -*- coding:utf-8
# Using Mean Shift against the titanic dataset
# Special thanks: Harisson@pythonprogramming.net

'''
    File name: titanic_mean_shift.py
    Author: chimney37
    Date created: 09/22/2017
    Python Version: 3.62
'''

import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
import pandas as pd
from sklearn.cluster import MeanShift, KMeans
from sklearn import preprocessing, cross_validation

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

#read and print raw contents
df = pd.read_excel('titanic.xls')
original_df = pd.DataFrame.copy(df)
df.drop(['body','name'],1,inplace=True)
df.convert_objects(convert_numeric=True)
df.fillna(0, inplace=True)


# do non-numerical processing and assign numericals to non-numerical values
df = handle_non_numerical_data(df)
df.drop(['ticket','home.dest'],1,inplace=True)

print(df.head())

#Get data into numpy and also drop the survived column when feeding into X
X = np.array(df.drop(['survived'],1).astype(float))
X = preprocessing.scale(X)

#set expected output as survived
y = np.array(df['survived'])

#train using MeanShift
clf = MeanShift()
clf.fit(X)

#get attributes from clf object
labels= clf.labels_
cluster_centers = clf.cluster_centers_

#add a new column to original dataframe
original_df['cluster_group']=np.nan

#populate labels to the empty column
for i in range(len(X)):
    original_df['cluster_group'].iloc[i]=labels[i]

#check the survival rates for each groups we find
n_clusters_ = len(np.unique(labels))
survival_rates={}
for i in range(n_clusters_):
    temp_df = original_df[ (original_df['cluster_group']==float(i))]
    #print(temp_df.head())

    survival_cluster= temp_df[ (temp_df['survived']==1) ]

    survival_rate=len(survival_cluster) / len(temp_df)
    i#print(i,survival_rate)
    survival_rates[i]=survival_rate


#digging deeper into what each class is about
print("Look into what the cluster group 1 is:")
print(original_df[ (original_df['cluster_group']==1) ])

#dig deep into group 0, which is diverse
print("Look into cluster group 0")
print(original_df[ (original_df['cluster_group']==0) ].describe())

#dig deep into group 2
print("Look into cluster group 2")
print(original_df[ (original_df['cluster_group']==2) ].describe())

print("survival rates clusters:",survival_rates)
