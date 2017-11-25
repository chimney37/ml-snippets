#! /usr/bin/env python
# -*- coding:utf-8
# Against Large dataset, create a feature set for identifying positive or negative sentiments using neural networks
# Special thanks: Harisson@pythonprogramming.net

'''
    File name: data_preprocessing.py
    Author: chimney37
    Date created: 11/23/2017
    Python Version: 3.62
'''
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from collections import Counter
import pickle
import numpy as np
import pandas as pd

'''
Notes on the dataset: Stanford's 1.6MM examples
http://help.sentiment140.com/for-students/
polarity 0 : negative. 2 : neutral. 4 : positive
id
date
query
user
tweet
'''

class SentimentPreProcess():
    def __init__(self):
        self.data_delimiter=':;#$%&'
        self.lemmatizer=WordNetLemmatizer()

    def init_process(self,fin,fout):
        outfile = open(fout,'w')
        with open(fin, buffering=200000, encoding='latin-1') as f:
            try:
                for line in f:
                    line = line.replace('"','')
                    elements=line.split(',')
                    initial_polarity = elements[0]
                    if initial_polarity == '0':
                        initial_polarity = [1,0]
                    elif initial_polarity == '4':
                        initial_polarity = [0,1]

                    tweet = elements[-1]
                    outline = str(initial_polarity)+self.data_delimiter+tweet
                    outfile.write(outline)
            except Exception as e:
                print(str(e))
        outfile.close()

    def create_lexicon(self,fin):
        lexicon=[]
        with open(fin,'r',buffering=100000, encoding='latin-1') as f:
            try:
                counter=1
                content=''
                for line in f:
                    counter+=1
                    # random out of every 2500 samples
                    if(counter/2500.0).is_integer():
                        tweet=line.split(self.data_delimiter)[1]
                        content+=' '+tweet
                        words = word_tokenize(content)
                        words = [self.lemmatizer.lemmatize(i).lower() for i in words]
                        lexicon=list(set(lexicon+words))
                        print(counter,len(lexicon))

            except Exception as e:
                print(str(e))

        with open('lexicon.pickle','wb') as f:
            pickle.dump(lexicon,f)

    def convert_to_vec(self,fin,fout,lexicon_pickle):
        with open(lexicon_pickle,'rb') as f:
            lexicon = pickle.load(f)
        outfile = open(fout,'w')
        with open(fin,buffering=20000,encoding='latin-1') as f:
            counter=0
            for line in f:
                counter+=1
                label=line.split(self.data_delimiter)[0]
                tweet=line.split(self.data_delimiter)[1]
                current_words=word_tokenize(tweet.lower())
                current_words=[self.lemmatizer.lemmatize(i) for i in current_words]

                features=np.zeros(len(lexicon))

                for word in current_words:
                    if word.lower() in lexicon:
                        index_value=lexicon.index(word.lower())
                        features[index_value]+=1

                features=list(features)
                outline=str(features)+self.data_delimiter+str(label)+'\n'
                outfile.write(outline)

            print("Size of test data:",str(counter))

    def shuffle_data(self,fin):
        df=pd.read_csv(fin,names=['sentiment','tweet'],error_bad_lines=False)
        df=df.iloc[np.random.permutation(len(df))]
        print(df.head())
        print(Counter(df['sentiment']))
        df.to_csv('train_set_shuffled.csv',header=False,index=False)

    def create_test_data_pickle(self,fin):
        feature_sets=[]
        labels=[]
        counter=0
        with open(fin,buffering=20000) as f:
            for line in f:
                try:
                    features=eval(line.split(self.data_delimiter)[0])
                    label=eval(line.split(self.data_delimiter)[1])

                    feature_sets.append(features)
                    labels.append(label)
                    counter+=1
                except Exception as e:
                    print(str(e))
        print("Test Data size for creating pickle:",str(counter))
        feature_sets=np.array(feature_sets)
        labels=np.array(labels)
        with open('processed-test-set-feature-sets.pickle','wb') as f:
            pickle.dump(feature_sets,f)
        with open('processed-test-set-labels.pickle','wb') as f:
            pickle.dump(labels,f)

    def generate_all_data(self):
        #create training and test set
        self.init_process('training.1600000.processed.noemoticon.csv','train_set.csv')
        self.init_process('testdata.manual.2009.06.14.csv','test_set.csv')

        self.create_lexicon('train_set.csv')

        #create a pre-processed vectorized set for testing
        self.convert_to_vec('test_set.csv','processed-test-set.csv','lexicon.pickle')
        #shuffle the training set
        self.shuffle_data('train_set.csv')
        #create test data and pickle using processed test set
        self.create_test_data_pickle('processed-test-set.csv')


if __name__ == '__main__':
    process = SentimentPreProcess()
    process.generate_all_data()

