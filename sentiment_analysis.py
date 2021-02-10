# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 20:43:05 2021

@author: kamar
"""
import spacy
import pandas as pd 
import numpy as np
import matplotlib
import nltk
import matplotlib
import matplotlib.pyplot as plt
from os import path
from wordcloud import WordCloud


mobile_data = pd.read_csv("D:/Kee cllg/spring 2021/Data mining QMST 5343/assgmnt 2 naive bias txtmining/Amazon_Unlocked_Mobile.csv")

mobile_data.shape
mobile_data.info()
mobile_data.head(5)
mobile_data.columns

x= mobile_data.nunique()
x

x= mobile_data.isnull().sum()
x
mobile_dataclean = mobile_data.dropna()
mobile_dataclean.isnull().sum()

goodr_df = mobile_dataclean[mobile_dataclean['Rating']>=4]
goodr_df.info()
goodr_df.head()
goodr_df.shape[0]

goodr_dfnew = goodr_df.reset_index(drop=True)
goodr_dfnew.head()

badr_df = mobile_dataclean[mobile_dataclean['Rating']<4]
badr_df.info()
badr_df.head()
badr_df.shape[0]

badr_dfnew = badr_df.reset_index(drop=True)
badr_dfnew.head()
badr_dfnew['Rating']

#########################################
R_fr = mobile_dataclean['Rating']==4
R_fv = mobile_dataclean['Rating']==5

Goodratng_data = mobile_dataclean[R_fr|R_fv]
Goodratng_data.info()

Goodratng_ndf = Goodratng_data.reset_index(drop=True)
Goodratng_ndf.info()

Badratng_data = mobile_dataclean[~(R_fr|R_fv)]
Badratng_data['Rating']
#####################
nltk.download()
from nltk.tokenize import RegexpTokenizer    #this is func name we can use wthout havng to type nltk.tokenize.regexptokenizer

# Load library
from nltk.corpus import stopwords

stop_words = stopwords.words('english')
print(stop_words)
extraSW=('How','What','I','the','may','to','a','in','Why','is','get','Which','why','is','Is','would','If')
for i in range(len(extraSW)):
 stop_words.append(extraSW[i])
# Show stop words
stop_words[:5]
#############
texttotokens=[]
stemmedtexttotokens=[]

tokenizer = RegexpTokenizer(r'[A-Za-z]+')

goodr_dfnew.shape[1]
goodr_dfnew['Reviews'][0]
goodr_dfnew[1]
for i in range(goodr_dfnew.shape[0]):
    texttotoken=goodr_dfnew.Reviews[i]
    texttotokens.append(tokenizer.tokenize(texttotoken))
print(texttotokens[0])
flat_list = []
for sublist in texttotokens:
    for item in sublist:
        flat_list.append(item)
##############################
#for good r
frequency_dist = nltk.FreqDist([word for word in tokenizer.tokenize(str(flat_list)) if word not in stop_words]) 

top50n=sorted(frequency_dist,key=frequency_dist.__getitem__, reverse=True)[0:50]
numberoftimes=[]
for i in range(50):
 numberoftimes.append(frequency_dist[top50n[i]])
 
 frequency_dist.B()
#220996
frequency_dist.plot(40)


listofdocs= []
#good
wordcloud_good = WordCloud(stopwords=stop_words).generate_from_frequencies(frequency_dist)
plt.imshow(wordcloud_good, interpolation='bilinear')
plt.axis("off")
plt.show()
##############################
#for bad r
texttotokensbad=[]
stemmedtexttotokens=[]

tokenizer = RegexpTokenizer(r'[A-Za-z]+')

badr_dfnew.shape[1]
badr_dfnew['Reviews'][0]
#badr_dfnew[1]
for i in range(badr_dfnew.shape[0]):
    texttotoken=badr_dfnew.Reviews[i]
    texttotokensbad.append(tokenizer.tokenize(texttotoken))
print(texttotokens[0])
flat_listbad = []
for sublist in texttotokens:
    for item in sublist:
        flat_listbad.append(item)
frequency_dist_bad = nltk.FreqDist([word for word in tokenizer.tokenize(str(flat_listbad)) if word not in stop_words]) 

top50n=sorted(frequency_dist_bad,key=frequency_dist_bad.__getitem__, reverse=True)[0:50]
numberoftimes=[]
for i in range(50):
 numberoftimes.append(frequency_dist_bad[top50n[i]])
 
 frequency_dist_bad.B()
#220996
frequency_dist_bad.plot(40)

listofdocs= []
#good
wordcloud_bad = WordCloud(stopwords=stop_words).generate_from_frequencies(frequency_dist_bad)
plt.imshow(wordcloud_bad, interpolation='bilinear')
plt.axis("off")
plt.show()
###########################
wordcloud = WordCloud(stopwords=stop_words).generate(str(Goodratng_ndf.Reviews))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

wordcloud = WordCloud(stopwords=stop_words).generate(str(Badratng_data.Reviews))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
############################
#6Qstn
subsetted=pd.DataFrame()        
def subsetter(datasetname,idc,value):   
 newdatasetname=datasetname[datasetname[idc]==value]
 return newdatasetname    

goodr_dfnew.shape[0]
goodr_dfnew[goodr_dfnew['Rating']][1]
goodr_dfnew['Rating'][1]

subsetted=pd.DataFrame()        
def subsetter(datasetname,idc,value):   
 for i in range(datasetname.shape[0]):
     newdatasetname=datasetname[datasetname[idc]==value]
     return newdatasetname    

newdatasetname = pd.DataFrame()
for i in range(goodr_dfnew.shape[0]):
    print(goodr_dfnew['Rating'][i])
    newdatasetname = goodr_dfnew[goodr_dfnew['Rating'][i]==1]
    

subsetted=subsetter(goodr_dfnew,'Rating',1)
subsetted.info()
####
goodr_df = goodr_dfnew.copy()
replaceStruct = {
                "Rating":     {4: 1, 5 :1}
                }
goodr_dfn=goodr_df.replace(replaceStruct)
goodr_dfn.Rating
goodr_dfn.head()

badr_df = badr_dfnew.copy()
replaceStruct = {
                "Rating":     {0: 0, 1 :0, 2:0, 3:0}
                }
badr_dfn=badr_df.replace(replaceStruct)
badr_dfn.Rating
badr_dfn.head()
###############################
#mobile_dataclean

mobiledata_newr = mobile_dataclean.copy()
replaceStruct = {
                "Rating":     {0: 0, 1 :0, 2:0, 3:0, 4: 1, 5 :1}
                }
mobiledata_newreplcd=mobiledata_newr.replace(replaceStruct)
mobiledata_newreplcd.head()

playsetr=mobiledata_newreplcd.sample(int(mobiledata_newreplcd.shape[0]*.8))
playsetr.shape[0]
from sklearn.model_selection import train_test_split
pX_train, pX_test, py_train, py_test = train_test_split(playsetr['Reviews'], playsetr['Rating'], train_size=int(playset.shape[0]*.7),random_state=1)

#################################################################

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(stop_words='english',binary=True,lowercase=True,
strip_accents='unicode',token_pattern=r'[A-Za-z]+',tokenizer=my_tokenizer)

#since no spcay

import re
REGEX = re.compile(r",\s*")
def tokenize(text):
    return [tok.strip().lower() for tok in REGEX.split(text)]

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(stop_words='english',binary=True,lowercase=True,
token_pattern=r'[A-Za-z]+',tokenizer=tokenize)



X_train_cv = cv.fit_transform(pX_train)
y_train=py_train
########################################################
from sklearn.naive_bayes import BernoulliNB
naive_bayes = BernoulliNB()
#.fit instead of .fit_transform is necc to ignore the 0 prob
naive_bayes.fit(X_train_cv, y_train)

X_test_cv = cv.transform(pX_test)
predictions = naive_bayes.predict(X_test_cv)
predictions[12]
####################
from sklearn.metrics import accuracy_score, precision_score, recall_score
print('Accuracy score: ', accuracy_score(py_test, predictions))
print('Precision score: ', precision_score(py_test, predictions))
print('Recall score: ', recall_score(py_test, predictions))
nmpredictions=[]
for j in predictions:
    nmpredictions.append(0)
nmpredictions=pd.array(nmpredictions)