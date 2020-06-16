# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 14:36:17 2020

@author: arijit.hazra
"""

import pandas as pd

messages = pd.read_csv(r'D:\ML\NLP\NLP Datasets\smsspamcollection\SMSSpamCollection',sep='\t',
                       names = ['label','message'])

## Data Cleaning and preprocessing

import re
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

ps = PorterStemmer()
corpus = []
for i in range(len(messages)):
    review = re.sub('[^a-zA-Z]',' ',messages['message'][i])
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if word not in stopwords.words('english')]
    review = ' '.join(review)  
    corpus.append(review)
    
from sklearn.feature_extraction.text import CountVectorizer

cv =CountVectorizer(max_features=2500)
X= cv.fit_transform(corpus).toarray()

y= pd.get_dummies(messages['label'])
y =y.iloc[:,1].values

from sklearn.model_selection import train_test_split
X_train, X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)

from sklearn.naive_bayes import MultinomialNB
mn = MultinomialNB()
spam_detect = mn.fit(X_train,y_train)
y_pred = spam_detect.predict(X_test)

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
confusion_m = confusion_matrix(y_test,y_pred)

classification_score = classification_report(y_test,y_pred)
accuracy_score      =  accuracy_score(y_test,y_pred)
