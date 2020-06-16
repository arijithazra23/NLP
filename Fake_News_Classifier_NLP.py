# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 21:18:29 2020

@author: arijit.hazra
"""

import pandas as pd
df = pd.read_csv(r'D:\ML\NLP\NLP Datasets\fake-news\train.csv')

df['label'].value_counts()
df_head= df.head(500)

X =df.drop('label',axis=1)
y= df['label']

initial_shape= df.shape
shape_after_na = df.dropna()

df=df_head.dropna()
X =df.drop('label',axis=1)
y= df['label']

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer

messages = df.copy()
messages.reset_index(inplace=True)
messages['text'][6]

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re
ps = PorterStemmer()
corpus=[]

for i in range(len(messages)):
    review = re.sub('[^a-zA-Z]',' ',messages['text'][i])
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if word not in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)

corpus[3]

tfidf_v= TfidfVectorizer(max_features=2500, ngram_range=(1,3))
X =tfidf_v.fit_transform(corpus).toarray()

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0)

tfidf_v.get_feature_names()[1:20]
tfidf_v.get_params()

data = pd.DataFrame(X_train,columns = tfidf_v.get_feature_names())

import numpy as np
import matplotlib.pyplot as plt
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    See full source and example: 
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    

from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB()

from sklearn import metrics
import numpy as np
import itertools

classifier.fit(X_train,y_train)
pred= classifier.predict(X_test)
score = metrics.accuracy_score(y_test,pred)
print(score)



def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    See full source and example: 
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

cm = metrics.confusion_matrix(y_test,pred)
plot_confusion_matrix(cm,classes = ['FAKE','REAL'])




classifier = MultinomialNB(alpha=0.1)
previous_score=0
for alpha in np.arange(0,1,0.1):
    sub_classifier= MultinomialNB(alpha = alpha)
    sub_classifier.fit(X_train,y_train)
    y_pred = sub_classifier.predict(X_test)
    score = metrics.accuracy_score(y_test,y_pred)
    if score>previous_score:
        classifier=sub_classifier
    print('Alpha: {} , Score : {} '.format(alpha,score))














