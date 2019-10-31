#!/usr/bin/env python
# coding: utf-8

# In[167]:


import os
import gensim
import string
import nltk
import itertools
import json
import pandas as pd
import gzip

from nltk.stem.porter import PorterStemmer
from string import digits
from statistics import mean 
from nltk.corpus import stopwords
from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn import model_selection, naive_bayes, svm

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


# In[144]:


def parse(path):
    g = gzip.open(path, 'rb')
    for l in g:
        yield json.loads(l)

def getDF(path):
    i = 0
    df = {}
    for d in parse(path):
        df[i] = d
        i += 1
    return pd.DataFrame.from_dict(df, orient='index')

df = getDF('All_Beauty_5.json.gz')
# df = getDF('Books_5.json.jz')
df.head()


# In[145]:


# Cut size of df down if necessary
if len(df) > 500000:
    df = df[:500000]


# In[146]:


df = df[['overall', 'reviewText']]
df['reviewText'] = df['reviewText'].fillna("")


# In[147]:


df['label'] = [1 if x >= 3 else 0 for x in df['overall']]
df.head(10)


# In[148]:


# Clean & normalize dataset
# normalization (e.g. convert to lowercase, remove non-alphanumeric chars, numbers,
textdata = df['reviewText']

def strip(textdata):
    textdata_stripped = []

    for l in textdata:
        # split into words by white space
        words = l.split()
        # remove punctuation from each word
        table = str.maketrans('', '', string.punctuation)
        textdata_stripped.append([w.translate(table) for w in words])
    
    return textdata_stripped

def to_lower(textdata):
    textdata_lower = []
    
    for l in textdata:
        # convert to lower case
        textdata_lower.append([word.lower() for word in l])
    
    return textdata_lower

def rm_stopwords(textdata):
    # filter out stop words
    words = []
    stop_words = set(stopwords.words('english'))
    
    for l in textdata:
        words.append([word for word in l if not word in stop_words])
    
    return words

def rm_numbers(textdata):
    words = []
    remove_digits = str.maketrans('', '', digits)
    
    for l in textdata:
        words.append([word.translate(remove_digits) for word in l])
    
    return words
    
def stem_words(textdata):
    words = []
    porter = PorterStemmer()
    
    for l in textdata:
        words.append([porter.stem(word) for word in l])

    return words

textdata = strip(textdata)
print("Removed punctuation. New length =", len(textdata))
textdata = to_lower(textdata)
print("Converted to lowercase. New length =", len(textdata))
textdata = rm_stopwords(textdata)
print("Removed stopwords. New length =", len(textdata))
textdata = rm_numbers(textdata)
print("Removed numbers. New length =", len(textdata))
textdata = stem_words(textdata)
print("Stemmed words. New length =", len(textdata))

print(textdata[0:30])


# In[149]:


df['reviews_cleaned'] = textdata
df['reviews_cleaned'] = df.reviews_cleaned.apply(' '.join)
df = df[['label', 'reviews_cleaned']]
df.head()


# In[151]:


# Check for class imbalance
print('Label = 0: ', df[df['label'] == 1].shape) # these are good reviews
print('Label = 1: ', df[df['label'] == 0].shape) # these are bad reviews
# There is major class imbalance - will need to balance later


# In[152]:


def get_tfidf(df = df, ngram_range = (1, 2), min_df = 500):
    '''Get tfidf vectors for the cleaned labels in the dataframe.'''
    
    tfidf = TfidfVectorizer(sublinear_tf = True, 
                            min_df = min_df, 
                            norm = 'l2', 
                            encoding = 'latin-1', 
                            ngram_range = (1, 2), 
                            stop_words = 'english')
    
    features = tfidf.fit_transform(df.reviews_cleaned).toarray()
    labels = df.label
    print("Number of features:", features.shape[1]) 
    
    return features, labels, tfidf


# In[153]:


def T_T_split(features, labels):
    X_train, X_test, y_train, y_test = train_test_split(
                                        features, 
                                        labels, 
                                        test_size=0.33, #)
                                        random_state=42)
    return X_train, X_test, y_train, y_test


# In[164]:


def fit_logistic(df, model_name, ngram_range = (1, 2), min_df = 500, fit_bool = True):
    features, labels, tfidf = get_tfidf(df, ngram_range, min_df)
    X_train, X_test, y_train, y_test = T_T_split(features, labels)
    
    model = LogisticRegression(random_state = 0,
                               class_weight = 'balanced')
    if fit_bool:
        model.fit(X_train, y_train)

    CV = 10
    cv_df = pd.DataFrame(index = range(CV))

    accuracy = cross_val_score(model, 
                               X_train, 
                               y_train, 
                               scoring = 'accuracy', 
                               cv = CV)
    roc = cross_val_score(model, 
                               X_train, 
                               y_train, 
                               scoring = 'roc_auc', 
                               cv = CV)
    
    idf = tfidf.idf_
    feat_importances = dict(zip(tfidf.get_feature_names(), idf))
    d = Counter(feat_importances)

    y_pred = model.predict(X_train)
    print("Average accuracy is", round(mean(accuracy), 3))
    print("Average ROC_AUC is", round(mean(roc), 3))
    print('Confusion matrix:')
    print(confusion_matrix(y_train, y_pred)); print(" ")
    
    print("ngram_range:", ngram_range)
    print("min_df:", min_df); print(" ")
    
    print("Top 10 most important features:")
    for k, v in d.most_common(10):
        print('%s: %i' % (k, round(v, 5)))
    
    return model


# In[165]:


logistic1 = fit_logistic(df, 
                         'Logistic1',
                         ngram_range = (1),
                         min_df = 30)


# In[166]:


logistic2 = fit_logistic(df,
                        'Logistic2',
                         ngram_range = (1, 2),
                         min_df = 30)


# In[107]:


logistic3 = fit_logistic(df,
                        'Logistic3',
                         ngram_range = (1, 3),
                         min_df = 30)


# In[108]:


logistic4 = fit_logistic(df,
                        'Logistic4',
                         ngram_range = (1, 4),
                         min_df = 300)


# In[173]:


def fit_svm(df, model_name, min_df = 50, fit_bool = True):
    features, labels, tfidf = get_tfidf(df, ngram_range, min_df)
    X_train, X_test, y_train, y_test = T_T_split(features, labels)
    
    model = svm.SVC(C = 1.0, 
                    kernel = 'linear', 
                    degree = 3, 
                    gamma = 'auto',
                    class_weight = 'balanced')
    if fit_bool:
        model.fit(X_train, y_train)

    CV = 10
    cv_df = pd.DataFrame(index = range(CV))

    accuracy = cross_val_score(model, 
                               X_train, 
                               y_train, 
                               scoring = 'accuracy', 
                               cv = CV)
    roc = cross_val_score(model, 
                               X_train, 
                               y_train, 
                               scoring = 'roc_auc', 
                               cv = CV)
    
    idf = tfidf.idf_
    feat_importances = dict(zip(tfidf.get_feature_names(), idf))
    d = Counter(feat_importances)

    y_pred = model.predict(X_train)
    print("Average accuracy is", round(mean(accuracy), 3))
    print("Average ROC_AUC is", round(mean(roc), 3))
    print('Confusion matrix:')
    print(confusion_matrix(y_train, y_pred)); print(" ")
    
    print("ngram_range:", ngram_range)
    print("min_df:", min_df); print(" ")
    
    print("Top 10 most important features:")
    for k, v in d.most_common(10):
        print('%s: %i' % (k, round(v, 5)))
    
    return model


# In[174]:


fit_svm(df, 'SVM1', min_df = 50) # accuracy is about 0.98 with min_df = 50 and about 0.75 with min_df = 500


# In[ ]:




