# Imports
from __future__ import print_function

import os
# Mac workaround
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import sys, time, random, gc, socket
import matplotlib.pyplot as plt
from collections import Counter

import fasttext
import os
import gensim
import string
import nltk
import itertools
import json
import gzip
import pandas as pd
import numpy as np

from nltk.stem.porter import PorterStemmer
from string import digits
from statistics import mean 
from nltk.corpus import stopwords
from collections import Counter
from prettytable import PrettyTable

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn import model_selection, naive_bayes, svm

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

def parse_labels(path):
    with open(path, 'r') as f:
        return np.array(list(map(lambda x: x[9:], f.read().split())))
        
def create_preds(model, model_name):
    test_file = 'testing_data_fasttext.txt'
    preds = []

    with open(test_file, "r") as inputf:
        for line in inputf.readlines():
            test_doc = line.split("__label__")[0]  # my per line format is "text __label__$label$"
            preds.append(model.predict(test_doc))
                
    predsPath = str('testing_predictions_' + model_name + ".txt")

    f = open(predsPath,"w") 

    for i in range(0, len(preds)):
        f.write(str(preds[i][0][0]))
        f.write("\n")

    f.close()

def model_accuracy(model, model_name):

    create_preds(model, model_name)
        
    test = 'testing_labels_only.txt'
    predict = str('testing_predictions_' + model_name + ".txt")
        
    test_labels = parse_labels(test)
    pred_labels = parse_labels(predict)
        
    eq = test_labels == pred_labels
        
    acc = round(eq.sum() / len(test_labels), 5)
        
    print("Accuracy: " + str(round(eq.sum() / len(test_labels), 5)))
    print(confusion_matrix(test_labels, pred_labels))
        
    return acc

def evaluate_model(model_name, epoch = 5, wordNgrams = 2, lr = .1):
    print(' '); print('Training model', model_name, '.')
        
    model = fasttext.train_supervised('training_data_fasttext.txt',
                                    epoch = epoch,
                                    wordNgrams = wordNgrams,
                                    lr = lr
                                    )

    print("Nearest neighbors:")
    print(model.get_nearest_neighbors("kitchen")); print(" ")
    print("Get_analogies:")
    print(model.get_analogies('pattern', 'decoration', 'citrus')); print(" ")
        
    acc = model_accuracy(model, model_name)
        
    return model, acc

def fit_ft(df, model_name, epoch, wordNgrams, lr, minCount):
    # Split into test and training datasets
    features = df['reviews_cleaned']
    labels = df['label']

    X_train, X_test, y_train, y_test = train_test_split(
                                            features, 
                                            labels, 
                                            test_size=0.33,
                                            random_state=42)

    X_train = pd.DataFrame(X_train, columns = ['reviews_cleaned']).reset_index()
    X_test = pd.DataFrame(X_test, columns = ['reviews_cleaned']).reset_index()
    y_train = pd.DataFrame(y_train, columns = ['label']).reset_index()
    y_test = pd.DataFrame(y_test, columns = ['label']).reset_index()

    # write data in a file. 
    f = open("training_data_fasttext.txt","w") 
    
    for i in range(0, len(X_train)):
        f.write(str(X_train['reviews_cleaned'][i]) + ' ')
        f.write('__label__' + str(y_train['label'][i]))
        f.write("\n")
        
    f.close()

    f = open("testing_data_fasttext.txt","w") 
    
    for i in range(0, len(X_test)):
        f.write(X_test['reviews_cleaned'][i] + ' ')
        f.write('__label__' + str(y_test['label'][i]))
        f.write("\n")
        
    f.close()

    f = open("testing_labels_only.txt","w") 
    
    for i in range(0, len(X_test)):
        f.write('__label__' + str(y_test['label'][i]))
        f.write("\n")
        
    f.close()

    model, acc = evaluate_model(model_name, epoch = 25, wordNgrams = wordNgrams, lr = lr)

    return model, acc


if __name__ == "__main__":
    df = pd.read_csv("home_and_kitchen_ready_to_model.csv")
    df = df[['label', 'reviews_cleaned']]
    df.dropna(inplace=True)
    print('Pulled dataframe from file.'); print(" ")

    model_names = ['FT1', 'FT2', 'FT3', 'FT4', 'FT5', 'FT6', 'FT7', 'FT8']
    wordNgramses = [1, 2, 1, 2, 1, 2, 1, 2] 
    lrs = [.3, .3, .5, .5, .7, .7, .9, .9]
    minCounts = [10,10,10,20,20,20,20]
    
    tab = PrettyTable()
    tab.field_names = ["Model Name", "wordNgrams", "Learning_Rate", "min_df_values", "Accuracy"]

    num_models = 8

    for i in range(0, num_models-1):

        model, acc = fit_ft(df, 
                        model_name = model_names[i],
                        epoch = 25,
                        wordNgrams = wordNgramses[i],
                        minCount = minCounts[i],
                        lr = lrs[i]
                        )
        
        tab.add_row([model_names[i], wordNgramses[i], lrs[i], minCounts[i], acc])
    
    print(tab)

'''
FastText Results:

For this experiment I varied:
    -  the number of ngrams (1 or 2), 
    -  the learning rate of the model (0.3, 0.5, 0.7, or 0.9), and 
    -  the min_df_values parameter for TF-IDF, which controls the number of features created. 
I used Accuracy for my reporting metric because I want to know how the model performs as far as the misclassification rate. 
ROC Score and F1 were not key here because the dataset was balanced, so the model would not be biased toward either class.

I took the following preprocessing steps:
*   Labelling reviews with 3+ stars "positive" (1) and reviews with 1 or 2 stars "negative" (0).
*   Removing all numbers from the text
*   Removing all punctuation from the text
*   Removing all stopwords (English) from the text
*   Stemming all words

I used the following parameters identically for all models:

*   Number of epochs: 25 - this sped up processing some and allowed for rapid iteration, which I valued greatly.
*   Used the same dataset and sample size for each model - 

All models performed decently well, with accuracies in the 90%s. The best-performing model on both metrics used:
    - Bigrams (wordNgrams = 2)
    - Learning_Rate = 0.3
    - min_df_values = 10
    . This model had an accuracy of 0.940.

This is likely because the model was given more information to work with (both with bigrams and more features) and the decreased learning rate.
Learning rate determines how much the model changes after each iteration; a low learning rate is more precise, but is slower. In this case, the low learning rate avoided large swings in learning direction and resulted in a more accurate model. 
Overall, the model could likely be improved upon if I used more features, more n-grams, and a lower max_df, but the accuracy is already very high and the amount of increased performance possible requires further experimentation.'''

