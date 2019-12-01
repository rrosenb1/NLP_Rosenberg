import os
import gensim
import string
import nltk
import itertools
import json
import gzip
import yaml

import pandas as pd
import numpy as np

from nltk.stem.porter import PorterStemmer
from string import digits
from statistics import mean 
from nltk.corpus import stopwords
from collections import Counter
from prettytable import PrettyTable

from sklearn import model_selection, naive_bayes, svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_curve, auc, f1_score
from sklearn.metrics import classification_report 
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import ParameterGrid
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

from intake import retrieve_data
from clean import clean

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

def get_tfidf(df, min_df, ngram_range, penalty):
    '''Get tfidf vectors for the cleaned labels in the dataframe.'''
    
    tfidf = TfidfVectorizer(min_df = min_df, 
                            norm = penalty, 
                            encoding = 'latin-1', 
                            max_df = 0.8,
                            binary = False,
                            ngram_range = ngram_range, 
                            stop_words = 'english')

    features = tfidf.fit_transform(df.text_cleaned.astype('U').tolist())
    labels = df.Label
    print("Number of features:", features.shape[1]); print(" ")
    
    return features, labels, tfidf

def T_T_split(features, labels):
    X_train, X_test, y_train, y_test = train_test_split(
                                        features, 
                                        labels, 
                                        test_size=0.33
                                        )
    
    return X_train, X_test, y_train, y_test

def preprocess(df, min_df, ngram_range, penalty):
    print("Preprocessing with parameters ngram_range = ", ngram_range, ", min_df = ", min_df, ', penalty = ', penalty)
  
    features, labels, tfidf = get_tfidf(df, min_df, ngram_range, penalty)

    X_train, X_test, y_train, y_test = T_T_split(features, labels)

    return X_train, X_test, y_train, y_test

def comparison_table(model_names, accuracies, f1s):
    tab = PrettyTable()
    tab.field_names = ["Model", "Accuracy", "F1 Score"]

    for i in range(0, len(model_names)):
        tab.add_row([model_names[i], round(accuracies[i], 3), round(f1s[i], 3)])

    print(tab)

    return    

def multiple_model_runner(df):

    X_train, X_test, y_train, y_test = preprocess(df, min_df = 100, ngram_range= (1, 1), penalty = "l2") # **config_build_models

    # Append classifier to preprocessing pipeline for three kinds of models
    # Leave all with default parameters for now - will fit hyperparameters for the best model after
    log_reg = Pipeline(steps=[('classifier', LogisticRegression(class_weight = 'balanced'))])
    clf = Pipeline(steps=[('classifier', DecisionTreeClassifier(class_weight = 'balanced'))])
    rd = Pipeline(steps=[('classifier', RandomForestClassifier(class_weight = 'balanced'))])
    svm_class = Pipeline(steps=[('classifier', svm.SVC(class_weight = 'balanced'))])
    knn = Pipeline(steps=[('classifier', KNeighborsClassifier(n_neighbors=4))])
    NN = Pipeline(steps=[('classifier', MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1))])
    
    # Test which model generally performs the best with default parameters
    model_list = [log_reg, clf, rd, svm_class, knn, NN]
    model_names = ['Logistic Regression', 'Decision Tree Classifier', 'Random Forest Classifier', 'SVM', 'KNN', 'MLPC']
    accuracies = []; f1s = []

    for i in range(0, len(model_list)):
        model = model_list[i]
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        
        accuracies.append(model.score(X_test, y_test))
        f1s.append(f1_score(y_test, preds, average = 'macro'))
        print(model_names[i]); print(confusion_matrix(y_test, preds))

    comparison_table(model_names, accuracies, f1s)
        
    return 

def fine_tune(df):

    parameters = {
        'min_df': [25, 50, 75, 100],
        'ngram_range': [(1,1), (1,2), (1,3)],
        'penalty': ['l1', 'l2']
    }
    parameters = list(ParameterGrid(parameters))

    model_names, accuracies, f1s = [], [], []

    for i in range(0, 5): #len(parameters)):
        X_train, X_test, y_train, y_test = preprocess(df, 
                                                  min_df = parameters[i]['min_df'], 
                                                  ngram_range= parameters[i]['ngram_range'], 
                                                  penalty = parameters[i]['penalty'])

        model = Pipeline(steps=[('classifier', LogisticRegression(class_weight = 'balanced'))])
        model.fit(X_train, y_train)
    
        preds = model.predict(X_test)

        model_names.append(parameters[i])
        accuracies.append(model.score(X_test, y_test))
        f1s.append(f1_score(y_test, preds, average = 'macro'))

    comparison_table(model_names, accuracies, f1s)
    '''
    Best model is min_df = 25, ngram_range = (1,2), penalty  = 'l2' with acc = 0.585 and f1 = 0.571
    '''
    X_train, X_test, y_train, y_test = preprocess(df, 
                                                  min_df = 25, 
                                                  ngram_range= (1,2), 
                                                  penalty = 'l2')

    parameters = {'max_iter' : [100, 200, 500], 'C' : [0.5, 1, 5], 'multi_class': ['ovr', 'auto']}
    model = LogisticRegression(class_weight = 'balanced')
    clf = GridSearchCV(model, parameters, cv=5)
    clf.fit(X_train, y_train)
    print('Best parameters are:', clf.best_params_)

    preds = clf.predict(X_test)
    print('Accuracy:', round(clf.score(X_test, y_test), 5))
    print('F1 Score:', round(f1_score(y_test, preds, average = 'macro'), 5))
    print("Confusion Matrix:"); print(confusion_matrix(y_test, preds))

    return

if __name__ == '__main__':
   
    # Intake or Create Cleaned Data
    try:
        print("Pulling dataframe from working directory.")
        df = pd.read_csv("df_ready_to_model_5000.csv")
    except:
        print("Running intake script.")
        df = retrieve_data(1000)
        df = clean.clean(df)

    multiple_model_runner(df)
    # fine_tune(df)
