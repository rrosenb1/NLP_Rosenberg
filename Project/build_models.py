import os
import gensim
import string
import nltk
import itertools
import json
import gzip
# import yaml

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

from intake import retrieve_data
from clean import clean

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

def get_tfidf(df, min_df, ngram_range, penalty):
    '''Get tfidf vectors for the cleaned labels in the dataframe.'''
    
    tfidf = TfidfVectorizer(sublinear_tf = True, 
                            min_df = min_df, 
                            max_features = min_df, # set these equal to one another
                            norm = penalty, 
                            encoding = 'latin-1', 
                            max_df = 0.3,
                            binary = False,
                            ngram_range = ngram_range, 
                            stop_words = 'english')

    features = tfidf.fit_transform(df.text_cleaned.tolist())
    labels = df.Label
    print("Number of features:", features.shape[1]) 
    
    return features, labels, tfidf

def T_T_split(features, labels):
    X_train, X_test, y_train, y_test = train_test_split(
                                        features, 
                                        labels, 
                                        test_size=0.33 #random_state = 43
                                        )
    
    return X_train, X_test, y_train, y_test

def preprocess(df, min_df, ngram_range, penalty):
    print("Preprocessing with parameters ngram_range = ", ngram_range, "and min_df = ", min_df)
    print(" ")
  
    features, labels, tfidf = get_tfidf(df, min_df, ngram_range, penalty)

    X_train, X_test, y_train, y_test = T_T_split(features, labels)

    return X_train, X_test, y_train, y_test

def comparison_table(model_names, accuracies, f1s):
    tab = PrettyTable()
    tab.field_names = ["Model Name", "Accuracy", "F1 Score"]

    for i in range(0, len(model_names)):
        tab.add_row([model_names[i], round(accuracies[i], 3), round(f1s[i], 3)])

    print(tab)

    return    

def multiple_model_runner(X_train, X_test, y_train, y_test):

    # Append classifier to preprocessing pipeline for three kinds of models
    # Leave all with default parameters for now - will fit hyperparameters for the best model after
    log_reg = Pipeline(steps=[('classifier', LogisticRegression(class_weight = 'balanced'))])

    clf = Pipeline(steps=[('classifier', DecisionTreeClassifier(class_weight = 'balanced'))])

    rd = Pipeline(steps=[('classifier', RandomForestClassifier(class_weight = 'balanced'))])
    
    # Test which model generally performs the best with default parameters
    model_list = [log_reg, clf, rd]
    model_names = ['Logistic Regression', 'Decision Tree Classifier', 'Random Forest Classifier']
    accuracies = []; f1s = []

    for i in range(0, len(model_list)):
        model = model_list[i]
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        
        accuracies.append(model.score(X_test, y_test))
        f1s.append(f1_score(y_test, preds, average = 'macro'))
        
    return model_names, accuracies, f1s

if __name__ == '__main__':

    # # Configure YAML
    # yml_path = "Project/config.yml"

    # # Load args from YAML file
    # with open(yml_path, "r") as f:
    #         config = yaml.load(f)
    # config_build_models = config["build_models"]

    # create_model(**config_build_models)
   
    # Intake or Create Cleaned Data
    try:
        print("Pulling dataframe from working directory.")
        df = pd.read_csv("Project/df_ready_to_model.csv")
    except:
        print("Running intake script.")
        df = intake.retrieve_data(1000)
        df = clean.clean(df)

    X_train, X_test, y_train, y_test = preprocess(df, min_df = 50, ngram_range= (1, 3), penalty = "l1") # **config_build_models
    model_names, accuracies, f1s = multiple_model_runner(X_train, X_test, y_train, y_test)
    comparison_table(model_names, accuracies, f1s)

