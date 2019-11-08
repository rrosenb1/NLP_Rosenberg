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
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn import model_selection, naive_bayes, svm

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Get on GPU
# Fix tensorflow GPU allocation

#%% GPU memory fix
# import tensorflow as tf
# from tensorflow import keras
# def get_session(gpu_fraction=0.25):    
#     gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction, allow_growth=True)    
#     return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
# keras.backend.set_session(get_session())


def get_tfidf(df, min_df = 500, ngram_range = (1, 2)):
    '''Get tfidf vectors for the cleaned labels in the dataframe.'''
    
    tfidf = TfidfVectorizer(sublinear_tf = True, 
                            min_df = min_df, 
                            max_features = 500,
                            norm = 'l2', 
                            encoding = 'latin-1', 
                            max_df = 0.3,
                            binary = True,
                            ngram_range = ngram_range, 
                            stop_words = 'english')

    features = tfidf.fit_transform(df.reviews_cleaned.tolist())
    labels = df.label
    print("Number of features:", features.shape[1]) 
    
    return features, labels, tfidf

def T_T_split(features, labels):
    X_train, X_test, y_train, y_test = train_test_split(
                                        features, 
                                        labels, 
                                        test_size=0.33)
    
    return X_train, X_test, y_train, y_test

def fit_svm(df, model_name, min_df, ngram_range, kernel = 'linear'):

    print("Running", model_name, "with parameters ngram_range = ", ngram_range, "and min_df = ", min_df)
    print(" ")
    
    if df.shape[0] > 500000:
        df = df.sample(n=500000, replace = True, random_state=1)

    print('Got data loaded - ready for tfidf')
  
    features, labels, tfidf = get_tfidf(df, min_df)
    
    X_train, X_test, y_train, y_test = T_T_split(features, labels)
    
    model = svm.SVC(C = 1.0, 
                    kernel = kernel, 
                    degree = 3, 
                    gamma = 'auto',
                    class_weight = 'balanced')
    
    model.fit(X_train, y_train)

    CV = 5
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

    print(" ")
    
    return model, mean(accuracy), mean(roc)


if __name__ == "__main__":
    df = pd.read_csv("home_and_kitchen_ready_to_model.csv")
    df = df[['label', 'reviews_cleaned']]
    df.dropna(inplace=True)
    print('Pulled dataframe from file.'); print(" ")

    model_names = ['SVM5', 'SVM6', 'SVM7', 'SVM8']
    ngram_ranges = [ (1,1), (1,2), (1,1), (1,2)] 
    kernel_values = ['linear', 'rbf', 'linear', 'rbf']
    min_df_values = [500, 500, 200, 200]
    
    tab = PrettyTable()
    tab.field_names = ["Model Name", "Min_Df", "Ngram_Range", "Penalty", "Accuracy", "ROC"]

    num_models = 8

    for i in range(0, num_models-1):
        model, acc, roc = fit_svm(df, 
                                model_name = model_names[i],
                                min_df = min_df_values[i],
                                ngram_range = ngram_ranges[i],
                                kernel = kernel_values[i])
        
        tab.add_row([model_names[i], min_df_values[i], ngram_ranges[i], kernel_values[i], acc, roc])
    
    print(tab)


'''
SVM Results:
For this experiment I varied:
    -  the number of ngrams (1 or 2), 
    -  the kernel type used (either linear or RBF)
    -  the min_df_values parameter for TF-IDF, which controls the number of features created. 
I used Accuracy and ROC for my reporting metrics because I want to know how the model performs as far as the misclassification rate as well as on an unbalanced dataset. 

I took the following preprocessing steps:
*   Labelling reviews with 3+ stars "positive" (1) and reviews with 1 or 2 stars "negative" (0).
*   Removing all numbers from the text
*   Removing all punctuation from the text
*   Removing all stopwords (English) from the text
*   Stemming all words

All models performed decently well, with accuracies between 0.77 and 0.81 and ROC values between 0.86 and 0.88. The best-performing model used:
    - wordNgrams = 2 (able to use bigrams)
    - Kernel type = RBF (based on mean absolute error)
    - min_df_values = 200
    . This model had an accuracy of 0.813 and an ROC score of 0.866.

This is likely because the model was given more information to work with (both with bigrams and more features) and the decreased min_df. The RBF kernel type was also better here, though the choice of kernel type didn’t make a huge difference either way. The most significant factor was the min_df_values value; since this dataset varies so much and spans so many different products, it is important to be able to create many features from the data.

Overall, the model could likely be improved upon if I used more features, more n-grams, and a lower max_df. I also used some less data to train my SVM models because the large models were getting hung up on both Google Colab and Deepdish; given better compute resources I could have gotten better SVMs.
'''

# %%
