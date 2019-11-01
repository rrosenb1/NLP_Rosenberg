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

def new_df(path):
    print('Pulling and cleaning dataframe from', path)
    df = getDF(path)

    # Cut size of df down if necessary
    if len(df) > 500000:
        df = df[:500000]

    df = df[['overall', 'reviewText']]
    df['reviewText'] = df['reviewText'].fillna("")
    df['label'] = [1 if x >= 3 else 0 for x in df['overall']]

    # Clean & normalize dataset
    # normalization (e.g. convert to lowercase, remove non-alphanumeric chars, numbers,
    textdata = df['reviewText']

    print("Made it to data cleaning")
    textdata = strip(textdata); print("stripped")
    textdata = to_lower(textdata); print("all lowercase")
    textdata = rm_numbers(textdata); print('removed numbers')
    textdata = stem_words(textdata); print('stemmed words')

    df['reviews_cleaned'] = textdata
    df['reviews_cleaned'] = df.reviews_cleaned.apply(' '.join)
    df = df[['label', 'reviews_cleaned']]

    print("Finished cleaning data. Saving to file.")
    df.to_csv('home_and_kitchen_ready_to_model.csv')

    return df

if __name__ == '__main__':
    try:
        df = pd.read_csv('home_and_kitchen_ready_to_model.csv')
        df = df[['label', 'reviews_cleaned']]
        df.dropna(inplace=True)
        print('Pulled dataframe from file.'); print(" ")
    except:
        print('Pulling and cleaning dataset from scratch'); print(" ")
        df = new_df('reviews_Home_and_Kitchen_5.json.gz')

    print(" "); print("Dataset description: ")
    print("This dataset is a large corpus of Amazon reviews for kitchen & home products. It comes from the Computer Science department at UCSD.")
    print("It contains", df.shape[0], "unique reviews, among a subset of only users who have more than 5 reviews on the site overall (to clean the data).")
    print("The original data is labeled with a number of stars, 1-5. When cleaning, I assign \"good\" reviews as those with 3+ stars and \'bad\' reviews as those with one or two stars.")
    print(" ")

    # Discuss class imbalance
    print("There are two classes in the cleaned dataset: good review (1) and bad review (0).")
    print("Class distributions below:")
    print('Label = 0: ', df[df['label'] == 1].shape) # these are good reviews
    print('Label = 1: ', df[df['label'] == 0].shape) # these are bad reviews
    print('There is some class imbalance; this will be addressed in the models.'); print(" ")
    
    # Find average word length of documents
    length = []
    for row in df['reviews_cleaned']:
        length.append(len(row.split()))
    avg_words = round(sum(length)/len(length), 3)
    print("There are", avg_words, "average words in each document in the corpus.")