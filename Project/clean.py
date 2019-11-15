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

from intake import retrieve_data


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

def clean(df):

    df = df[['Text', 'Label']]
    df['Text'] = df['Text'].fillna("")
    # df['label'] = [1 if x >= 3 else 0 for x in df['overall']] # group labels

    # Clean & normalize dataset
    # normalization (e.g. convert to lowercase, remove non-alphanumeric chars, numbers,
    textdata = df['Text']

    print("Made it to data cleaning")
    textdata = strip(textdata); print("stripped")
    textdata = to_lower(textdata); print("all lowercase")
    textdata = rm_numbers(textdata); print('removed numbers')
    textdata = stem_words(textdata); print('stemmed words')

    df['text_cleaned'] = textdata
    df['text_cleaned'] = df.text_cleaned.apply(' '.join)
    df = df[['Label', 'text_cleaned']]

    print("Finished cleaning data. Saving to file.")
    df.to_csv('df_ready_to_model.csv')

    return df

if __name__ == '__main__':
    try:
        df = pd.read_csv("Project/df.csv")
    except:
        df = intake.retrieve_data(1000)

    df_cleaned = clean(df)
    print(df.head())
