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

    df = df[['Posts', 'Label']].dropna()
    textdata = df['Posts']

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

def create_label(df):

    def f(row):
        if row['StarSign'] == 'Aries':
            val = 'Fire'
        elif row['StarSign'] == 'Sagittarius':
            val = 'Fire'
        elif row['StarSign'] == 'Leo':
            val = 'Fire'
        elif row['StarSign'] == 'Taurus':# | row['StarSign'] == 'Virgo' | row['StarSign'] == 'Capricorn'):
            val = 'Earth'
        elif row['StarSign'] == 'Virgo':
            val = 'Earth'
        elif row['StarSign'] == 'Capricorn':
            val = 'Earth'
        elif row['StarSign'] == 'Gemini': #| row['StarSign'] == 'Libra' | row['StarSign'] == 'Aquarius'):
            val = 'Air'
        elif row['StarSign'] == 'Libra':
            val = 'Air'
        elif row['StarSign'] == 'Aquarius':
            val = 'Air'
        elif row['StarSign'] == 'Cancer':# | row['StarSign'] == 'Scorpio' | row['StarSign'] == 'Pisces'):
            val = 'Water'
        elif row['StarSign'] == 'Scorpio':
            val = 'Water'
        elif row['StarSign'] == 'Pisces':
            val = 'Water'
        return val

    df['Label'] = df.apply(f, axis=1)

    return df

if __name__ == '__main__':
    try:
        df = pd.read_csv("df.csv")
    except:
        df = intake.retrieve_data(1000)

    df_labelled = create_label(df)
    df_cleaned = clean(df_labelled)
    print(df_cleaned.head())
