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
from nltk.stem import WordNetLemmatizer 
from string import digits
from statistics import mean 
from nltk.corpus import stopwords
from collections import Counter
from prettytable import PrettyTable

from intake import retrieve_data

# nltk.download('wordnet')


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
    
def lemmatize_words(textdata):
    words = []
    # porter = PorterStemmer()
  
    lemmatizer = WordNetLemmatizer() 
    
    for l in textdata:
        words.append([lemmatizer.lemmatize(word) for word in l])

    return words

def clean(df):

    df = create_label(df)

    df = df[['Posts', 'Label']].dropna()
    textdata = df['Posts']

    print("Made it to data cleaning")
    textdata = strip(textdata); print("stripped")
    textdata = to_lower(textdata); print("all lowercase")
    textdata = rm_numbers(textdata); print('removed numbers')
    textdata = lemmatize_words(textdata); print('lemmatized words') # chose lemmatizing bc it is less aggressive + makes more sense

    df['text_cleaned'] = textdata
    df['text_cleaned'] = df.text_cleaned.apply(' '.join)
    df = df[['Label', 'text_cleaned']]

    print("Finished cleaning data. Saving to file.")
    df.to_csv('df_ready_to_model_5000.csv')

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
        df = pd.read_csv("df_5000.csv")
    except:
        df = intake.retrieve_data(5000)

    df_cleaned = clean(df)
    print(df_cleaned.head())
