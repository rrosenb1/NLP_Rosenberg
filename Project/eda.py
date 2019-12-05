import os
import gensim
import string
import nltk
import itertools
import logging
import logging.config
import argparse

import pandas as pd
import numpy as np
import random as rand

from nltk.stem.porter import PorterStemmer
from string import digits
from logging import Logger

from intake import retrieve_data
from clean import clean


if __name__ == '__main__':

    # Intake or Create Cleaned Data
    try:
        print("Pulling dataframe from working directory.")
        df = pd.read_csv("Project/df_ready_to_model.csv")
    except:
        print("Running intake script.")
        df = intake.retrieve_data(1000)
        df = clean.clean(df)

    print(df.shape)
    print(df.head()); print(" ")

    # Discuss class imbalance
    print("Class distributions below:")
    for i in range(0, 6):
        print("Label =", i, ":", df[df['Label'] == i].shape[0])

    print('There is some class imbalance; this will be addressed in the models.'); print(" ")
    
    # Find average word length of documents
    length = []
    for row in df['text_cleaned']:
        length.append(len(row.split()))
    avg_words = round(sum(length)/len(length), 3)
    print("There are", avg_words, "average words in each document in the corpus."); print(" ")

    # Find number of documents
    print("There are", df.shape[0], "documents in the corpus.")