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
from build_models import multiple_model_builder


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
    print(df.head())



