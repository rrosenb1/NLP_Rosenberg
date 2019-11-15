import os
import gensim
import string
import nltk
import itertools
import random
import glob

import pandas as pd
import numpy as np

from nltk.stem.porter import PorterStemmer
from string import digits


def retrieve_data(end_len):
    path = '/Users/rachelrosenberg/MSiA/490 - Text Analytics/blogs/*.xml'
    textdata = []
    userid = []; age = []; gender = []; category = []; star_sign = []
    files_parsed = 0

    for filename in glob.glob(str(path))[:end_len]:
        try:
            with open(str(filename), 'r') as f:
                attributes = filename.split('.')
                if len(attributes) == 6:
                    text = f.read()
                    if text:
                        textdata.append(text)
                        gender.append(attributes[1])
                        age.append(attributes[2])
                        category.append(attributes[3])
                        star_sign.append(attributes[4])

                    files_parsed += 1
        except:
            pass
    print(len(gender))
    print(len(age))
    print(len(category))
    print(len(star_sign))
    print(len(textdata))

    df = pd.DataFrame({"Text": textdata, 
                       "Gender": gender,
                       "Age": age,
                       "Category": category,
                       "StarSign": star_sign
                       }).reset_index()

    print(files_parsed, "files parsed.")
    print("Returning first", end_len, "files.")
    
    return(df[:end_len])



if __name__ == '__main__':
    df = retrieve_data(10)
    print("Retrieved", len(df), "documents from folder.")
    print("Saving to working directory as df.csv.")
    print(df.head(10))

    df.to_csv("df.csv")