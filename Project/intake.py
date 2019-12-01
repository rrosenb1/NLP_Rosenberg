import os
import gensim
import string
import nltk
import itertools
import random
import glob
import re

import pandas as pd
import numpy as np

from nltk.stem.porter import PorterStemmer
from string import digits


def retrieve_data(end_len):
    path = '/Users/rachelrosenberg/MSiA/490 - Text Analytics/blogs/*.xml'
    textdata = []
    userid = []; age = []; gender = []; category = []; star_sign = []
    files_parsed = 0

    for filename in glob.glob(str(path))[:end_len*5]:
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

    df = pd.DataFrame({"Text": textdata, 
                       "Gender": gender,
                       "Age": age,
                       "Category": category,
                       "StarSign": star_sign
                       }).reset_index()
    
    df = df.sample(n = end_len).reset_index() # randomly sample N rows from dataframe

    return df

def split_text(text, i):

    s = "<date>.*</date>\n<post>"
    replaced = re.sub(s, '###', text[i])
    replaced = re.sub(r"\n", '', replaced)
    replaced = re.sub(r"\t", '', replaced)
    replaced = re.sub(r"</post>", '', replaced)
    replaced = re.sub(r"<Blog>", '', replaced)
    replaced = re.sub(r"</Blog>", '', replaced)

    user_text = replaced.split("###")
    
    return user_text

def create_user_df(line, i):
    
    user_text = split_text(line['Text'].astype('str'), i)

    # Create dataframe of repeated rows for the user
    user_df = pd.DataFrame(np.repeat(line.values, len(user_text), axis = 0), columns = df.columns)
    user_df['Posts'] = user_text
    user_df = user_df[['Gender', 'Age', 'Category', 'StarSign', 'Posts']]
        
    return user_df

def create_long_df(df):

    df_long = create_user_df(df.iloc[[0]], 0)

    for i in range(1, len(df)):
        line = df.iloc[[i]]
        user_df = create_user_df(line, i)
        user_df.head()
        
        df_long = df_long.append(user_df)
        
    df_long = df_long.reset_index()
    
    return df_long


if __name__ == '__main__':
    df = retrieve_data(5000)
    print("Retrieved", len(df), "documents from folder.")
    print("Saving to working directory.")

    df_long = create_long_df(df)
    print(df_long.shape)

    df_long.to_csv("df_5000.csv")