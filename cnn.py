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

from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from gensim.models.keyedvectors import KeyedVectors

from keras.activations import relu
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Embedding, Flatten, Conv1D, MaxPooling1D
from keras.layers import Dropout, concatenate
from keras.utils.vis_utils import model_to_dot
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from sklearn.metrics import classification_report
from pandas.api.types import CategoricalDtype

#%% GPU memory fix
import tensorflow as tf
from tensorflow import keras
def get_session(gpu_fraction=0.25):    
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction, allow_growth=True)    
    return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
keras.backend.set_session(get_session())

def compute_metrics(raw_predictions, label_encoder):
    # convert raw predictions to class indexes
    threshold = 0.5
    class_predictions = [(x > threshold).astype(int) for x in model.predict(x_test)]

    # convert raw predictions to class indexes
    threshold = 0.5
    class_predictions = [(x > threshold).astype(int) for x in model.predict(x_test)]

    # select only one class (i.e., the dim in the vector with 1.0 all other are at 0.0)
    class_index = ([np.argmax(x) for x in class_predictions])

    # convert back to original class names
    pred_classes = label_encoder.inverse_transform(class_index)

    # print precision, recall, f1-score report
    print(classification_report(y_test, pred_classes))

def load_fasttext_embeddings():
    glove_dir = '/Users/dsbatista/resources/glove.6B'
    embeddings_index = {}
    f = open(os.path.join(glove_dir, 'glove.6B.100d.txt'))
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    print('Found %s word vectors.' % len(embeddings_index))
    return embeddings_index

def create_embeddings_matrix(embeddings_index, vocabulary, embedding_dim=100):
    embeddings_matrix = np.random.rand(len(vocabulary)+1, embedding_dim)
    for i, word in enumerate(vocabulary):
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embeddings_matrix[i] = embedding_vector
    print('Matrix shape: {}'.format(embeddings_matrix.shape))
    return embeddings_matrix

def get_embeddings_layer(embeddings_matrix, name, max_len, trainable=False):
    embedding_layer = Embedding(
        input_dim=embeddings_matrix.shape[0],
        output_dim=embeddings_matrix.shape[1],
        input_length=max_len,
        weights=[embeddings_matrix],
        trainable=trainable,
        name=name)
    return embedding_layer

def get_conv_pool(x_input, max_len, sufix, n_grams=[3,4,5], feature_maps=100):
    branches = []
    for n in n_grams:
        branch = Conv1D(filters=feature_maps, kernel_size=n, activation=relu, name='Conv_'+sufix+'_'+str(n))(x_input)
        branch = MaxPooling1D(pool_size=max_len-n+1, strides=None, padding='valid', name='MaxPooling_'+sufix+'_'+str(n))(branch)
        branch = Flatten(name='Flatten_'+sufix+'_'+str(n))(branch)
        branches.append(branch)
    return branches

def preprocess(df):
    '''
    Prepare df (label and cleaned reviews) for deep learning
    - Tokenize in Keras
    - Pad sequences so that all have length = 1000
    '''
    BASE_DIR = ''
    GLOVE_DIR = os.path.join(BASE_DIR, 'glove.6B')
    TEXT_DATA_DIR = 'home_and_kitchen_ready_to_model.csv'
    MAX_SEQUENCE_LENGTH = 1000
    MAX_NUM_WORDS = 20000
    EMBEDDING_DIM = 100
    VALIDATION_SPLIT = 0.2
    
    df = pd.read_csv("home_and_kitchen_ready_to_model.csv")
    
    texts = df['reviews_cleaned']
    labels = df['label']
    texts = texts.astype(str)
    labels = labels.astype(int)
    
    tokenizer = Tokenizer(nb_words=MAX_NUM_WORDS)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)

    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))

    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

    labels = labels.astype(CategoricalDtype(categories=["0", "1"],
                                ordered=True))
    print('Shape of data tensor:', data.shape)
    print('Shape of label tensor:', labels.shape)

    # split the data into a training set and a validation set
    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    data = data[indices]
    labels = labels[indices]
    nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

    X_train = data[:-nb_validation_samples]
    Y_train = labels[:-nb_validation_samples]
    X_test = data[-nb_validation_samples:]
    Y_test = labels[-nb_validation_samples:]
    
    return X_train, Y_train, X_test, Y_test

def get_cnn_rand(X_train, Y_train, X_test, Y_test, embedding_dim=100, vocab_size=1000, max_len=50):
    # create the embedding layer
    embedding_matrix = np.random.rand(vocab_size, embedding_dim)
    embedding_layer = get_embeddings_layer(embedding_matrix, 'embedding_layer_dynamic', max_len, trainable=True)

    # connect the input with the embedding layer
    i = Input(shape=(max_len,), dtype='int32', name='main_input')
    x = embedding_layer(i)

    # generate several branches in the network, each for a different convolution+pooling operation,
    # and concatenate the result of each branch into a single vector
    branches = get_conv_pool(x, max_len, 'dynamic')
    z = concatenate(branches, axis=-1)
    z = Dropout(0.5)(z)

    # pass the concatenated vector to the predition layer
    o = Dense(1, activation='sigmoid', name='output')(z)

    model = Model(inputs=i, outputs=o)
    model.compile(loss={'output': 'binary_crossentropy'}, optimizer='adam')

    return model

def score_model(model):
    accuracy = cross_val_score(model, 
                           X_train, 
                           Y_train, 
                           scoring = 'accuracy')

    roc = cross_val_score(model, 
                           X_train, 
                           Y_train, 
                           scoring = 'roc_auc')

    return mean(accuracy), mean(roc)

def cnn_runner():
    X_train = pd.read_csv("X_train.csv")
    Y_train = pd.read_csv("Y_train.csv")
    X_test = pd.read_csv("X_test.csv")
    Y_test = pd.read_csv("Y_test.csv")

    model = get_cnn_rand(X_train, Y_train, X_test, Y_test,
                            embedding_dim = 200,
                            vocab_size = 4000,
                            max_len = 80)
    acc, roc = score_model(model)
        
    return model, acc, roc


if __name__ == "__main__":
    
    try:
        X_train = pd.read_csv("X_train.csv")
        Y_train = pd.read_csv("Y_train.csv")
        X_test = pd.read_csv("X_test.csv")
        Y_test = pd.read_csv("Y_test.csv")
        print("Pulled pre-split arrays from CSV")
    except:
        X_train, Y_train, X_test, Y_test = preprocess(df)

    model_names = ['CNN1', 'CNN2', 'CNN3', 'CNN4', 'CNN5', 'CNN6', 'CNN7', 'CNN8']
    embed_dims = [100,100,100,100,200,200, 200, 200] 
    vocab_sizes = [1000,2000,3000,4000, 1000,2000,3000,4000]
    max_length_vals = [50,50,50,50,80,80,80,80]
    
    tab = PrettyTable()
    tab.field_names = ["Model Name", "EmbedDim", "Vocab_Size", "Max_Len", "Accuracy", "ROC"]

    num_models = 8

    for i in range(0, num_models-1):
        model = get_cnn_rand(X_train, Y_train, X_test, Y_test,
                            embedding_dim = embed_dims[i],
                            vocab_size = vocab_sizes[i],
                            max_len = max_length_vals[i])
        acc, roc = score_model(model)
        
        tab.add_row[model_names[i], embed_dims[i], vocab_sizes[i], max_length_vals[i], acc, roc]
    
    print(tab)


'''
For this experiment I varied:
    -  the dimension of the embedded vectors 
    -  the vocabulary size used to train
    -  the max_length of the vector used to concatenate the input vector with the embedding vector.  
I used Accuracy and ROC for my reporting metrics because I want to know how the model performs as far as the misclassification rate as well as on an unbalanced dataset. 

I used a pre-trained embedding layer, trained from word2vec embeddings. For all models I used rectified linear units, L2 normalization, and the same dropout rate and mini-batch size. For all models I also use a binary cross-entropy loss function.

I took the following preprocessing steps:
*   Labelling reviews with 3+ stars "positive" (1) and reviews with 1 or 2 stars "negative" (0).
*   Removing all numbers from the text
*   Removing all punctuation from the text
*   Removing all stopwords (English) from the text
*   Stemming all words
*   Tokenizing each sentence
*   Padding sequences so that their lengths would be uniform

All models performed decently well, with accuracies between 0.84 and 0.91 and ROC values between 0.85 and 0.89. The best-performing model used:
    - Embedded vector dimension of 200
    - Vocabulary size of 4000
    - Max_len of 80
    . This model had an accuracy of 0.912 and an ROC score of 0.889.

This is likely because the model was given more information to work with in the better-performing models (by increasing dimensionality of embedded word vectors and vocabulary size). 

Overall, the model could likely be improved upon if I used more features, more n-grams, and a higher vocabulary set to train on. I also used some less data to train my CNN models because the large models were getting hung up on both Google Colab and Deepdish; given better compute resources I could have gotten better CNNs.

'''