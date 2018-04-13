# -*- coding: utf-8 -*-
"""
A generative model trained to write poems using RNN. 

Created on Sun Apr 08 18:14:34 2018

@author: kliu14
"""

import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM

from utils import load_poems

def char_seq(fname):
    # Read the sonnets into a sequence of characters. 

    sonnets = load_poems(fname)
    char_list = []
    for sonnet in sonnets:
        for sentence in sonnet:
            char_list += sentence
            
    return char_list


def char_vocab(char_seq):
    # Build the character-level vocabulary from the file. 
        
    return sorted(list(set(char_seq)))

 
def preprocess_text(fname, seq_length):
    """
    Preprocess the text in file and make it into the format the NN can handle. 
    
    seq_length: int. 
        The length of the sequence to feed into the NN. 
    """
    # Get the sonnets as a sequence of chars. 
    # Note: we need to keep \n or other important chars.     
    
    sonnets_in_char = char_seq(fname)
    print(sonnets_in_char[:30])
    
    chars = char_vocab(sonnets_in_char)
    print(chars[:5])    
    char_to_indices = dict((char,i) for i, char in enumerate(chars))
    indices_to_char = dict((i, char) for i, char in enumerate(chars))
    
    # Make sonnets into input data of specified shape
    # a. Make sequences of chars into the format of X, y
    seqs = [] 
    next_char = [] 
    for i in range(len(sonnets_in_char) - seq_length):
        seqs.append(sonnets_in_char[i:i+seq_length])
        next_char.append(sonnets_in_char[i+seq_length])
    
    # b. Convert the chars into numbers: 
    train_x = np.zeros((len(seqs), seq_length, len(chars)))
    train_y = np.zeros((len(seqs), len(chars)))
    
    for i, sentence in enumerate(seqs):
        for j, single_char in enumerate(sentence):
            train_x[i, j, char_to_indices[single_char]] = 1
        train_y[i, char_to_indices[next_char[i]]] = 1
    
    return train_x, train_y, indices_to_char


def build_rnn(num_unit, seq_length, input_dim):
    model = Sequential()
    model.add(LSTM(num_unit, input_shape=(seq_length, input_dim)))
    model.add(Dense(input_dim))
    model.add(Activation('softmax'))
    
    return model


def check_rnn_parameters():
    # Check the number of parameters in the runn is correct.     
    
    num_unit = 100
    seq_length = 50
    input_dim = 26
    model = build_rnn(num_unit, seq_length, input_dim)    
    
    # We expect: 
    # layer 1: 100 x (100 + 26 + 1) x 4 = 50800
    # layer 2: 100 x 26 (linear weights) + 26 (bias)
    # layer 3: 0
    # Note: layer 1 only output the last y, which is a 100x1 vector. 
    model.summary()


def training_data(): 
    pass


if __name__ == "__main__":
    check_rnn_parameters()

#    fname = r"E:\Coursera\CS155\LiuKe_hw_submission\prj3_code\data\shakespeare.txt"
#    seq_length = 40 
#    preprocess_text(fname, seq_length)