# -*- coding: utf-8 -*-
"""
A generative model trained to write poems using RNN.
Reference: https://github.com/keras-team/keras/blob/master/examples/lstm_text_generation.py.  

Created on Sun Apr 08 18:14:34 2018

@author: kliu14
"""
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.callbacks import LambdaCallback
from keras.optimizers import RMSprop

import sys

from utils import load_poems, load_poems_char_level


#def char_seq(fname):
#    # Read the sonnets into a sequence of characters. 
#
#    sonnets = load_poems(fname)
#    char_list = []
#    for sonnet in sonnets:
#        for sentence in sonnet:
#            char_list += sentence
#            
#    return char_list

#def char_vocab(char_seq):
#    # Build the character-level vocabulary from the file. 
#        
#    return sorted(list(set(char_seq)))

def char_vocab(char_seq):
    """
    [Inputs]:
    char_seq: a list of string. 
        Each string is a sonnet with \n.
        
    [Outputs]:
    vocab: a sorted list of all the characters. 
    """
    vocab = set() 
    
    for sonnet in char_seq:
        sonnet_in_char = list(sonnet)
        vocab = vocab.union(sonnet_in_char)
       
    return sorted(list(vocab))
    
    
def seq_with_next_char(str_list, seq_length):
    """
    Make the string list into (sequence, next_char) pair for training. 
    
    [INPUTS]:
    str_list: a list.
        Each element is a string (a sonnet for the poems) from the data file.
    seq_length: integer. 
        The length of the resultant sequence. 
    [OUTPUTS]:
    seqs: a list.
        Each element is string with the fixed length "seq_length". 
    next_char: a list. 
        The next char for the corresponding string. 
    """
    seqs = []
    next_char = []
    
    for sonnet in str_list:
        for i in range(len(sonnet)-seq_length):
            seqs.append(sonnet[i : i + seq_length])
            next_char.append(sonnet[i + seq_length])
            
    return seqs, next_char

 
def preprocess_text(fname, seq_length):
    """
    Preprocess the text in file and make it into the format the NN can handle. 
    
    seq_length: int. 
        The length of the sequence to feed into the NN. 
    """
    # Get the sonnets as a sequence of chars. 
    # Note: we need to keep \n or other important chars.     
    sonnets_in_char = load_poems_char_level(fname)
    
    
    chars = char_vocab(sonnets_in_char)
    print(chars[:5])    
    char_to_indices = dict((char,i) for i, char in enumerate(chars))
    indices_to_char = dict((i, char) for i, char in enumerate(chars))
    
    # Make sonnets into input data of specified shape
    # a. Make sequences of chars into the format of X, y
    seqs, next_char = seq_with_next_char(sonnets_in_char, seq_length)
    
    # b. Convert the chars into numbers: 
    train_x = np.zeros((len(seqs), seq_length, len(chars)))
    train_y = np.zeros((len(seqs), len(chars)))
    
    # One-hot encodding: 
    for i, sentence in enumerate(seqs):
        for j, single_char in enumerate(sentence):
            train_x[i, j, char_to_indices[single_char]] = 1
        train_y[i, char_to_indices[next_char[i]]] = 1
    
    return train_x, train_y, indices_to_char
  
    
    
def sampling(preds, temperature):
    # Sample a character according to the distribution predicted by the softmax function
    # in RNN 
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    
    return np.argmax(probas)


def on_epoch_end(epoch, logs):
    # Collect things to output at the end of an epoch. 
    
    print("")
    print('----- Generating text after Epoch: %d' % epoch)
    
    temperature = 1.0
    # Here we start from a random position between 0 and 100.
    # start_index = np.random.randint(0, 100)
    start_index = 0
    num_words_to_generate = 400
    
    # Get a sequence to start with:
    generated = ''
    sentence = ''.join(sonnets_in_char[start_index:start_index+seq_length])
    generated += sentence
    
    # Make a prediction based on the trained model after this epoch: 
    for i in range(num_words_to_generate):
        # Make a vector for the inputs: 
        x_pred = np.zeros((1, seq_length, len(chars)))
        for i in range(seq_length):
            x_pred[0, i, char_to_indices[sentence[i]]] = 1
        
        # Make a prediction using the trained model: 
        prds = model.predict(x_pred, verbose=0)[0]
        prd_index = sampling(prds, temperature)
        # Convert the model prediction into a character
        next_char = indices_to_char[prd_index]
        
        generated += next_char
        # Merge the prediction into the inputs for the next prediction: 
        sentence = sentence[1:] + next_char
        
        sys.stdout.write(next_char)
        sys.stdout.flush()

    print()


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
    
    
def test_char_vocab():
    char_seq = ['hello, this is me', 'who is that?']
    
    print("")
    print("The list of string:")
    print(char_seq)
    
    vocab = char_vocab(char_seq)
    print("The vocabulary:")
    print(vocab)
    
    
def test_seq_with_next_char():
    str_list = ["Hello, this is Jim Green.", "What's up, body?"]
    print("")
    print("The string list is:", str_list)
    seq_length = 4
    
    seqs, next_char = seq_with_next_char(str_list, seq_length)
    
    print("The 1st sequence:", seqs[0])
    print("The next char:", next_char[0])
    
    
def test_preprocess():
    fname = r"E:\Coursera\CS155\LiuKe_hw_submission\prj3_code\data\shakespeare.txt"
    train_x, train_y, _ = preprocess_text(fname, 7)
    
    print("The first training sequence: ", train_x[0, :, :])
    print("The first training label:", train_y[0, :])
    
    return None  


if __name__ == "__main__":
    # check_rnn_parameters()
    # test_preprocess()
    # test_char_vocab()
    # test_seq_with_next_char()


    # ======================================================
    # Function 1
    # ======================================================
    fname = r"E:\Coursera\CS155\LiuKe_hw_submission\prj3_code\data\shakespeare.txt"    
    seq_length = 10
    
    sonnets_in_char = load_poems_char_level(fname)
        
    chars = char_vocab(sonnets_in_char)
    print(chars[:5])    
    char_to_indices = dict((char,i) for i, char in enumerate(chars))
    indices_to_char = dict((i, char) for i, char in enumerate(chars))
    
    seqs, next_char = seq_with_next_char(sonnets_in_char, seq_length)
    
    # b. Convert the chars into numbers: 
    train_x = np.zeros((len(seqs), seq_length, len(chars)))
    train_y = np.zeros((len(seqs), len(chars)))
    
    # One-hot encodding: 
    for i, sentence in enumerate(seqs):
        for j, single_char in enumerate(sentence):
            train_x[i, j, char_to_indices[single_char]] = 1
        train_y[i, char_to_indices[next_char[i]]] = 1
        
    
    # ======================================================
    # Function 2: Train RNN
    # ====================================================== 
    num_unit = 100
    input_dim = len(chars)   
    model = build_rnn(num_unit, seq_length, input_dim) 
    
    optimizer = RMSprop(lr=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)
    
    print_callback = LambdaCallback(on_epoch_end=on_epoch_end)
    model.fit(train_x, train_y, batch_size=128, epochs=60, callbacks=[print_callback])