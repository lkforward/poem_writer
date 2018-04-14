# -*- coding: utf-8 -*-
"""
Utility functions for the RNN poem writer.

Created on Sat Apr 14 17:03:38 2018

@author: kliu14
"""
import numpy as np
# from utils import load_poems

    
def load_poems_char_level(fname):
    """
    [INPUTS]:
    fname: string. 
        The file name for the text data. 
        
    [OUTPUTS]:
    poems: list. 
        The list of characters in the text file. Each sonnet is a string 
        (including the change of line). 
    """
    
    sonnets = []
    
    with open(fname, 'r') as fid: 
        sonnet = ''
        do_append = False
        
        for line in fid:
            # A digit suggests the next line is a new sonnet: 
            if line.strip().split(' ')[-1].isdigit():
                do_append = True
            # Blank line: 
            elif line == '\n':
                if do_append: 
                    sonnets.append(sonnet)
                    sonnet = ''
                    do_append = False
            # Other lines are the poems:         
            else:
                sonnet += line.strip(' ').lower()
    
    return sonnets
    
    
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
    
    return train_x, train_y, char_to_indices, indices_to_char
    

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
    # test_preprocess()
    test_char_vocab()
    test_seq_with_next_char()