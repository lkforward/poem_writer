# -*- coding: utf-8 -*-
"""
Utility functions for the generative model for Shakespeare's poeam. 
Created on Thu Mar 15 13:21:18 2018

@author: kliu14
"""

# from nltk.tokenize import word_tokenize
import re


def load_poems(filename):
    """ 
    [OUTPUTS]
    Sonnets: list of sonnets.
        Each sonnet is a list of sentences. 
    """
    sonnets = []
    with open(filename, 'r') as fpoems: 
        sonnet = []
        do_append = False
        
        for line in fpoems:
            if line.strip().split(' ')[-1].isdigit():
                do_append = True
                continue
            elif line == '\n':
                if do_append:
                    sonnets.append(sonnet)
                    sonnet = []
                    do_append = False
                continue
            sonnet.append(line.strip().lower())
            
        sonnets.append(sonnet)
        
    return sonnets
    
    
def load_poems_char_level(fname):
    """
    [INPUTS]:
    fname: string. 
        The file name for the text data. 
        
    [OUTPUTS]:
    poems: list. 
        The list of characters in the text file. 
    """
            
    # Open the file and read it line by line:
    #   if it is not a blank line:
    #       Read the text char by char (include the punctuation and line break);
    #   else:
    #       skip. 
    
    sonnets = []
    
    with open(fname, 'r') as fid: 
        sonnet = ''
        do_append = False
        
        for line in fid:
            if line.strip().split(' ')[-1].isdigit():
                do_append = True
            elif line == '\n':
                if do_append: 
                    sonnets.append(sonnet)
                    sonnet = ''
                    do_append = False
            else:
                sonnet += line.strip(' ')
    
    return sonnets


def tokenize(sentence):
    return [re.sub(r'[^\w\s\']', '', w) for
            w in sentence.strip().split(' ')]


class poems_translator():
    """
    A class to translate from poems to emissions (of integers) and also translate
    sequence of emissions back into poems. 
    """
    def __init__(self, poemfilename):
        self.sonnets = load_poems(poemfilename)
        self.unique_words = self.unique_words_in_poems()
        

    def unique_words_in_poems(self):
        unique_words = set()
        
        for sonnet in self.sonnets:
            for sentence in sonnet:
                # tokens = word_tokenize(sentence)
                tokens = tokenize(sentence)
                unique_words |= set(tokens)
        
        return list(unique_words)


    def poems2emissions(self):
        """
        [INPUTS]:
        sonnets: list of sonnets.
            Each sonnet is a list of sentences.
        [OUTPUTS]:
        obs: list of emissions (observations).
            Each emission is a list of integers from the tokens. 
        """
        
        obs = []    
        for sonnet in self.sonnets:
            for sentence in sonnet: 
                # tokens = word_tokenize(sentence)
                tokens = tokenize(sentence)
                emission_one_line = [self.unique_words.index(token) for token in tokens]
                obs.append(emission_one_line)
        
        return obs


    def emissions2poems(self, obs):
        """
        Generate poems from another observations using the word set in the data. 
        [INPUTS]:
        obs: list of sentences. 
            Each sentence is a list of integers. 

        [OUTPUTS]
        sonnet: a list of sentences (string). 
        """
        sonnet = []
        for i, sentence in enumerate(obs):
            words = []
            for token in sentence:
                words.append(self.unique_words[token])
                
            if i < len(obs) - 1:
                sentence = ' '.join(words).join(('', ','))
            else:
                sentence = ' '.join(words).join(('', '.'))
                
            sonnet.append(sentence)
        return sonnet
    
    
def test_generating_emissions():
    fname = r"data\shakespeare.txt"
    poems = poems_translator(fname)
    obs = poems.poems2emissions()
    
    print("")
    print("The first line of the poems:")
    print(poems.sonnets[0][0])
    print(obs[0])
    print("")
    print("The last line of the poems:")
    print(poems.sonnets[-1][-1])
    print(obs[-1])


if __name__ == "__main__":
    # test_generating_emissions()
    fname = r"data\shakespeare.txt"
    sonnets = load_poems_char_level(fname)