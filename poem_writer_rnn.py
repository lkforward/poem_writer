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
  
from rnn_utils import preprocess_text, load_poems_char_level
        
    
def sampling(preds, temperature):
    # Sample a character according to the distribution predicted by the softmax function
    # in RNN 
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    
    return np.argmax(probas)


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
    
    
def write_str_into_file(fname, text_str):
    with open(fname, 'w') as fid: 
        fid.write(text_str)


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
    print("seq length =", seq_length)
    generated = ''
    sentence = ''.join(sonnets_in_char[start_index : start_index + seq_length])
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
    if epoch == 2:
        fname = r"E:\Coursera\CS155\LiuKe_hw_submission\prj3_code\generated.txt"
        write_str_into_file(fname, generated)
        

 
class rnn_writer():
    def __init__(self, fname, seq_length):
        self.text_fname = fname
        self.seq_length = seq_length
        
        
        
    def fit_generate(self, num_units=100, batch_size=128, epochs=60):
        
        self.train_x, self.train_y, self.indices_to_char = preprocess_text(self.fname, self.seq_length)
        self.chars = list(indices_to_char.values())
        
        self.num_units = num_units
        self.model = build_rnn(self.num_units, self.seq_length, len(self.chars))
        
        optimizer = RMSprop(lr=0.01)
        self.model.compile(loss='categorical_crossentropy', optimizer=optimizer)
        
        print_callback = LambdaCallback(on_epoch_end=on_epoch_end)
        self.model.fit(self.train_x, self.train_y, batch_size=batch_size, 
                       epochs=epochs, callbacks=[print_callback])
        
        
            
        
# if __name__ == "__main__":
# ======================================================
# Function 1
# ======================================================
fname = r"E:\Coursera\CS155\LiuKe_hw_submission\prj3_code\data\shakespeare.txt"    
seq_length = 10

sonnets_in_char = load_poems_char_level(fname)  

train_x, train_y, char_to_indices, indices_to_char = preprocess_text(fname, seq_length)
chars = list(indices_to_char.values())
    

# ======================================================
# Function 2: Train RNN
# ====================================================== 
num_unit = 100
input_dim = len(chars)   
model = build_rnn(num_unit, seq_length, input_dim) 

optimizer = RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)

print_callback = LambdaCallback(on_epoch_end=on_epoch_end)
model.fit(train_x, train_y, batch_size=128, epochs=3, callbacks=[print_callback])