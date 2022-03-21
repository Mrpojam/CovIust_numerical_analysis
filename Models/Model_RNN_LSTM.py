# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 19:11:01 2022

@author: Pouya-Jafari
"""
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Dropout
from tensorflow.python.keras.layers import SimpleRNNCell
from tensorflow.python.keras.layers import RNN, LSTM

def simple_rnn (
        output_size,
        neurans,
        activation='tanh',
        dropout = 0.2,
        loss='mse',
        optimizaer='adam'):
    model = Sequential()
    model.add(RNN(cell=[SimpleRNNCell(128),
                        SimpleRNNCell(256),
                        SimpleRNNCell(128)]))
    model.add(Dropout(dropout))
    model.add(Dense(units=output_size))
    model.add(Activation(activation))
    
    model.compile(loss=loss, optimizer=optimizaer)
    
    return model

def lstm_rnn (
        output_size,
        neurans,
        activation='linear',
        dropout = 0.2,
        loss='mse',
        optimizaer='adagrad'):
    
    model = Sequential()
    model.add(LSTM(256, input_shape = (4, 1), return_sequences = True))
    model.add(LSTM(128))
    model.add(Dense(output_size))
    model.add(Activation(activation))
    
    model.compile(loss=loss, optimizer=optimizaer)
    
    return model 





