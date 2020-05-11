#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Defines functions returning tensorflow.keras model instances.

Author: Espen Hagen (<https://github.com/espenhgn>)

LICENSE: <https://github.com/espenhgn/RippleNet/blob/master/LICENSE>
'''
from tensorflow import keras
from tensorflow.keras.initializers import GlorotUniform, Orthogonal


def get_unidirectional_LSTM_model(input_shape, lr=0.005, dropout_rate=0.2,
                                  layer_sizes=[20, 10, 10, 10], stddev=0.001,
                                  seed=0):
    '''
    Parameters
    ----------
    input_shape: length 2 tuple
        input dimensionality (int or None, 1)
    lr: float
        Adam optimizer learning rate
    dropout_rate: float in <0, 1>
        Dropout layer dropout fraction during training
    layer_sizes: list of int
        size of layers [Cov1D, Conv1D, LSTM, LSTM]
    stddev: float
        standard deviation of Gaussian noise layer
    seed: int
        random seed+i for layers where i is index of layers that can be seeded
    Returns
    -------
    tf.keras Model instance
    '''
    keras.backend.clear_session()

    # input layer
    inputs = keras.layers.Input(shape=input_shape)

    # Gaussian noise layer
    x = keras.layers.GaussianNoise(stddev)(inputs)

    # conv layer 1
    x = keras.layers.Conv1D(layer_sizes[0],
                            kernel_size=11, strides=1,
                            kernel_initializer=GlorotUniform(seed=seed),
                            padding='same',
                            use_bias=False,
                            )(x)
    x = keras.layers.Dropout(dropout_rate, seed=seed+1)(x)

    # conv layer 2
    x = keras.layers.Conv1D(layer_sizes[1],
                            kernel_size=11, strides=1,
                            kernel_initializer=GlorotUniform(seed=seed+2),
                            padding='same',
                            use_bias=True,
                            )(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.Dropout(dropout_rate, seed=seed+3)(x)

    # LSTM layer 1
    x = keras.layers.LSTM(layer_sizes[2], return_sequences=True,
                          kernel_initializer=GlorotUniform(seed=seed+4),
                          recurrent_initializer=Orthogonal(seed=seed+5),
                          )(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(dropout_rate, seed=seed+6)(x)

    # LSTM layer 2
    x = keras.layers.LSTM(layer_sizes[3], return_sequences=True,
                          kernel_initializer=GlorotUniform(seed=seed+7),
                          recurrent_initializer=Orthogonal(seed=seed+8),
                          )(x)
    x = keras.layers.Dropout(dropout_rate, seed=seed+9)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(dropout_rate, seed=seed+10)(x)

    # dense output layer
    predictions = keras.layers.TimeDistributed(
        keras.layers.Dense(1, activation='sigmoid',
                           kernel_initializer=GlorotUniform(seed=seed+11)
                           ))(x)

    # Define model
    model = keras.models.Model(inputs=inputs,
                               outputs=predictions,
                               name='RippleNet')

    opt = keras.optimizers.Adam(lr=lr)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['mse'])

    return model


def get_bidirectional_LSTM_model(input_shape, lr=0.01, dropout_rate=0.2,
                                 layer_sizes=[20, 10, 10, 10], stddev=0.001,
                                 seed=0):
    '''
    Parameters
    ----------
    input_shape: length 2 tuple
        input dimensionality (int or None, 1)
    lr: float
        Adam optimizer learning rate
    dropout_rate: float in <0, 1>
        Dropout layer dropout fraction during training
    layer_sizes: list of int
        size of layers [Cov1D, Conv1D, LSTM, LSTM]
    stddev: float
        standard deviation of Gaussian noise layer

    Returns
    -------
    tf.keras Model instance
    '''
    keras.backend.clear_session()

    # input layer
    inputs = keras.layers.Input(shape=input_shape)

    # Gaussian noise layer
    x = keras.layers.GaussianNoise(stddev)(inputs)

    # conv layer 1
    x = keras.layers.Conv1D(layer_sizes[0],
                            kernel_size=11, strides=1,
                            kernel_initializer=GlorotUniform(seed=seed),
                            padding='same',
                            use_bias=False,
                            )(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(dropout_rate, seed=seed+1)(x)

    # conv layer 2
    x = keras.layers.Conv1D(layer_sizes[1],
                            kernel_size=11, strides=1,
                            kernel_initializer=GlorotUniform(seed=seed+2),
                            padding='same',
                            use_bias=True,
                            )(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.Dropout(dropout_rate, seed=seed+3)(x)

    # LSTM layer 1
    x = keras.layers.Bidirectional(keras.layers.LSTM(layer_sizes[2],
                                                     kernel_initializer=GlorotUniform(seed=seed+4),
                                                     recurrent_initializer=Orthogonal(seed=seed+5),
                                                     return_sequences=True,
                                                     )
                                   )(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(dropout_rate, seed=seed+6)(x)

    # LSTM layer 2
    x = keras.layers.Bidirectional(keras.layers.LSTM(layer_sizes[3],
                                                     return_sequences=True,
                                                     kernel_initializer=GlorotUniform(seed=seed+7),
                                                     recurrent_initializer=Orthogonal(seed=seed+8),
                                                     )
                                   )(x)
    x = keras.layers.Dropout(dropout_rate, seed=seed+9)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(dropout_rate, seed=seed+10)(x)

    # dense output layer
    predictions = keras.layers.TimeDistributed(
        keras.layers.Dense(1, activation='sigmoid',
                           kernel_initializer=GlorotUniform(seed=seed+11)))(x)

    # Define model
    model = keras.models.Model(inputs=inputs,
                               outputs=predictions,
                               name='RippleNet')

    opt = keras.optimizers.Adam(lr=lr)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['mse'])

    return model
