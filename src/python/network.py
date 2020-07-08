import os
import matplotlib.pyplot as plt
import sys
import numpy as np
import tensorflow.keras as keras


class Network:
    def __init__(self, batch_size, time_step):
        self.batchSize = batch_size
        self.timeSteps = time_step

    def build_network(self):
        '''
            Build the network using keras functional API

        '''

        input_labels = keras.layers.Input(
            shape=(70, 70), batch_size=10, name='forward_input')

        input_full = keras.layers.Input(
            shape=(258, 512, 13, 1), batch_size=10, name='backward_input')

        input_backwards = keras.layers.Input(
            shape=(70, 70), batch_size=10, name='backward_input')

        deep_rnn1 = keras.layers.SimpleRNN(
            units=70, return_sequences=True)(input_labels)

        deep_rnn2 = keras.layers.SimpleRNN(units=70)(deep_rnn1)



        hidden_cnn = keras.layers.TimeDistributed(
            keras.layers.Conv3D(filters=64, kernel_size=3))(input_full)

        output1 = keras.layers.Dense(1)([deep_rnn2, hidden_cnn])

        model = keras.models.Model(
            inputs=[input_forwards, input_backwards], outputs=[output1])
        return model


class Simple_Model:
    def __init__(self):
        self.layers = []
        self.x


    def run():
        model = keras.models.Sequential(layers=self.layers, name='simple')
        model.compile(optimizer='rmsprop', loss=None, metrics=None,
                      loss_weights=None, sample_weight_mode=None, weighted_metrics=None)
        model.fit(x=None, y=None, batch_size=None, epochs=1, verbose=1, callbacks=None, 
                  validation_split=0., validation_data=None, shuffle=True, class_weight=None,
                  sample_weight=None, initial_epoch=0, steps_per_epoch=None, validation_steps=None, 
                  validation_batch_size=None, validation_freq=1, max_queue_size=10, workers=1, 
                  use_multiprocessing=False)





class R_CNN:
    '''
        Subclassing API used here
    '''
    def __init__(self, filters, strides, activation, **kwargs):
        '''
            Initialize an recurrent cnn layer  
        '''
        super.__init__(**kwargs)




    def call(self, inputs):
        '''
            pg 478
        '''
        pass


    # print(tf.config.list_physical_devices('GPU'))
    # print(tf.test.is_built_with_cuda())
    # print(tf.test.is_built_with_gpu_support())



















    # def get_model(self, data_in, data_out, _cnn_nb_filt, _cnn_pool_size, _rnn_nb, _fc_nb):

    #     spec_start = Input( shape=(data_in.shape[-3], data_in.shape[-2], data_in.shape[-1]))
        
    #     spec_x = spec_start

    #     for _i, _cnt in enumerate(_cnn_pool_size):
    #         spec_x = Conv2D(filters=cnn_nb_filt, kernel_size=(2, 2), padding='same')(spec_x)
    #         spec_x = BatchNormalization(axis=1)(spec_x)
    #         spec_x = Activation(‘relu’)(spec_x)
    #         spec_x = MaxPooling2D(pool_size=(1, _cnn_pool_size[_i]))(spec_x)
    #         spec_x = Dropout(dropout_rate)(spec_x)
    #         spec_x = Permute((2, 1, 3))(spec_x)
    #         spec_x = Reshape((data_in.shape[-2], -1))(spec_x)
        
    #     for _r in _rnn_nb:
    #         spec_x = Bidirectional(
    #             GRU(_r, activation=’tanh’, dropout=dropout_rate, recurrent_dropout=dropout_rate, return_sequences=True),
    #             merge_mode=’concat’)(spec_x)
        
    #     for _f in _fc_nb:
    #         spec_x = TimeDistributed(Dense(_f))(spec_x)
    #         spec_x = Dropout(dropout_rate)(spec_x)
        
    #     spec_x = TimeDistributed(Dense(data_out.shape[-1]))(spec_x)
    #     out = Activation(‘sigmoid’, name=’strong_out’)(spec_x)
        
    #     _model = Model(inputs=spec_start, outputs=out)
    #     _model.compile(optimizer=’Adam’, loss=’binary_crossentropy’, metrics=[‘accuracy’])
    #     _model.summary()
        
        
    #     return _model
