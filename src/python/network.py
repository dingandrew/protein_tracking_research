import os
import matplotlib.pyplot as plt
import sys
import numpy as np
import torch
import torch.nn as nn
from feature_extractor import FeatureExtractor
from loss_calculator import Loss_Calculator


class Network(nn.Module):
    '''
        End to end tracker on segmented input
        1. Put frames through feature tracker

        2. Feed feature frames to RNN

        3. Use forwards/backwards loss, bidirectional on each time_step go backwards to 
           ensure we do not claim the cluster exists before its birth 

        4. Find untracked clusters then track them using new unique ground truth IDs, 

        5. Since we already know the location of object from segmentation we only
           need to determine if the object exists in the next frame 
           OUTPUT_SHAPE = [batch, time_steps, 1], feature of channel 1 predicts whther or not
           this id exixts in this frame

        
    '''
    def __init__(self, args):
        '''
            See model_config.json for parameter values
        '''
        super(Network, self).__init__()
        self.args = args.default.copy()
        self.feature_extractor = FeatureExtractor(self.args)

        #bi directional to get forwards and backwards predicions
        self.rnn = nn.RNN(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE,
                          batch_first=True, num_layers=1, bidirectional=True)

        # input size : (batch_size , seq_len, input_size)
        inputs = data.view(BATCH_SIZE, SEQ_LENGTH, INPUT_SIZE)

        # out shape = (batch, seq_len, num_directions * hidden_size)
        # h_n shape  = (num_layers * num_directions, batch, hidden_size)

        out, h_n = rnn(inputs)
        self.loss_calculator = Loss_Calculator(self.args)

    def forward(self, input_seq):
        '''

        '''
