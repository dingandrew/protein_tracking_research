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

        2. Feed feature frames and target cluster to RNN 

        3. Use forwards/backwards loss, bidirectional RNN on each time_step going backwards to 
           ensure we do not claim the cluster exists before its birth 

        4. Find untracked clusters then track them using new unique ground truth IDs, 

        5. Since we already know the location of object from segmentation we only
           need to determine if the object exists in the next frame, RNN has  
           OUTPUT_SHAPE = [batch, time_steps, 1], feature of channel 1 predicts whther or not
           this id exixts in this frame

        6. (Inference mode only) store the predictions results

        7. Calculate loss and return 
    '''
    def __init__(self, params):
        '''
            See model_config.json for parameter values

            Input: args, for the task to set up the network for
        '''
        super(Network, self).__init__()
        self.params = params.copy()
        # self.labels = train_labels

        self.mask_feature = FeatureExtractor(self.params)

        self.frame_features = FeatureExtractor(self.params)

        #bi directional to get forwards and backwards predicions
        # self.rnn = nn.RNN(input_size=64, hidden_size=3,
        #                   batch_first=True, num_layers=3, bidirectional=True)

        self.loss_calculator = Loss_Calculator(self.params)

    def forward(self, input_seq, target, init_label):
        '''
            1. input_seq -> fetaure_extractor => features
            2. features -> RNN(bidrectional=true) => forwards and backwards predictions
            3. predictions -> loss_calculator => loss
            4. return loss

            Input: input_seq has shape [batch, time_step, depth, z, x, y]
                   target, the object we are trying to track through time
                           it is an binary mask where
                   init_label, labels for frame where target exists in series other 
                               time_steps have initialized 0 labels, update with forward tracking 
                               as we go, [batch, time_steps, 1]
        '''
        label = init_label

        #run cnn's on input
        frameFeatures = self.frame_features(input_seq)
        maskFeatures = self.mask_feature(target)


      

        # out shape = (batch, seq_len, num_directions * hidden_size)
        # h_n shape  = (num_layers * num_directions, batch, hidden_size)

        # out, h_n = self.rnn(inputs)






        return loss, label
