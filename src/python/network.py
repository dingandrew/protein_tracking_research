import os
import matplotlib.pyplot as plt
import sys
import numpy as np
import torch
import torch.nn as nn
import time
from tqdm import tqdm
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

        NOTE: all batch sizes are 1
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
        self.rnn = nn.RNN(input_size=4096, hidden_size=4,
                          batch_first=True, num_layers=3, bidirectional=True)

        self.loss_calculator = Loss_Calculator()

    def forward(self, input_seq, target, init_time):
        '''
            1. input_seq -> fetaure_extractor => features
            2. features -> RNN(bidrectional=true) => forwards and backwards predictions
            3. predictions -> loss_calculator => loss
            4. return loss

            Input: input_seq has shape [batch, time_step, depth, z, x, y]
                   target, the object we are trying to track through time
                           it is h_0 

        '''
        #run cnn's on input
        frameFeatures = self.frame_features(input_seq)
        frameFeatures = frameFeatures.reshape((1, 70 , 4096))
        tqdm.write('frame feautres shape: {}'.format(frameFeatures.shape))

        # maskFeatures = self.mask_feature(target)
        # maskFeatures = maskFeatures.reshape((1, 1, 4096)) [6,1,4]
        # tqdm.write('mask features shape: {}'.format(maskFeatures.shape))

        # batch, time_frame, (x_filter_size * y_filter_size), final output filters
        # INPUT = batch sepquence_len input_size
        # h_0 shape = (num_layers * 2, batch, input_size)
        # out shape = (batch, seq_len, num_directions * hidden_size)
        # h_n shape  = (num_layers * num_directions, batch, hidden_size)

        out, h_n = self.rnn(frameFeatures, target)
        # print(out.shape, h_n.shape) # torch.Size([1, 70, 8]) torch.Size([6, 1, 4])
        # print('out: ', out) 

        loss = self.loss_calculator(out, target[0, 0, ...], init_time - 1)

        return loss, out
