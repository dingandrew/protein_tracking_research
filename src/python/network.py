import os
import matplotlib.pyplot as plt
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from random import random
from tqdm import tqdm
from encoder import Encoder
from decoder import Decoder
from loss_calculator import Loss_Calculator
from util import open_model_json, showTensor


class Network(nn.Module):
    '''
        End to end tracker on segmented input
    '''

    def __init__(self, params):
        '''
            See model_config.json for parameter values

            Input: args, for the task to set up the network for
        '''
        super(Network, self).__init__()
        self.params = params.copy()
        # self.labels = train_labels

        # self.mask_feature = FeatureExtractor(self.params)

        self.encoder = Encoder(self.params)
        self.decoder = Decoder(self.params)
        self.loss_calculator = Loss_Calculator(self.params)

        self.sigmoid = nn.Sigmoid()

    def forward(self, frame1, frame2, target, init_ground):
        '''
            1. input_seq -> fetaure_extractor => features
            2. features -> RNN(bidrectional=true) => forwards and backwards predictions
            3. predictions -> loss_calculator => loss
            4. return loss

            Input: input_seq has shape [batch, time_step, depth, z, x, y]
                   target, the object we are trying to track through time
                           it is h_0 

        '''
        # print(frame1.shape, frame2.shape, target.shape)
        # mutate the target for frame1 to simulate it for a diff frame
        mutated_target = self.mutate_target(target)

        # run cnn's on input
        frame1Features = self.encoder(frame1 + target)
        frame2Features = self.encoder(frame2 + target)

        # print('Frames feature', F.mse_loss(frame1Features, frame2Features))
        # print('Raw frames', F.mse_loss(frame1, frame2))

        f1 = self.decoder(self.sigmoid(frame1Features))
        f2 = self.decoder(self.sigmoid(frame2Features))
        # print(f1.shape, f2.shape)
        # print(target)
        loss1 = self.loss_calculator(f1, target)
        loss2 = self.loss_calculator(f2, target) # num of points is equal to orig freame

        return loss1 + loss2, f1, f2


    def mutate_target(self, target):
        '''
            Randomly shift, rotate, and zoom the target. 
        '''
        return None



if __name__ == "__main__":
    # test
    model_config = open_model_json('./model_config.json')
    model = Network(model_config['default'])
    print(model)
    param_num = sum([param.data.numel()
                     for param in model.parameters()])
    print('Parameter number: %.3f M' % (param_num / 1024 / 1024))
