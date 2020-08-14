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
from feature_extractor import FeatureExtractor
from loss_calculator import Loss_Calculator
from util import open_model_json, showTensor

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

        # self.mask_feature = FeatureExtractor(self.params)

        self.frame_features = FeatureExtractor(self.params)

        # bi directional to get forwards and backwards predicions
        # self.rnn = nn.RNN(input_size=4096, hidden_size=4, nonlinearity='tanh',
        #                   batch_first=True, num_layers=32, bidirectional=True)

        self.fcIn = nn.Linear(6240, 3120)
        # add target features here
        self.fc1 = nn.Linear(3120, 1560)
        self.fc2 = nn.Linear(1560, 780)
        # self.fc3 = nn.Linear(780, 6240)
        
        self.fc4 = nn.Linear(780, 100)
        self.fc5 = nn.Linear(100, 50)
        self.fc6 = nn.Linear(50, 4)
        # concatenate init_ground here
        self.fcOut = nn.Linear(8, 4)

        self.tanhshrink = nn.Tanhshrink()
        self.softsign = nn.Softsign()
        self.sigmoid = nn.Sigmoid()

        self.loss_calculator = Loss_Calculator(self.params)

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
        # run cnn's on input
        frame1Features = self.frame_features(frame1)
        frame2Features = self.frame_features(frame2)
        targetFeatures = self.frame_features(target)


        # torch.set_printoptions(profile="full")
        # print(targetFeatures)
        # print(frame1Features)
        # print(frame2Features)
        # # print(torch.abs(frame1Features) - torch.abs(frame2Features))
        
        # # ff = torch.abs(frame1Features) - torch.abs(frame2Features)
        # # ft1 = torch.abs(frame1Features) - torch.abs(targetFeatures)
        # # ft2 = torch.abs(frame2Features) - torch.abs(targetFeatures)

        # print(3120 - (targetFeatures == 0).sum(dim=0))
        # print(3120 - (frame1Features == 0).sum(dim=0))
        # print(3120 - (frame2Features == 0).sum(dim=0))

        # print('Frames feature', F.mse_loss(frame1Features, frame2Features))
        # print('Raw frames', F.mse_loss(frame1, frame2))
        # print('frame1 and targ', F.mse_loss(frame1Features, targetFeatures))
        # print('frame2 and targ', F.mse_loss(frame2Features, targetFeatures))


        # exit()
        # forward frame1 to frame2
        fullFeatures1 = torch.cat([frame1Features, frame2Features], dim=0)
        H = self.fcIn(fullFeatures1)
        
        H = H + targetFeatures

        H = self.sigmoid(self.fc1(H))
        H = self.fc2(H)
        # H = self.fc3(H)
    
        H = self.fc4(H)
        H = self.fc5(H)
        H = self.sigmoid(self.fc6(H))
        H = torch.cat([H, init_ground], dim=0)
        H = self.fcOut(H)
        # apply activations first index is confidence last three are coords
        confidence = H[0:1]
        coordinates = H[1:4]
        tuple_of_activated_parts = (
            self.sigmoid(confidence),
            F.relu(coordinates)
        )
        out1 = torch.cat(tuple_of_activated_parts, dim=0)
        
        # print('loss1', loss1)
        # get pseudolabel
        pseudo_ground = out1.detach().clone()
        # if pseudo_ground[0] < self.params['confidence_thresh']:
        #     pseudo_ground[1:4] = torch.tensor([0, 0, 0]).float().cuda()
        # elif random() < self.params['cnn']['drop_out']:
        #     pseudo_ground = torch.tensor([0, 0, 0, 0]).float().cuda()
        # else:
        # causes both to be the same
        #     pseudo_ground[1:4] = init_ground[1:4]

        # calculate loss
        loss1 = self.loss_calculator(out1, init_ground, 'forward')

        # print('pseudo=-----', pseudo_ground)

        # backward frame2 to frame1
        fullFeatures2 = torch.cat([frame2Features, frame1Features], dim=0)
        H = self.fcIn(fullFeatures2)

        H = H + targetFeatures

        H = self.sigmoid(self.fc1(H))
        H = self.fc2(H)
        # H = self.fc3(H)
        
        H = self.fc4(H)
        H = self.fc5(H)
        H = self.sigmoid(self.fc6(H))
        H = torch.cat([H, pseudo_ground], dim=0)
        H = self.fcOut(H)
        # apply activations first index is confidence last three are coords
        confidence = H[0:1]
        coordinates = H[1:4]
        tuple_of_activated_parts = (
            self.sigmoid(confidence),
            F.relu(coordinates)
        )
        out2 = torch.cat(tuple_of_activated_parts, dim=0)
        # calculate loss
        loss2 = self.loss_calculator(out2, init_ground, 'backward')
        # print('loss2', loss2)

        tqdm.write('INIT: {} OUT1: {} OUT2: {}\n\tpseudo: {}'.format(
            init_ground.detach(), out1.detach(), out2.detach(), pseudo_ground))
        
        # + 0.25 * loss1 + 0.25 * loss2
        lossTotal = loss1 + 0.25 * loss2 


        return lossTotal, out1, out2
