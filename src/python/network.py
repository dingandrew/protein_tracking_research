import os
import matplotlib.pyplot as plt
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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

        # self.mask_feature = FeatureExtractor(self.params)

        self.frame_features = FeatureExtractor(self.params)

        # bi directional to get forwards and backwards predicions
        # self.rnn = nn.RNN(input_size=4096, hidden_size=4, nonlinearity='tanh',
        #                   batch_first=True, num_layers=32, bidirectional=True)

        self.fcIn = nn.Linear(12480, 12480)
        self.fc1 = nn.Linear(12480, 3120)
        self.fc2 = nn.Linear(3120, 3120)
        self.fc3 = nn.Linear(3120, 6240)
        # add target features here
        self.fc4 = nn.Linear(6240, 3120)
        self.fc5 = nn.Linear(3120, 390)
        self.fc6 = nn.Linear(390, 4)
        # concatenate init_ground here
        self.fcOut = nn.Linear(8, 4)

        self.loss_calculator = Loss_Calculator()

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

        # forward frame1 to frame2
        fullFeatures1 = torch.cat([frame1Features, frame2Features], dim=0)
        # print(fullFeatures1.shape)
        H = self.fcIn(fullFeatures1)
        H = self.fc1(H)
        H = F.relu(self.fc2(H))
        H = self.fc3(H)
        H = H + targetFeatures
        H = self.fc4(H)
        H = F.relu(self.fc5(H))
        H = self.fc6(H)
        H = torch.cat([H, init_ground], dim=0)
        H = self.fcOut(H)
        # apply activations first index is confidence last three are coords
        confidence = H[0:1]
        coordinates = H[1:4]
        tuple_of_activated_parts = (
            F.sigmoid(confidence),
            F.relu(coordinates)
        )
        out1 = torch.cat(tuple_of_activated_parts, dim=0)
        # calculate loss
        # loss1 = self.loss_calculator(out1, init_ground)
        # print('loss1', loss1)

        # get pseudolabel
        pseudo_ground = out1.detach().clone()
        # if pseudo_ground[0] < 0.5:
        #     pseudo_ground[:] = 0

        # print('pseudo=-----', pseudo_ground)

        # backward frame2 to frame1
        fullFeatures2 = torch.cat([frame2Features, frame1Features], dim=0)
        # print(fullFeatures2.shape)
        H = self.fcIn(fullFeatures2)
        H = self.fc1(H)
        H = F.relu(self.fc2(H))
        H = self.fc3(H)
        H = H + targetFeatures
        H = self.fc4(H)
        H = F.relu(self.fc5(H))
        H = self.fc6(H)
        H = torch.cat([H, pseudo_ground], dim=0)
        H = self.fcOut(H)
        # apply activations first index is confidence last three are coords
        confidence = H[0:1]
        coordinates = H[1:4]
        tuple_of_activated_parts = (
            F.sigmoid(confidence),
            F.relu(coordinates)
        )
        out2 = torch.cat(tuple_of_activated_parts, dim=0)
        # calculate loss
        # loss2 = self.loss_calculator(out2, pseudo_ground)
        # print('loss2', loss2)

        tqdm.write('INIT: {} OUT1: {} OUT2: {}'.format(
            init_ground.detach(), pseudo_ground.detach(), out2.detach()))
        # print(out1)
        # print(init_ground)
        # print(out2) + 0.25 * loss1 + 0.25 * loss2

        lossTotal = self.loss_calculator(out2, init_ground)

        # print('loss--', lossTotal)
        # batch, time_frame, (x_filter_size * y_filter_size), final output filters
        # INPUT = batch sepquence_len input_size
        # h_0 shape = (num_layers * 2, batch, input_size)
        # out shape = (batch, seq_len, num_directions * hidden_size)
        # h_n shape  = (num_layers * num_directions, batch, hidden_size)

        # out, h_n = self.rnn(frameFeatures, target.cuda())
        # print(out.shape, h_n.shape) # torch.Size([1, 70, 8]) torch.Size([6, 1, 4])
        # print('out: ', out)

        # loss = self.loss_calculator(out, pseudo_ground)

        return lossTotal, out1, out2
