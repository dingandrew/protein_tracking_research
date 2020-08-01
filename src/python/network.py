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

        #bi directional to get forwards and backwards predicions
        # self.rnn = nn.RNN(input_size=4096, hidden_size=4, nonlinearity='tanh',
        #                   batch_first=True, num_layers=32, bidirectional=True)

        self.fcIn = nn.Linear(8192, 16384)
        self.fc1 = nn.Linear(16384, 16384)
        self.fc2 = nn.Linear(16384, 8192)
        self.fc3 = nn.Linear(8192, 4096)
        # concatenate target here
        self.fc4 = nn.Linear(8192, 2048)
        self.fc5 = nn.Linear(2048, 512)
        self.fc6 = nn.Linear(512, 4)
        # concatenate init_ground here
        self.fcOut = nn.Linear(8, 4)

                                             
        # self.fcInput = nn.Linear(28 * 28, 200)
        # self.fc1 = nn.Linear(200, 200)
        # self.fc2 = nn.Linear(200, 10)
        # self.fc3 = nn.Linear(200, 10)
        # self.fc4 = nn.Linear(200, 10)
        # self.fcOutput = nn.Linear(200, 10)

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
        #run cnn's on input
        frame1Features = self.frame_features(frame1)
        frame1Features = frame1Features.reshape((4096))
        
        frame2Features = self.frame_features(frame2)
        frame2Features = frame2Features.reshape((4096))

        targetFeatures = self.frame_features(target)
        targetFeatures = targetFeatures.reshape((4096))

        # forward frame1 to frame2        
        fullFeatures1 = torch.cat([frame1Features, frame2Features], dim=0)
        print(fullFeatures1.shape)
        H = F.relu(self.fcIn(fullFeatures1))
        H = F.relu(self.fc1(H))
        H = F.relu(self.fc2(H))
        H = self.fc3(H)
        H = torch.cat([H, targetFeatures], dim=0)
        H = self.fc4(H)
        H = F.relu(self.fc5(H))
        H = F.relu(self.fc6(H))
        H = torch.cat([H, init_ground], dim=0)
        out1 = F.relu(self.fcOut(H))
        loss1 = self.loss_calculator(out1, init_ground)
        print('loss1', loss1)

        # get pseudolabel
        pseudo_ground = out1.detach().clone()
        if pseudo_ground[0] < 0.5:
            pseudo_ground[:] = 0

        print('pseudo=-----', pseudo_ground)

        # backward frame2 to frame1
        fullFeatures2 = torch.cat([frame2Features, frame1Features], dim=0)
        print(fullFeatures2.shape)
        H = F.relu(self.fcIn(fullFeatures2))
        H = F.relu(self.fc1(H))
        H = F.relu(self.fc2(H))
        H = self.fc3(H)
        H = torch.cat([H, targetFeatures], dim=0)
        H = self.fc4(H)
        H = F.relu(self.fc5(H))
        H = F.relu(self.fc6(H))
        H = torch.cat([H, pseudo_ground], dim=0)
        out2 = F.relu(self.fcOut(H))
        loss2 = self.loss_calculator(out2, pseudo_ground)
        print('loss2', loss2)

        print(out1)
        print(init_ground)
        print(out2)
        
        lossTotal = self.loss_calculator(out2, init_ground) + 0.25 * loss1 + 0.25 * loss2

        print('loss--', lossTotal)
        # batch, time_frame, (x_filter_size * y_filter_size), final output filters
        # INPUT = batch sepquence_len input_size
        # h_0 shape = (num_layers * 2, batch, input_size)
        # out shape = (batch, seq_len, num_directions * hidden_size)
        # h_n shape  = (num_layers * num_directions, batch, hidden_size)
   
        # out, h_n = self.rnn(frameFeatures, target.cuda())
        # print(out.shape, h_n.shape) # torch.Size([1, 70, 8]) torch.Size([6, 1, 4])
        # print('out: ', out) 
        
        # loss = self.loss_calculator(out, pseudo_ground)

        return loss1, out1, out2
