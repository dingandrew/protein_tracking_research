import torch.nn as nn
import torch
import numpy as np




class Loss_Calculator(nn.Module):
    '''
        Calculate the loss by using forwards backwards tracking

    '''
    def __init__(self):
        super(Loss_Calculator, self).__init__()
        
        self.mse = nn.MSELoss()


    def forward(self, prediction, target, ground_time):
        # torch.Size([1, 70, 8]) torch.Size([6, 1, 4])
        actual = prediction.clone().detach()
        actual[0, ground_time, 0:4] = target
        actual[0, ground_time, 4:8] = target
        loss = self.mse(prediction, actual)
        return loss
