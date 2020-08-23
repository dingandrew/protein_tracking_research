import torch.nn as nn
import torch
import numpy as np


class Loss_Calculator(nn.Module):
    '''
        Calculate the loss by using forwards backwards tracking

    '''

    def __init__(self, params):
        super(Loss_Calculator, self).__init__()
        self.params = params
        self.mse = nn.MSELoss()

    def forward(self, prediction, target):
        target = target.view(-1,
                             target.size(2),
                             target.size(3),
                             target.size(4),
                             target.size(5))  # (batch * time_frame), D, Z , H, W
        return self.mse(prediction, target)
