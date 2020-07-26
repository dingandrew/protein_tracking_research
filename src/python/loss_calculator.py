import torch.nn as nn
import torch
import numpy as np




class Loss_Calculator(nn.Module):
    '''
        Calculate the loss by using forwards backwards tracking

    '''
    def __init__(self, args):
        super(Loss_Calculator, self).__init__()
        


    def forward(self, prediction, target):
        pass
