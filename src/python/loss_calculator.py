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

    def forward(self, prediction, target, loss_type):
        loss = 0
        if loss_type == 'f2':
            
            # TODO: refactor this
            error_dist = torch.abs(prediction - target)
            
            if error_dist[0] < self.params['z_window']:
                loss += 0
            else:
                loss += self.mse(prediction[0] , target[0] )

            if error_dist[1] < self.params['x_window']:
                loss += 0
            else:
                loss += self.mse(prediction[1] , target[1])

            if error_dist[2] < self.params['y_window']:
                loss += 0
            else:
                loss += self.mse(prediction[2] , target[2] )

        elif loss_type == 'f1':
            loss += self.mse(prediction[0], target[0] )
            loss += self.mse(prediction[1] , target[1] )
            loss += self.mse(prediction[2] , target[2] )
        return loss
