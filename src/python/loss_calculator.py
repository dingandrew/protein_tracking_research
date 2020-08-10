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
        # torch.Size([1, 70, 8]) torch.Size([6, 1, 4])
        # actual = prediction[:, :, 4:].clone().detach()
        # actual[0, ground_time, 0:4] = target

        # loss = self.mse(prediction[:,:, :4], actual)
        # return loss
        pred_confidence = prediction[0:1]
        pred_coordinate = prediction[1:4]

        targ_confidence = target[0:1]
        targ_coordinate = target[1:4]

        # print(pred_confidence, targ_confidence, loss_type)

        if loss_type == 'forward':
            loss = 0

            if pred_confidence < self.params['confidence_thresh']:
                if pred_confidence == 0:
                    multiplyer = 10
                elif pred_confidence < 0.02:
                    multiplyer = 5
                else:
                    multiplyer = 1

                loss = multiplyer * self.mse(pred_confidence,
                                             torch.tensor([self.params['confidence_thresh']]).float().cuda())
            else:
                if pred_confidence == 1:
                    multiplyer = 10
                elif pred_confidence > 0.99:
                    multiplyer = 5
                else:
                    multiplyer = 1

                # TODO: refactor this

                error_dist = torch.abs(pred_coordinate - targ_coordinate)
                # print('fffff', error_dist, multiplyer)
                if error_dist[0] < self.params['z_window']:
                    loss += 0
                else:
                    loss += multiplyer * \
                        self.mse(pred_coordinate[0], targ_coordinate[0])

                if error_dist[1] < self.params['x_window']:
                    loss += 0
                else:
                    loss += multiplyer * \
                        self.mse(pred_coordinate[1], targ_coordinate[1])

                if error_dist[2] < self.params['y_window']:
                    loss += 0
                else:
                    loss += multiplyer * \
                        self.mse(pred_coordinate[2], targ_coordinate[2])

        elif loss_type == 'backward':
            loss = self.mse(pred_coordinate, targ_coordinate)

        # loss = self.mse(prediction, target)
        # print('loss------------------', loss_type ,  loss)
        return loss
