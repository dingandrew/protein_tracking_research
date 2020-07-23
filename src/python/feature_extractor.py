import torch.nn as nn
import torch
import numpy as np
from util import showTensor


class FeatureExtractor(nn.Module):
    '''
        Extract features from a frame. Imagine we have a cluster of variable
        size and shape in a frame. We only need represent it as a single point 
        within a smaller feature map, simplyfying the model. Simplify representation
        of data.
    '''

    def __init__(self, params):
        '''
            Initialize the feature extractor

            Input: args, dictionary containing all parameter information for TASK

            Params:
                params, model parameters
                layer_num, number of layers in the feature extractor 
                conv_features, the number of filters  
                conv_kernels, size of conv kernel 
                stride, note set all with equal stride
                padding

        '''
        super(FeatureExtractor, self).__init__()
        self.params = params.copy()
        self.cnnParams = self.params['cnn']

        # print('cnn params: ', self.params, ' ', type(self.params))
        self.layer_num = len(self.cnnParams['conv_features'])
        # print(self.params['conv_kernels'][1])

        # create a feature extractor deep model
        for layer in range(0, self.layer_num - 1):
            setattr(self,
                    'cnn3D_' + str(layer),
                    nn.Conv3d(in_channels=self.cnnParams['conv_features'][layer],
                              out_channels=self.cnnParams['conv_features'][layer+1],
                              kernel_size=self.cnnParams['conv_kernels'][layer],
                              stride=1,
                              padding=(self.cnnParams['conv_kernels'][layer][0]//2,
                                       self.cnnParams['conv_kernels'][layer][1]//2))
                    )
            setattr(self,
                    'batchNorm3D_' + str(layer),
                    nn.BatchNorm3d(
                        num_features=self.cnnParams['conv_features'][layer+1])
                    )
            setattr(self,
                    'maxPool3D_' + str(layer),
                    nn.AdaptiveMaxPool3d(
                        tuple(self.cnnParams['out_sizes'][layer]))
                    )
            setattr(self,
                    'dropOut3D_' + str(layer),
                    nn.Dropout3d(self.cnnParams['drop_out'])
                    )
        self.activation = nn.ReLU()

    def forward(self, input_seq):
        '''
            input shape: (batch, time_frame, depth, z, x, y)
            output shape: (batch, time_frame, (x_filter_size * y_filter_size), num of final output filters)
        '''
        # TODO: need to modify the shapes for 3d
        # print(input_seq.shape)
        time_steps = input_seq.shape[1]
        print(time_steps)
        input_seq = input_seq.view(-1,
                                   input_seq.size(2),
                                   input_seq.size(3),
                                   input_seq.size(4))  # (batch * time_frame), D, Z , H, W

        # print('view', input_seq.shape)

        # will have shape (batch * time_frame), final output filters, x_filter_size, y_filter_size
        H = input_seq
        for layer in range(0, self.layer_num - 1):
            H = getattr(self, 'cnn3D_' + str(layer))(H)
            H = getattr(self, 'batchNorm3D_' + str(layer))(H)
            H = getattr(self, 'maxPool3D_' + str(layer))(H)
            H = getattr(self, 'dropOut3D_' + str(layer))(H)

        # print('conved', H.shape)
        H = self.activation(H)

        # print('activated', C3_seq.shape)
        # showTensor(H[0, 0, ...])
        # (batch * time_frame), x_filter_size, y_filter_size, final output filters
        H = H.permute(0, 2, 3, 1)

        # print('permutes', H.shape)
        # batch, time_frame, (x_filter_size * y_filter_size), final output filters
        H = H.reshape(-1,
                      self.params['T'],
                      self.params['dim_C2_1'],
                      self.params['dim_C2_2'])
        # print('reshaped/', C2_seq.shape)
        return H
