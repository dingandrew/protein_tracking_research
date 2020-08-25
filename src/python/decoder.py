import torch.nn as nn
import torch
import numpy as np
import time
from tqdm import tqdm
from util import open_model_json, showTensor


class Decoder(nn.Module):
    '''
        Extract features from a frame. These features are to be used
        in predicting the location of a cluster in the next frame.
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
        super(Decoder, self).__init__()
        self.params = params.copy()
        self.cnnParams = self.params['decode-cnn']
        self.layer_num = len(self.cnnParams['conv_features'])

        # create a feature extractor deep model
        for layer in range(0, self.layer_num - 1):
            setattr(self,
                    'cnn3D_' + str(layer),
                    nn.ConvTranspose3d(in_channels=self.cnnParams['conv_features'][layer],
                              out_channels=self.cnnParams['conv_features'][layer+1],
                              stride=self.cnnParams['conv_strides'][layer],
                              kernel_size=self.cnnParams['conv_kernels'][layer],
                              )
                    )
            setattr(self,
                    'batchNorm3D_' + str(layer),
                    nn.BatchNorm3d(
                        num_features=self.cnnParams['conv_features'][layer+1])
                    )

        # init cnn weights to zero
        # for layer in range(0, self.layer_num - 1):
        #     cnn = getattr(self, 'cnn3D_' + str(layer))
        #     nn.init.zeros_(cnn.weight.data)
        #     nn.init.zeros_(cnn.bias.data)

        self.activation = nn.ReLU6()


    def forward(self, input_seq):
        '''
            input shape: (batch, features, z, x, y)
            output shape: (batch, depth=1, z, x, y)
        '''

        # tqdm.write('DECODE view: {}'.format(input_seq.shape))

        # will have shape (batch * time_frame), final output filters, x_filter_size, y_filter_size
        H = input_seq
        for layer in range(0, self.layer_num - 1):
            H = getattr(self, 'cnn3D_' + str(layer))(H)
            # tqdm.write('\tDECODE conved {}: {}'.format(str(layer), H.shape))
            H = getattr(self, 'batchNorm3D_' + str(layer))(H)
            # tqdm.write('\tDECODE normed {}: {}'.format(str(layer), H.shape))
            H = self.activation(H)

        # H = H.reshape(2160)

        # tqdm.write('DECODE Out: {}'.format(H.shape))
        return H


if __name__ == "__main__":
    model_config = open_model_json('./model_config.json')
    model = FeatureExtractor(model_config['default'])
    print(model)
    param_num = sum([param.data.numel()
                     for param in model.parameters()])
    print('Parameter number: %.3f M' % (param_num / 1024 / 1024))
