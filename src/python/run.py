import time
import os
import os.path as path
import argparse
import numpy as np
import math
import torch
from util import open_model_json, showTensor
from feature_extractor import FeatureExtractor

# Parse arguments
parser = argparse.ArgumentParser()
# Task
parser.add_argument('--type', choices=['tracker', 'deep'],
                    help="Choose a method to track with")

parser.add_argument('--model', default='default',
                    help="Choose a model with different hyper-parameters (specified in 'modules/model_config.json')")

parser.add_argument('--train', type=int, default=1, choices=[0, 1],
                    help="Choose to train (1) or test (0) the model")

parser.add_argument('--init_model', default='',
                    help="Recover training from a checkpoint, e.g., 'sp_latest.pt', 'sp_3000.pt'")
# Model settings
parser.add_argument('--r', type=int, default=1, choices=[0, 1],
                    help="Choose whether to remember the recurrent state from the previous sequence")
# Training
parser.add_argument('--epoch_num', type=int, default=500,
                    help="The number of training epoches")
parser.add_argument('--reset_interval', type=float, default=0.01,
                    help="Set how to reset the recurrent state, \
                    (-inf, 0): do not reset, [0, 1): the probability to reset, [1, inf): time steps to reset")
parser.add_argument('--print_interval', type=int, default=1,
                    help="Iterations to print training messages")
parser.add_argument('--train_log_interval', type=int, default=100,
                    help="Iterations to log training messages")
parser.add_argument('--save_interval', type=int, default=100,
                    help="Iterations to save checkpoints (will be overwitten)")
parser.add_argument('--validate_interval', type=int, default=1000,
                    help="Iterations to validate model and save checkpoints")
# Optimization
parser.add_argument('--lr', type=float, default=5e-4,
                    help="Learning rate")
parser.add_argument('--lr_decay_factor', type=float, default=1,
                    help="Learning rate decay factor")
parser.add_argument('--grad_clip', type=float, default=5,
                    help='Gradient clipping value')

args = parser.parse_args()






if __name__ == "__main__":
    #load model params
    model_config = open_model_json('./model_config.json')
    for k, v in model_config.items():
        vars(args)[k] = v
    
    print(args)
    
    #init fe
    fe = FeatureExtractor(args)

    print(fe)



    data_dir = '/home/andrew/unsupervised_samples/tracking-by-animation/data/sprite/pt'
    split = 'train'
    filename = split + '_' + str(1) + '.pt'
    X_seq = torch.load(path.join(data_dir, 'input', filename))
    X_seq = X_seq[:5, :, :, :, :]


    X_seq = X_seq.float().div_(255)
    print('start')
    a = fe(X_seq)

    # Show the tensor.

    # showTensor(a[0,0,...])
