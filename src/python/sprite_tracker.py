import argparse
from os import path
import numpy as np
import torch
import torch.nn as nn
import pickle
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
from math import sqrt

from util import open_model_json, save_as_json, showTensor
from protein_tracker import Tracker
from detector_2D import Detector
from track import Track

# Parse arguments
parser = argparse.ArgumentParser(
    description='Edit model hyper-parameters in model_config.json')
# Task
parser.add_argument('--model_task', default='default',
                    help="Choose a model with different hyper-parameters (specified in 'modules/model_config.json')")
parser.add_argument('--task', required=True, choices=['train', 'predict'],
                    help="Choose to train or predict with the model")

args = parser.parse_args()


class SpriteTracker(Tracker):
    '''
        Wrapper class to train/use the tracker
    '''

    def __init__(self, args, params):
        super().__init__(args, params)
        self.args = args
        self.params = params
        # init network and optimizer
        self.detector = Detector(self.params).cuda()
        # track the current example
        self.currSearchFrame = 0
        self.trainExample = 0
        ######################## need to call load_data
        self.full_data = FramesData(root_dir="../../data/raw_data/Segmentation_and_result",
                                    params=model_config[args.model_task],
                                    transform=transforms.Compose(
                                        [ToTensorFrame()])
                                    )
        self.tracks = None
        self.counts = None
        ########################
        
    def getMask(self, curr_track):
        '''
            Create initial input weights
        '''
        mask = torch.zeros((128, 128))
        for index in curr_track.locs:
            mask[index[0], index[1]] = 1
        mask = mask.reshape((1, 1, 1, mask.size(0), mask.size(1)))
        centroid = torch.zeros(2)
        centroid[0] = curr_track.centroid[0]
        centroid[1] = curr_track.centroid[1]
        return mask, centroid


    def crop_frame(self, tensor, center, width, height):
        '''
            keep z stack but crop x and y
        '''
        x_max = int(center[0] + width if center[0] + width < 128 else 128)
        x_min = int(center[0] - width if center[0] - width > 0 else 0)
        y_max = int(center[1] + height if center[1] + height < 128 else 128)
        y_min = int(center[1] - height if center[1] - height > 0 else 0)
        # print(center, x_max, x_min, y_max, y_min)
        return tensor[:, :, :, x_min:x_max, y_min:y_max]

    def calc_euclidean_dist(self, p1, p2):
        '''
            Calculate the weighted euclidean distace between 2 points in 2D
            Return: float distance
        '''
        return sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

if __name__ == "__main__":
    print('-------------- Train the Deep Tracker ------------------')
    # load model params
    model_config = open_model_json('./model_config.json')

    torch.cuda.set_device(0)

    # init the training wrapper
    sprites_tracker = SpriteTracker(args, model_config['sprites'])
    sprites_tracker.load_data(track='../../data/sprites-MOT/sprite/pt/tracks_sprite.pickle',
                              count='../../data/sprites-MOT/sprite/pt/counts.json')



