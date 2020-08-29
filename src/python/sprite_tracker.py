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
from network import Network
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


class Trainer():
    '''
        Wrapper class to train the deep tracker
    '''

    def __init__(self, args, params):
        self.args = args
        self.params = params
        # init network and optimizer
        self.detector = Detector(self.params).cuda()
        # track the current example
        self.currSearchFrame = 0
        self.trainExample = 0
        ######################## need to call load_data
        self.full_data = None
        self.tracks = None
        self.counts = None
        ########################
        

    def load_data(self, track, full_data, count):
        '''
            Load numpy data of unlabeled segemented data and tracks

            Return: numpy array
        '''
        # load the tracks
        with open(track, 'rb') as f:
            # The protocol version used is detected automatically, so we do not
            # have to specify it.
            self.tracks = pickle.load(f)
        # load the cluster counts per frame
        self.counts = open_model_json(count)
        # load the frames
        self.full_data = torch.from_numpy(np.load(full_data))


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


    def predictor_features(self, epoch_id):
        # get the clusters in this frame
        frame_tracks = self.tracks[self.currSearchFrame]
        currTrack = frame_tracks[self.trainExample]
        mask, label = self.getMask(currTrack)
        frame1 = self.full_data[0, self.currSearchFrame, 0, ...]
        frame2 = self.full_data[0, self.currSearchFrame + 1, 0, ...]
        frame1 = frame1.reshape(
            (1, 1, 1, frame1.size(0), frame1.size(1)))
        frame2 = frame2.reshape(
            (1, 1, 1, frame2.size(0), frame2.size(1)))
        # print(frame1.shape, frame2.shape)

        frame1_crop = self.crop_frame(frame1, label, 20, 20)
        frame2_crop = self.crop_frame(frame2, label, 20, 20)
        mask_crop = self.crop_frame(mask, label, 20, 20)

        f1_feature, f2_feature = self.detector(frame1_crop.cuda().float(),
                                               frame2_crop.cuda().float(),
                                               mask_crop.cuda().float())

        tqdm.write('Predictor Epoch: {}, batch: {}'.format(epoch_id,
                                                           self.trainExample))

        return f1_feature, f2_feature

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

    def calc_euclidean_dist(self, p1, p2, weights):
        '''
            Calculate the weighted euclidean distace between 2 points in 2D
            Return: float distance
        '''
        return sqrt(weights[0] * (p1[0] - p2[0])**2 + weights[1] * (p1[1] - p2[1])**2)

if __name__ == "__main__":
    print('-------------- Train the Deep Tracker ------------------')
    # load model params
    model_config = open_model_json('./model_config.json')

    torch.cuda.set_device(0)

    # init the training wrapper
    trainer = Trainer(args, model_config['sprites'])
    trainer.load_data(track='../../data/sprites-MOT/sprite/pt/tracks_sprite.pickle',
                      full_data='../../data/sprites-MOT/sprite/pt/full_train.npy',
                      count='../../data/sprites-MOT/sprite/pt/counts.json')

    # Run the trainer
    if args.task == "train":
        f1 = []
        f2 = []
        for epoch_id in tqdm(range(0, trainer.params['detect_num'])):
            # dynamically calculate the number of training examples
            trainNum = trainer.counts[str(trainer.currSearchFrame)]
            
            if trainNum == 0:
                # this frame has no objects
                trainer.currSearchFrame += 1
            else:
                f1_feature, f2_feature = trainer.predictor_features(epoch_id)
                trainer.trainExample += 1
                f1.append(f1_feature.cpu().numpy())
                f2.append(f2_feature.cpu().numpy())

                if trainer.trainExample == trainNum:
                    trainer.currSearchFrame += 1
                    trainer.trainExample = 0

                # reset the current search frame if all clusters have been searched
                if trainer.currSearchFrame == trainer.params['T']:
                    trainer.currSearchFrame = 0
                    trainer.trainExample = 0
        
        f1 = np.asarray(f1)
        f2 = np.asarray(f2)

        with open('../../data/sprites-MOT/sprite/pt/f1.npy', 'wb') as f:
            np.save(f, f1)
        with open('../../data/sprites-MOT/sprite/pt/f2.npy', 'wb') as f:
            np.save(f, f2)

        with open('../../data/sprites-MOT/sprite/pt/f1.npy', 'rb') as f:
            f1 = np.load(f)
        with open('../../data/sprites-MOT/sprite/pt/f2.npy', 'rb') as f:
            f2 = np.load(f)


        trainer.detector.train_feat(f1, f2)

    elif args.task == 'predict':
        ''' 
            Will run the tracking algo on the entire dataset
        '''
        # retreive the largest id from first frame and increment this will be the next id
        if len(trainer.tracks[0]) == 0:
            nextID = 1
        else:
            nextID = max(track.id for track in trainer.tracks[0]) + 1
        # iterate through all frames , note we are checking next frame
        for currFrame in tqdm(range(0, trainer.params['T'] - 1)):
            # check all tracks in this frame
            for currTrack in trainer.tracks[currFrame]:
                # get current track info
                currId = currTrack.id
                currCentroid = currTrack.centroid
                currState = currTrack.state
                currOrigin = currTrack.origin
                currLocs = currTrack.locs
                # get the clusters in this frame
                mask, label = trainer.getMask(currTrack)
                frame1 = trainer.full_data[0, currFrame, 0, ...]
                frame2 = trainer.full_data[0,currFrame + 1, 0, ...]
                frame1 = frame1.reshape((1, 1, 1, frame1.size(0), frame1.size(1)))
                frame2 = frame2.reshape((1, 1, 1, frame2.size(0), frame2.size(1)))
                # print(frame1.shape, frame2.shape)

                frame1_crop = trainer.crop_frame(frame1, label, 20, 20)
                frame2_crop = trainer.crop_frame(frame2, label, 20, 20)
                mask_crop = trainer.crop_frame(mask, label, 20, 20)

                f1_feature, f2_feature = trainer.detector(frame1_crop.cuda().float(),
                                                            frame2_crop.cuda().float(),
                                                            mask_crop.cuda().float())
                # print(f1_feature, f2_feature)
                # exit()
                if trainer.detector.predict(f2_feature.cpu().numpy().reshape(1, -5)) == 0:
                    # this is tooo slow
                    # TODO: refactor min centroid finder algo
                    minDist = 999999
                    minTrack = None
                    for nextTrack in trainer.tracks[currFrame + 1]:

                        if nextTrack.id is None:

                            if trainer.calc_euclidean_dist(currCentroid, nextTrack.centroid, [1, 1]) < minDist:
                                minTrack = nextTrack
                                minDist = trainer.calc_euclidean_dist(currCentroid, nextTrack.centroid, [1, 1])
                        elif nextTrack.id == currId:
                            print("should never happen")

                    if minTrack:
                        # run the cluster backwards in time to ensure tracking
                        mask, label = trainer.getMask(minTrack)
                        frame1_crop = trainer.crop_frame(frame1, label, 50, 50)
                        frame2_crop = trainer.crop_frame(frame2, label, 50, 50)
                        mask_crop = trainer.crop_frame(mask, label, 50, 50)

                        f1_feature, f2_feature = trainer.detector(frame2_crop.cuda().float(),
                                                                    frame1_crop.cuda().float(),
                                                                    mask_crop.cuda().float(),
                                                                    train=False)
                        # TODO: also find min centroid in the reverse
                        if trainer.detector.predict(f2_feature.cpu().numpy().reshape(1, -5)) == 0:
                            tqdm.write('match')
                            minTrack.id = currId
                            minTrack.state = currState
                            minTrack.origin = currOrigin
                        else:
                            tqdm.write('no reverse match')
                            currTrack.state = 'dead'
                else:
                    tqdm.write('no forward match')
                    currTrack.state = 'dead'

            # remaining unlabled clusters in next frame are all the result
            # of birth, split, or merge
            for nextTrack in trainer.tracks[currFrame + 1]:
                if nextTrack.id is None:
                    # this cluster is birthed in this frame
                    nextTrack.id = nextID
                    nextID += 1
                    nextTrack.state = 'active'
                    nextTrack.origin = 'birth'

        with open('../../data/sprites-MOT/sprite/pt/labeled_tracks_sprites.pickle', 'wb') as f:
            # Pickle the 'data' dictionary using the highest protocol available.
            pickle.dump(trainer.tracks, f, pickle.HIGHEST_PROTOCOL)

        save_as_json(trainer.tracks, 0, 0, 0)
