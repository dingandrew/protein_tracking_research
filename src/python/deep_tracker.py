import argparse
from os import path
import time
from tqdm import tqdm
import pickle
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt

import subprocess
from joblib import Parallel, delayed
import multiprocessing

from util import open_model_json, save_as_json, calc_euclidean_dist
from detector import Detector
from track import Track, Status

# Parse arguments
parser = argparse.ArgumentParser(
    description='Edit model hyper-parameters in model_config.json')
# Task
parser.add_argument('--model_task', default='default',
                    help="Choose a model with different hyper-parameters (specified in 'modules/model_config.json')")
parser.add_argument('--task', required=True, choices=['train', 'predict', 'optim_param'],
                    help="Choose to train or predict with the model")
parser.add_argument('--init_model', default='',
                    help="Recover training from a checkpoint, e.g., 'latest.pt', '3000.pt'")
args = parser.parse_args()


class ToTensor(object):
    '''
        Convert ndarrays of (280 H, 512 W, 13 Z, 64 filter_num) 
        to Tensors (64 filter_num, 13 Z, 280 H, 512 W).
    '''

    def __call__(self, weight_arr):
        # convert and swap axis
        # numpy tensor: H x W x Z x C
        # torch tensor: C X Z x H X W
        weight_tensor = torch.from_numpy(weight_arr)
        weight_tensor = weight_tensor.reshape(weight_tensor.size(3),
                                              1,
                                              weight_tensor.size(2),
                                              weight_tensor.size(0),
                                              weight_tensor.size(1)
                                              )
        return weight_tensor.float()


class WeightsLoader(Dataset):
    '''
        Dataloader to load the frame weights
    '''

    def __init__(self, root_dir, params, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.weight_file = 'Weights_{}.npy'
        self.params = params

    def __len__(self):
        return self.params['T']

    def __getitem__(self, frame_num):
        if not isinstance(frame_num, int):
            Exception('Frame Number must be a Integer')

        file_path = path.join(self.root_dir,
                              str(frame_num),
                              self.weight_file.format(frame_num)
                              )

        weights = np.load(file_path)
        if self.transform:
            weights = self.transform(weights)
        return weights


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
        self.currSearchFrame = 1
        self.trainExample = 0

        #### need to call load_data ###
        self.full_data = None
        self.tracks = None
        self.counts = None
        ###############################

        # init the weights dataset loader
        self.weights_data = WeightsLoader(root_dir="../../data/raw_data/Segmentation_and_result",
                                          params=model_config[args.model_task],
                                          transform=transforms.Compose(
                                              [ToTensor()])
                                          )

    def load_data(self, track, full, count):
        '''
            Load numpy data of unlabeled segemented data and tracks

            Return: numpy array
        '''
        print('Load Files: {} \n\t{} \n\t{}'.format(track, full, count))
        # load the tracks
        with open(track, 'rb') as f:
            # The protocol version used is detected automatically, so we do not
            # have to specify it.
            self.tracks = pickle.load(f)

        # load the cluster counts per frame
        self.counts = open_model_json(count)

        # load the frames
        data = torch.from_numpy(np.load(full))
        data = data.permute(3, 2, 0, 1)
        data = data[None, :, None, :, :, :]
        #showTensor(full_data[0, 0, 0, 5, ...])
        # print('Shape of raw full sized data: ', data.shape, type(data))
        self.full_data = data

    def getMask(self, curr_track):
        '''
            Create partial cluster masks with each partial cluster
            containing 'piece_size' points of the whole cluster
        '''
        # print(len(curr_track.locs))
        num_parts = math.ceil(len(curr_track.locs) / self.params['piece_size'])
        mask = torch.zeros((num_parts,
                            1,
                            self.params['Z'],
                            self.params['W'],
                            self.params['H'])
                           )
        # create partial masks
        start = 0
        end = 0
        parts_count = 0

        while parts_count < num_parts:
            end += self.params['piece_size']
            if end > (len(curr_track.locs)):
                end = len(curr_track.locs)

            partial_locs = curr_track.locs[start:end]
            for index in partial_locs:
                mask[parts_count, :, index[2], index[0], index[1]] = 1

            start = end
            parts_count += 1

        centroid = torch.zeros(3)
        centroid[0] = curr_track.centroid[2]
        centroid[1] = curr_track.centroid[0]
        centroid[2] = curr_track.centroid[1]

        return mask, centroid

    def crop_frame(self, tensor, center, width, height):
        '''
            keep z stack but crop x and y
        '''
        x_max = int(center[1] + width if center[1] + width < 280 else 280)
        x_min = int(center[1] - width if center[1] - width > 0 else 0)
        y_max = int(center[2] + height if center[2] + height < 512 else 512)
        y_min = int(center[2] - height if center[2] - height > 0 else 0)

        return tensor[:, :, :, x_min:x_max, y_min:y_max]

    def find_min_crop(self, mask, label):
        '''
            Find the minimum width and height that we can crop the frame by.

            Params: mask - shape of (batch, depth, Z, X, Y)
                    label - coordinates of centroid [z_coord, x_coord, y_coord]
            Return: width, height
        '''
        locs = torch.nonzero(mask)
        x_max = max(locs[:, 3])
        x_min = min(locs[:, 3])
        y_min = min(locs[:, 4])
        y_max = max(locs[:, 4])
        x_range = max(abs(x_max - label[1]), abs(x_min - label[1]))
        y_range = max(abs(y_max - label[2]), abs(y_min - label[2]))

        width = math.ceil(x_range + self.params['crop_window'])
        height = math.ceil(y_range + self.params['crop_window'])

        return width, height

    def predictor_features(self, epoch_id):
        # get the clusters in this frame
        frame_tracks = self.tracks[self.currSearchFrame]
        currTrack = frame_tracks[self.trainExample]
        mask, label = self.getMask(currTrack)

        frame1 = self.full_data[0, self.currSearchFrame - 1, 0, ...]
        frame2 = self.full_data[0, self.currSearchFrame, 0, ...]
        frame1 = frame1.reshape(
            (1, 1, frame1.size(0), frame1.size(1), frame1.size(2)))
        frame2 = frame2.reshape(
            (1, 1, frame2.size(0), frame2.size(1), frame2.size(2)))

        width, height = self.find_min_crop(mask, label)
        frame1_crop = self.crop_frame(frame1, label, width, height)
        frame2_crop = self.crop_frame(frame2, label, width, height)
        mask_crop = self.crop_frame(mask, label, width, height)

        f1_feature, f2_feature = self.detector(frame1_crop.cuda().float(),
                                               frame2_crop.cuda().float(),
                                               mask_crop.cuda().float())

        tqdm.write('Predictor Epoch: {}, Cluster Num: {}, Frame Num: {}'.format(
            epoch_id, self.trainExample, self.currSearchFrame))

        return f1_feature, f2_feature

    def get_predictor_results(self, frame_num, forward_scan=True, backward_scan=True):
        '''
            frame_num: the track index [1, 70]
            [forward, backward]_state: each index points to list of 
                                       clusters that it may match too, if any

        '''
        # record the relation of each cluster [{min_indx:0.8}, ...]
        forward_state = {}
        backward_state = {}
        forward_rates = {}
        backward_rates = {}

        # prefetch current frame as it is constant for each case
        frameCurr = self.full_data[0, frame_num - 1, 0, ...]
        frameCurr = frameCurr.reshape(
            (1, 1, frameCurr.size(0), frameCurr.size(1), frameCurr.size(2)))

        if forward_scan:
            # get the clusters in this frame
            frame_tracks = self.tracks[frame_num]
            # scan the next frame
            frameNext = self.full_data[0, frame_num, 0, ...]
            frameNext = frameNext.reshape(
                (1, 1, frameNext.size(0), frameNext.size(1), frameNext.size(2)))

            for indx, currTrack in enumerate(frame_tracks):
                mask, label = self.getMask(currTrack)
                width, height = self.find_min_crop(mask, label)
                frame1_crop = self.crop_frame(frameCurr, label, width, height)
                frame2_crop = self.crop_frame(frameNext, label, width, height)
                mask_crop = self.crop_frame(mask, label, width, height)
                # print(frame1_crop.shape, frame2_crop.shape, mask_crop.shape)
                _, f2_feature = self.detector(frame1_crop.cuda().float(),
                                              frame2_crop.cuda().float(),
                                              mask_crop.cuda().float(),
                                              train=True)

                partial_predictions = self.detector.predict(
                    f2_feature.cpu().numpy())
                pos_pred = 0
                for label in partial_predictions:
                    if label == trainer.params['pos_lbl']:
                        pos_pred += 1
                forward_pos_rate = pos_pred / len(partial_predictions)

                if forward_pos_rate <= 0.06:

                    forward_state[currTrack] = None
                    forward_rates[currTrack] = forward_pos_rate
                    tqdm.write('FORWARD - Cluster Num: {}, Frame Num: {}, Match: NONE'.format(
                        indx, frame_num))
                else:
                    # has potential match min closest cluster and append with rate
                    minDist = math.inf
                    minTrack = None
                    for nextTrack in self.tracks[frame_num + 1]:
                        if calc_euclidean_dist(currTrack.centroid, nextTrack.centroid, [1, 1, 1]) < minDist:
                            minTrack = nextTrack
                            minDist = calc_euclidean_dist(
                                currTrack.centroid, nextTrack.centroid, [1, 1, 1])

                    forward_state[currTrack] = minTrack
                    forward_rates[currTrack] = forward_pos_rate
                    tqdm.write('FORWARD - Cluster Num: {}, Frame Num: {}, Match: {}'.format(
                        indx, frame_num, forward_pos_rate))

        if backward_scan:
            # get the clusters in this frame
            frame_tracks = self.tracks[frame_num]
            # scan the prev frame
            framePrev = self.full_data[0, frame_num - 2, 0, ...]
            framePrev = framePrev.reshape(
                (1, 1, framePrev.size(0), framePrev.size(1), framePrev.size(2)))

            for indx, currTrack in enumerate(frame_tracks):
                mask, label = self.getMask(currTrack)
                width, height = self.find_min_crop(mask, label)
                frame1_crop = self.crop_frame(frameCurr, label, width, height)
                frame2_crop = self.crop_frame(framePrev, label, width, height)
                mask_crop = self.crop_frame(mask, label, width, height)
                # print(frame1_crop.shape, frame2_crop.shape, mask_crop.shape)
                _, f2_feature = self.detector(frame1_crop.cuda().float(),
                                              frame2_crop.cuda().float(),
                                              mask_crop.cuda().float(),
                                              train=True)

                partial_predictions = self.detector.predict(
                    f2_feature.cpu().numpy())
                pos_pred = 0
                for label in partial_predictions:
                    if label == trainer.params['pos_lbl']:
                        pos_pred += 1
                backward_pos_rate = pos_pred / len(partial_predictions)

                if backward_pos_rate <= 0.06:

                    backward_state[currTrack] = None
                    backward_rates[currTrack] = backward_pos_rate
                    tqdm.write('BACKWARD - Cluster Num: {}, Frame Num: {}, Match: NONE'.format(
                        indx, frame_num))
                else:
                    # has potential match min closest cluster and append with rate
                    minDist = math.inf
                    minTrack = None
                    for nextTrack in self.tracks[frame_num - 1]:
                        if calc_euclidean_dist(currTrack.centroid, nextTrack.centroid, [1, 1, 1]) < minDist:
                            minTrack = nextTrack
                            minDist = calc_euclidean_dist(
                                currTrack.centroid, nextTrack.centroid, [1, 1, 1])

                    backward_state[currTrack] = minTrack
                    backward_rates[currTrack] = backward_pos_rate

                    tqdm.write('BACKWARD - Cluster Num: {}, Frame Num: {}, Match: {}'.format(
                        indx, frame_num, backward_pos_rate))

        return forward_state, forward_rates, backward_state, backward_rates


if __name__ == "__main__":
    print('-------------- Initializing the Protein Tracker ------------------')
    # load model params
    model_config = open_model_json('./model_config.json')

    # init the training wrapper
    trainer = Trainer(args, model_config[args.model_task])
    trainer.load_data(track='../../data/tracks_protein.pickle',
                      full='../../data/raw3data.npy',
                      count='../../data/counts.json')

    # use the GPU if there is one
    gpu_num = torch.cuda.device_count()
    print('GPU NUM: {:2d}'.format(gpu_num))
    if gpu_num > 1:
        torch.cuda.set_device(0)
        trainer.detector = torch.nn.DataParallel(
            trainer.detector, list(range(gpu_num))).cuda()

    if args.task == "optim_param":
        print('-------------- Best Parameters for Model ------------------')
        with open('../../data/f1_pos_rate_list.pickle', 'rb') as f:
            f1_pos_rate_list = pickle.load(f)
        with open('../../data/f2_pos_rate_list.pickle', 'rb') as f:
            f2_pos_rate_list = pickle.load(f)

        # histogram of positive detection rates of partial clusters
        fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)
        axs[0].hist(f1_pos_rate_list, bins=100)
        axs[0].set_title('Cluster on Own Frame')
        axs[1].hist(f2_pos_rate_list, bins=100)
        axs[1].set_title('Cluster on Next Frame')
        plt.show()

        # histogram of whole cluster sizes
        if not path.exists('../../data/cluster_size_list.pickle'):
            cluster_size_list = []
            for f in range(1, 71):
                for currTrack in trainer.tracks[f]:
                    cluster_size_list.append(len(currTrack.locs))

            with open('../../data/cluster_size_list.pickle', 'wb') as f:
                # Pickle the 'data' dictionary using the highest protocol available.
                pickle.dump(cluster_size_list, f, pickle.HIGHEST_PROTOCOL)
        else:
            with open('../../data/cluster_size_list.pickle', 'rb') as f:
                # Pickle the 'data' dictionary using the highest protocol available.
                cluster_size_list = pickle.load(f)

        plt.hist(cluster_size_list, bins=500)
        plt.title('Whole Cluster Sizes')
        plt.xlabel('Cluster Sizes')
        plt.xticks(np.arange(0, 300, 5))
        plt.xlim(left=0, right=300)
        plt.ylabel('Number of Clusters')
        plt.show()

    elif args.task == "train":
        print('-------------- Train the Protein Tracker ------------------')
        f1 = np.zeros((1, trainer.params['embedding_len']))
        f2 = np.zeros((1, trainer.params['embedding_len']))
        for epoch_id in tqdm(range(0, trainer.params['detect_num'])):
            # dynamically calculate the number of training examples
            trainNum = trainer.counts[str(trainer.currSearchFrame)]
            f1_feature, f2_feature = trainer.predictor_features(epoch_id)
            trainer.trainExample += 1
            f1 = np.append(f1, f1_feature.cpu().numpy(), axis=0)
            f2 = np.append(f2, f2_feature.cpu().numpy(), axis=0)

            if trainer.trainExample == trainNum:
                trainer.currSearchFrame += 1
                trainer.trainExample = 0
            # reset the current search frame if all clusters have been searched
            if trainer.currSearchFrame == 70:
                trainer.currSearchFrame = 0
                trainer.trainExample = 0

        with open('../../data/f1.npy', 'wb') as f:
            np.save(f, f1)
        with open('../../data/f2.npy', 'wb') as f:
            np.save(f, f2)

        print('-------------- Fit the Protein Tracker ------------------')
        with open('../../data/f1.npy', 'rb') as f:
            f1 = np.load(f)
        with open('../../data/f2.npy', 'rb') as f:
            f2 = np.load(f)

        trainer.detector.train_feat(f1[1:, ...], f2[1:, ...])

    elif args.task == 'predict':
        ''' 
            Will run the tracking algo on the entire dataset

            1. iterate through all frames
                a. for all clusters in each frame get the predictor_features
                b. for each cluster run prediction using the predictor_features
                c. if the cluster has a positive prediction in the next frame,
                   find the minimum distance cluster from it and store this relation
                   in a dict
            2. repeat step1 for the current frame and the next frame, be sure
               to use the predictor features from the next frame to reduct computation for the
               next set of frames
            3. state lists are filled then we can determine clusters relations by forming a graph


        '''
        print('-------------- Run Tracker Prediction ------------------')
        # check detector has already been trained
        if not path.exists('../../models/detector.pickle'):
            print('ERROR: You must train the detector first')
            print('\tpython3 deep_tracker.py --task train --type detect')
            exit()

        # store the posrates for graphing
        f1_pos_rate_list = []
        f2_pos_rate_list = []

        # store some statistics
        DEATH_COUNT = 0
        ONE_TO_ONE_COUNT = 0
        SPLIT_COUNT = 0
        MERGE_COUNT = 0
        BIRTH_COUNT = trainer.counts['1']  # count initial frame

        # will hold each track in a frame and what it points too
        next_forward_state = []
        forward_state = []
        backward_state = []

        # retreive the largest id from first frame and increment this will be the next id
        nextID = max(track.id for track in trainer.tracks[1]) + 1
        # iterate through all frames , note we are checking next frame
        for currFrame in tqdm(range(1, 70)):
            # check all tracks in this frame
            # TODO can this be multi-threaded
            if currFrame == 1:
                forward_state, forward_rates, _,  _ = trainer.get_predictor_results(currFrame,
                                                                 backward_scan=False)
            else:
                forward_state = next_forward_state
                forward_rates = next_forward_rates

            if currFrame == 69:
                _, _, backward_state, backward_rates = trainer.get_predictor_results(currFrame + 1,
                                                                  forward_scan=False)
            else:
                next_forward_state, next_forward_rates, backward_state, backward_rates = trainer.get_predictor_results(
                    currFrame + 1)

            tqdm.write('\tCheck Relation States')
            # NOTE: for now ignore the pos_rate
            for forward_key in forward_state.keys():
                # how many objects in next frame point to this object

                forward_match_count = list(
                    backward_state.values()).count(forward_key)

                backward_key = forward_state[forward_key]
                if backward_key:
                    # how many objects in previous frame point to this object
                    backward_match_count = list(
                        forward_state.values()).count(backward_key)
                else:
                    backward_match_count = 0

                # find the track asscoiated with forward_key
                track_indx = trainer.tracks[currFrame].index(forward_key)
                currTrack = trainer.tracks[currFrame][track_indx]

                if forward_match_count == 0 or backward_match_count == 0:
                    # object dead no match in next frame
                    tqdm.write('\t DEATH forward_match_count: {} backward_match_count; {}'
                               .format(forward_match_count, backward_match_count))
                    currTrack.state = Status.DEAD
                    DEATH_COUNT += 1
                    continue

                if forward_match_count == 1 and backward_match_count == 1:
                    # object has 1 to 1 match with next frame
                    tqdm.write('\t MATCH forward_match_count: {} backward_match_count; {}'
                               .format(forward_match_count, backward_match_count))
                    if forward_rates[forward_key] >= trainer.params['pos_threshhold']\
                        and backward_rates[backward_key] >= trainer.params['pos_threshhold']:
                        # find the track asscoiated with backward_key
                        track_indx = trainer.tracks[currFrame + 1].index(backward_key)
                        track = trainer.tracks[currFrame + 1][track_indx]
                        track.id = currTrack.id
                        track.state = currTrack.state
                        track.origin += currTrack.origin + \
                            Status.MATCH.format(currFrame)

                        ONE_TO_ONE_COUNT += 1

                # if forward_match_count > 1 and backward_match_count == 1:
                if forward_match_count > 1:
                    # object has split in next frame, 1 to many relation
                    tqdm.write('\t SPLIT forward_match_count: {} backward_match_count; {}'
                               .format(forward_match_count, backward_match_count))
                    for backward_keys in backward_state.keys():
                        if backward_state[backward_keys] == forward_key:
                            # find the track asscoiated with backward_key
                            track_indx = trainer.tracks[currFrame +
                                                        1].index(backward_keys)
                            track = trainer.tracks[currFrame + 1][track_indx]
                            track.id = nextID
                            nextID += 1
                            track.state = Status.ACTIVE
                            track.origin += Status.SPLIT.format(
                                currFrame, currTrack.id)

                    SPLIT_COUNT += 1

                # if forward_match_count == 1 and backward_match_count > 1:
                if backward_match_count > 1:
                    # object has merged with another object, many to 1 relation
                    tqdm.write('\t MERGE forward_match_count: {} backward_match_count; {}'
                               .format(forward_match_count, backward_match_count))
                    # find the track asscoiated with backward_key
                    track_indx = trainer.tracks[currFrame +
                                                1].index(backward_key)
                    track = trainer.tracks[currFrame + 1][track_indx]
                    track.id = nextID
                    nextID += 1
                    track.state = Status.ACTIVE

                    for forward_keys in forward_state.keys():
                        if forward_state[forward_keys] == backward_key:
                            track.origin += Status.MERGE.format(
                                currFrame, forward_keys.id)
                            # do this so we dont double count it
                            forward_state[forward_keys] = None

                    MERGE_COUNT += 1

            # remaining unlabled clusters in next frame are all the result of birth
            for back_key in backward_state.keys():
                # find the track asscoiated with back_key
                next_track_indx = trainer.tracks[currFrame + 1].index(back_key)
                nextTrack = trainer.tracks[currFrame + 1][next_track_indx]
                if nextTrack.id is None:
                    # this cluster is birthed in this frame
                    tqdm.write('\t BIRTH ')
                    nextTrack.id = nextID
                    nextID += 1
                    nextTrack.state = Status.ACTIVE
                    nextTrack.origin += Status.BIRTH.format(currFrame + 1)
                    BIRTH_COUNT += 1

        with open('../../data/labeled_tracks.pickle', 'wb') as f:
            # Pickle the 'data' dictionary using the highest protocol available.
            pickle.dump(trainer.tracks, f, pickle.HIGHEST_PROTOCOL)

        print('-------------- Tracking Statistics ------------------')
        print('Deaths: ', DEATH_COUNT)
        print('Births: ', BIRTH_COUNT)
        print('One to Ones: ', ONE_TO_ONE_COUNT)
        print('Splits: ', SPLIT_COUNT)
        print('Merges: ', MERGE_COUNT)

        print('-------------- Saving Labeled Tracks ------------------')
        save_as_json(trainer.tracks, 0, 0, 0)
        print('----------------------- Done --------------------------')
