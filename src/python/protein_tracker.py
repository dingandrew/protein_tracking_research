import argparse
from os import path
import time
from tqdm import tqdm
import pickle
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
import matplotlib.pyplot as plt

from torch.multiprocessing import Process, Queue
# import multiprocessing
import torch.multiprocessing as multiprocessing

from util import load_json, load_pickle, save_pickle, save_tracks_as_json, calc_euclidean_dist
from detector_3D import Detector
from track import Track, Status
import info

# Parse arguments
parser = argparse.ArgumentParser(
    description='Edit model hyper-parameters in model_config.json')
# Task
parser.add_argument('--model_task', default='protein',
                    help="Choose a model with different hyper-parameters (specified in 'modules/model_config.json')")
parser.add_argument('--task', required=True, choices=['train', 'predict', 'optim_param'],
                    help="Choose to train or predict with the model")
parser.add_argument('--init_model', default='',
                    help="Recover training from a checkpoint, e.g., 'latest.pt', '3000.pt'")
args = parser.parse_args()


class ToTensorFrame(object):
    '''
        Convert ndarrays of (280 H, 512 W, 13 Z) 
        to Tensors (1 batch, 1 channel, 13 Z, 280 H, 512 W).
    '''

    def __call__(self, frame_arr):
        # convert and swap axis
        # numpy tensor: H x W x Z
        # torch tensor: B X C X Z x H X W
        frame_tensor = torch.from_numpy(frame_arr)
        # print(frame_tensor.shape)

        frame_tensor = frame_tensor.permute(2, 0, 1)
        frame_tensor = frame_tensor.reshape(1,
                                            1,
                                            frame_tensor.size(0),
                                            frame_tensor.size(1),
                                            frame_tensor.size(2)
                                            )

        return frame_tensor


class FramesData(Dataset):
    '''
        Dataloader to load the full frames
    '''

    def __init__(self, root_dir, params, transform):
        self.root_dir = root_dir
        self.frame_file = 'Fullsize_{}.npy'
        self.transform = transform
        self.params = params

    def __len__(self):
        return self.params['T']

    def __getitem__(self, frame_num):
        if not isinstance(frame_num, int):
            Exception('Frame Number must be a Integer')

        file_path = path.join(self.root_dir,
                              str(frame_num),
                              self.frame_file.format(frame_num)
                              )
        frame = np.load(file_path)

        if self.transform:
            frame = self.transform(frame)
        return frame
        # .cuda()


class Tracker():
    '''
        Class to train/predict the tracker
    '''

    def __init__(self, args, params):
        self.args = args
        self.params = params
        # init network and optimizer
        self.detector = Detector(self.params)
        # .cuda()

        # track the current example
        self.currSearchFrame = 1
        self.trainExample = 0
        self.nextID = 0

        #### need to call load_data ###
        self.tracks = None
        self.counts = None
        ###############################

        # init dataset loader
        self.full_data = FramesData(root_dir="../../data/raw_data/Segmentation_and_result",
                                    params=model_config[args.model_task],
                                    transform=transforms.Compose(
                                        [ToTensorFrame()]),
                                    )

        # store the posrates for graphing
        self.f1_pos_rate_list = []
        self.f2_pos_rate_list = []

        # store some statistics
        self.DEATH_COUNT = 0
        self.ONE_TO_ONE_COUNT = 0
        self.SPLIT_COUNT = 0
        self.MERGE_COUNT = 0
        self.BIRTH_COUNT = 0

    def load_data(self, track, count):
        '''
            Load numpy data of unlabeled segemented data and tracks

            Return: numpy array
        '''
        print('Load Files: {} \n\t{}'.format(track, count))
        self.tracks = load_pickle(track)
        self.counts = load_json(count)

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
        x_max = int(center[1] + width if center[1] +
                    width < self.params['W'] else self.params['W'])
        x_min = int(center[1] - width if center[1] - width > 0 else 0)
        y_max = int(center[2] + height if center[2] +
                    height < self.params['H'] else self.params['H'])
        y_min = int(center[2] - height if center[2] - height > 0 else 0)

        return tensor[:, :, :, x_min:x_max, y_min:y_max]

    def find_min_crop(self, mask, label):
        '''
            Find the minimum width and height that we can crop the frame by.

            Params: mask - shape of (batch, depth, Z, X, Y)
                    label - coordinates of centroid [z_coord, x_coord, y_coord]
            Return: width, height
        '''
        locs = torch.nonzero(mask, as_tuple=False)
        x_max = max(locs[:, 3])
        x_min = min(locs[:, 3])
        y_min = min(locs[:, 4])
        y_max = max(locs[:, 4])
        x_range = max(abs(x_max - label[1]), abs(x_min - label[1]))
        y_range = max(abs(y_max - label[2]), abs(y_min - label[2]))

        width = math.ceil(x_range + self.params['crop_window'])
        height = math.ceil(y_range + self.params['crop_window'])

        return width, height

    def gen_pos_rates(self):
        pass

    def predictor_features(self, epoch_id):
        # get the clusters in this frame
        frame_tracks = self.tracks[self.currSearchFrame]
        currTrack = frame_tracks[self.trainExample]
        mask, label = self.getMask(currTrack)
        frame1 = self.full_data[self.currSearchFrame]
        frame2 = self.full_data[self.currSearchFrame + 1]
        width, height = self.find_min_crop(mask, label)
        frame1_crop = self.crop_frame(frame1, label, width, height)
        frame2_crop = self.crop_frame(frame2, label, width, height)
        mask_crop = self.crop_frame(mask, label, width, height)

        f1_feature, f2_feature = self.detector(frame1_crop.float(),
                                               frame2_crop.float(),
                                               mask_crop.float())  # .cuda()

        tqdm.write('Predictor Epoch: {}, Cluster Num: {}, Frame Num: {}'.format(
            epoch_id, self.trainExample, self.currSearchFrame))

        return f1_feature, f2_feature

    def forward_scan(self, frame_num):
        forward_state = {}
        forward_rates = {}

        frameCurr = self.full_data[frame_num]
        # get the clusters in this frame
        frame_tracks = self.tracks[frame_num]
        # scan the next frame
        frameNext = self.full_data[frame_num + 1]

        for indx, currTrack in enumerate(frame_tracks):
            mask, label = self.getMask(currTrack)
            width, height = self.find_min_crop(mask, label)
            frame1_crop = self.crop_frame(frameCurr, label, width, height)
            frame2_crop = self.crop_frame(frameNext, label, width, height)
            mask_crop = self.crop_frame(mask, label, width, height)
            # print(frame1_crop.shape, frame2_crop.shape, mask_crop.shape) .cuda()
            _, f2_feature = self.detector(frame1_crop.float(),
                                          frame2_crop.float(),
                                          mask_crop.float(),
                                          train=True)

            partial_predictions = self.detector.predict(
                f2_feature.cpu().numpy())
            pos_pred = 0
            for label in partial_predictions:
                if label in self.params['pos_lbl']:
                    pos_pred += 1
            forward_pos_rate = pos_pred / len(partial_predictions)

            if forward_pos_rate <= 0.06:

                forward_state[currTrack] = None
                forward_rates[currTrack] = forward_pos_rate
                tqdm.write('FORWARD - Cluster Num: {}, Frame Num: {}, Match: NONE'.format(
                    indx, frame_num))
            else:
                # has potential match min closest cluster and append with rate
                minDist = 100
                # maxIntersect = 0
                minTrack = None
                # currLocs = currTrack.locs

                # find closest track with most intersetions
                for nextTrack in self.tracks[frame_num + 1]:
                    if calc_euclidean_dist(currTrack.centroid, nextTrack.centroid) < minDist:
                        # intersect = np.isin(currLocs,nextTrack.locs, assume_unique=True)
                        # intersect_num = np.count_nonzero(intersect.all(0).any())

                        # if intersect_num >= maxIntersect:
                        #     minDist = calc_euclidean_dist(
                        #         currTrack.centroid, nextTrack.centroid)
                        #     maxIntersect = intersect_num
                        #     minTrack = nextTrack
                        minTrack = nextTrack
                        minDist = calc_euclidean_dist(
                            currTrack.centroid, nextTrack.centroid)

                forward_state[currTrack] = minTrack
                forward_rates[currTrack] = forward_pos_rate
                tqdm.write('FORWARD - Cluster Num: {}, Frame Num: {}, Match: {}'.format(
                    indx, frame_num, forward_pos_rate))

        # return_dict['fs'] = forward_state
        # return_dict['fr'] = forward_rates
        return forward_state, forward_rates

    def backward_scan(self, frame_num):
        backward_state = {}
        backward_rates = {}

        frameCurr = self.full_data[frame_num]
        # get the clusters in this frame
        frame_tracks = self.tracks[frame_num]
        # scan the prev frame
        framePrev = self.full_data[frame_num - 1]

        for indx, currTrack in enumerate(frame_tracks):
            mask, label = self.getMask(currTrack)
            width, height = self.find_min_crop(mask, label)
            frame1_crop = self.crop_frame(frameCurr, label, width, height)
            frame2_crop = self.crop_frame(framePrev, label, width, height)
            mask_crop = self.crop_frame(mask, label, width, height)

            _, f2_feature = self.detector(frame1_crop.float(),
                                          frame2_crop.float(),
                                          mask_crop.float(),
                                          train=True)  # .cuda()

            partial_predictions = self.detector.predict(
                f2_feature.cpu().numpy())
            pos_pred = 0
            for label in partial_predictions:
                if label in self.params['pos_lbl']:
                    pos_pred += 1
            backward_pos_rate = pos_pred / len(partial_predictions)

            if backward_pos_rate <= 0.06:

                backward_state[currTrack] = None
                backward_rates[currTrack] = backward_pos_rate
                tqdm.write('BACKWARD - Cluster Num: {}, Frame Num: {}, Match: NONE'.format(
                    indx, frame_num))
            else:
                # has potential match min closest cluster and append with rate
                minDist = 100
                # maxIntersect = 0
                minTrack = None
                # currLocs = currTrack.locs

                for nextTrack in self.tracks[frame_num - 1]:
                    if calc_euclidean_dist(currTrack.centroid, nextTrack.centroid) < minDist:
                        # find closest track with most intersetions
                        # intersect = np.isin(
                        #     currLocs, nextTrack.locs, assume_unique=True)
                        # intersect_num = np.count_nonzero(
                        #     intersect.all(0).any())

                        # if intersect_num >= maxIntersect:
                        #     minDist = calc_euclidean_dist(
                        #         currTrack.centroid, nextTrack.centroid)
                        #     maxIntersect = intersect_num
                        #     minTrack = nextTrack
                        minTrack = nextTrack
                        minDist = calc_euclidean_dist(
                            currTrack.centroid, nextTrack.centroid)
                backward_state[currTrack] = minTrack
                backward_rates[currTrack] = backward_pos_rate
                tqdm.write('BACKWARD - Cluster Num: {}, Frame Num: {}, Match: {}'.format(
                    indx, frame_num, backward_pos_rate))

        # return_dict['bs'] = backward_state
        # return_dict['br'] = backward_rates
        return backward_state, backward_rates

    def Tget_predictor_results(self, frame_num, forward_scan=True, backward_scan=True):
        '''
            frame_num: the track index [1, 70]
            [forward, backward]_state: each index points to list of 
                                       clusters that it may match too, if any

        '''
        forward_state = {}
        forward_rates = {}
        backward_state = {}
        backward_rates = {}

        manager = multiprocessing.Manager()
        return_dict = manager.dict()
        jobs = []

        if forward_scan:
            # forward_state, forward_rates = self.forward_scan(frame_num)
            p1 = Process(target=self.forward_scan,
                         args=(frame_num, return_dict))
            p1.start()
            jobs.append(p1)

        if backward_scan:
            # backward_state, backward_rates = self.backward_scan(frame_num)
            p2 = Process(target=self.backward_scan,
                         args=(frame_num, return_dict))
            p2.start()
            jobs.append(p2)

        for proc in jobs:
            proc.join()

        # print(return_dict.keys())
        if 'fs' in return_dict.keys():
            forward_state, forward_rates = return_dict['fs'], return_dict['fr']
        if 'bs' in return_dict.keys():
            backward_state, backward_rates = return_dict['bs'], return_dict['br']

        return forward_state, forward_rates, backward_state, backward_rates

    def get_predictor_results(self, frame_num, forward_scan=True, backward_scan=True):
        '''
            frame_num: the track index [1, 70]
            [forward, backward]_state: each index points to list of 
                                       clusters that it may match too, if any

        '''
        forward_state = {}
        forward_rates = {}
        backward_state = {}
        backward_rates = {}

        if forward_scan:
            forward_state, forward_rates = self.forward_scan(frame_num)

        if backward_scan:
            backward_state, backward_rates = self.backward_scan(frame_num)

        return forward_state, forward_rates, backward_state, backward_rates

    def find_relations(self, currFrame, forward_state, forward_rates, backward_state, backward_rates):
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
            track_indx = self.tracks[currFrame].index(forward_key)
            currTrack = self.tracks[currFrame][track_indx]

            if forward_match_count == 0 or backward_match_count == 0:
                # object dead no match in next frame
                tqdm.write('\t DEATH forward_match_count: {} backward_match_count; {}'
                           .format(forward_match_count, backward_match_count))
                currTrack.state = Status.DEAD
                self.DEATH_COUNT += 1
                continue

            if forward_match_count == 1 and backward_match_count == 1:
                # object has 1 to 1 match with next frame
                tqdm.write('\t MATCH forward_match_count: {} backward_match_count; {}'
                           .format(forward_match_count, backward_match_count))
                if forward_rates[forward_key] >= self.params['pos_threshhold']\
                        and backward_rates[backward_key] >= self.params['pos_threshhold']:
                    # find the track asscoiated with backward_key
                    track_indx = self.tracks[currFrame + 1].index(backward_key)
                    track = self.tracks[currFrame + 1][track_indx]
                    track.id = currTrack.id
                    track.state = currTrack.state
                    track.origin += currTrack.origin + \
                        Status.MATCH.format(currFrame)

                    self.ONE_TO_ONE_COUNT += 1

            # if forward_match_count > 1 and backward_match_count == 1:
            if forward_match_count > 1:
                # object has split in next frame, 1 to many relation
                tqdm.write('\t SPLIT forward_match_count: {} backward_match_count; {}'
                           .format(forward_match_count, backward_match_count))
                # TODO replace with filter
                for backward_keys in backward_state.keys():
                    if backward_state[backward_keys] == forward_key:
                        # find the track asscoiated with backward_key
                        track_indx = self.tracks[currFrame +
                                                 1].index(backward_keys)
                        track = self.tracks[currFrame + 1][track_indx]
                        track.id = self.nextID
                        self.nextID += 1
                        track.state = Status.ACTIVE
                        track.origin += Status.SPLIT.format(
                            currFrame, currTrack.id)

                self.SPLIT_COUNT += 1

            # if forward_match_count == 1 and backward_match_count > 1:
            if backward_match_count > 1:
                # object has merged with another object, many to 1 relation
                tqdm.write('\t MERGE forward_match_count: {} backward_match_count; {}'
                           .format(forward_match_count, backward_match_count))
                # find the track asscoiated with backward_key
                track_indx = self.tracks[currFrame + 1].index(backward_key)
                track = self.tracks[currFrame + 1][track_indx]
                track.id = self.nextID
                self.nextID += 1
                track.state = Status.ACTIVE
                # TODO replace with filter
                for forward_keys in forward_state.keys():
                    if forward_state[forward_keys] == backward_key:
                        track.origin += Status.MERGE.format(
                            currFrame, forward_keys.id)
                        # do this so we dont double count it
                        forward_state[forward_keys] = None
                self.MERGE_COUNT += 1

        # remaining unlabled clusters in next frame are all the result of birth
        for back_key in backward_state.keys():
            # find the track asscoiated with back_key
            next_track_indx = self.tracks[currFrame + 1].index(back_key)
            nextTrack = self.tracks[currFrame + 1][next_track_indx]
            if nextTrack.id is None:
                # this cluster is birthed in this frame
                tqdm.write('\t BIRTH ')
                nextTrack.id = self.nextID
                self.nextID += 1
                nextTrack.state = Status.ACTIVE
                nextTrack.origin += Status.BIRTH.format(currFrame + 1)
                self.BIRTH_COUNT += 1

    def train_all(self):
        f1 = np.zeros((1, self.params['embedding_len']))
        f2 = np.zeros((1, self.params['embedding_len']))
        for epoch_id in tqdm(range(0, self.params['detect_num'])):
            # dynamically calculate the number of training examples
            trainNum = self.counts[str(self.currSearchFrame)]
            f1_feature, f2_feature = self.predictor_features(epoch_id)
            self.trainExample += 1
            f1 = np.append(f1, f1_feature.cpu().numpy(), axis=0)
            f2 = np.append(f2, f2_feature.cpu().numpy(), axis=0)

            if self.trainExample == trainNum:
                self.currSearchFrame += 1
                self.trainExample = 0
            # reset the current search frame if all clusters have been searched
            if self.currSearchFrame == 70:
                self.currSearchFrame = 0
                self.trainExample = 0

        np.save('../../data/f1.npy', f1)
        np.save('../../data/f2.npy', f2)

    def predict_all(self, core_num):
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
            4. Will assign ID's to all tracks in self.tracks
        '''
        # will hold each track in a frame and what it points too
        next_forward_state = []
        forward_state = []
        backward_state = []

        # retreive the largest id from first frame and increment this will be the next id
        self.nextID = max(track.id for track in self.tracks[1]) + 1

        # iterate through all frames , note we are checking next frame
        for currFrame in tqdm(range(1, 70)):
            # check all tracks in this frame
            if currFrame == 1:
                forward_state, forward_rates, _,  _ = self.get_predictor_results(currFrame,
                                                                                 backward_scan=False)
            else:
                forward_state = next_forward_state
                forward_rates = next_forward_rates

            if currFrame == 69:
                _, _, backward_state, backward_rates = self.get_predictor_results(currFrame + 1,
                                                                                  forward_scan=False)
            else:
                next_forward_state, next_forward_rates, backward_state, backward_rates = self.get_predictor_results(
                    currFrame + 1)

            tqdm.write('\tCheck Relation States')
            self.find_relations(currFrame,
                                forward_state,
                                forward_rates,
                                backward_state,
                                backward_rates)

    def gen_pos_rates(self):
        self_pos_rate_list = []
        forward_pos_rate_list = []

        for frame_num in tqdm(range(1, self.params['T'])):
            frame_tracks = self.tracks[frame_num]
            frameCurr = self.full_data[frame_num]
            frameNext = self.full_data[frame_num + 1]

            for currTrack in frame_tracks:
                mask, label = self.getMask(currTrack)
                width, height = self.find_min_crop(mask, label)
                frame1_crop = self.crop_frame(frameCurr, label, width, height)
                frame2_crop = self.crop_frame(frameNext, label, width, height)
                mask_crop = self.crop_frame(mask, label, width, height)

                f1_feature, f2_feature = self.detector(frame1_crop.float(),
                                                       frame2_crop.float(),
                                                       mask_crop.float())  # .cuda()

                partial_predictions = self.detector.predict(
                    f2_feature.cpu().numpy())
                pos_pred = 0
                for label in partial_predictions:
                    if label in self.params['pos_lbl']:
                        pos_pred += 1
                forward_pos_rate = pos_pred / len(partial_predictions)
                forward_pos_rate_list.append(forward_pos_rate)

                partial_predictions = self.detector.predict(
                    f1_feature.cpu().numpy())
                pos_pred = 0
                for label in partial_predictions:
                    if label in self.params['pos_lbl']:
                        pos_pred += 1
                self_pos_rate = pos_pred / len(partial_predictions)
                self_pos_rate_list.append(self_pos_rate)

        save_pickle(self_pos_rate_list, '../../data/self_pos_rate_list.pickle')
        save_pickle(forward_pos_rate_list,
                    '../../data/forward_pos_rate_list.pickle')

    def calc_sizes():
        cluster_size_list = []
        for frame_num in tqdm(range(1, self.params['T'] + 1)):
            for currTrack in self.tracks[frame_num]:
                cluster_size_list.append(len(currTrack.locs))
        save_pickle(cluster_size_list,
                    '../../data/cluster_size_list.pickle')


if __name__ == "__main__":
    print('-------------- Initializing the Protein Tracker ------------------')
    # load model params
    model_config = load_json('./model_config.json')

    # init the training wrapper
    tracker = Tracker(args, model_config[args.model_task])
    tracker.load_data(track='../../data/tracks_protein.pickle',
                      count='../../data/counts.json')
    tracker.detector.share_memory()

    core_num = multiprocessing.cpu_count()
    print("Running with " + str(core_num) + " cores.")

    # use the GPU if there is one
    # gpu_num = torch.cuda.device_count()
    # print('GPU NUM: {:2d}'.format(gpu_num))
    # if gpu_num > 1:
    #     torch.cuda.set_device(0)
    # print(multiprocessing.get_all_start_methods())
    # multiprocessing.set_start_method('spawn')
    # tracker.detector = torch.nn.DataParallel(
    #     tracker.detector, list(range(gpu_num))).cuda()

    if args.task == "optim_param":
        print('-------------- Show Tracking Statistics ------------------')
        
        # # histogram of pos detection rates on forward scan and self scan
        # if not(path.exists('../../data/self_pos_rate_list.pickle') and path.exists('../../data/forward_pos_rate_list.pickle')):
        #     print('Warning - need to run tracker.gen_pos_rates first.')
        #     tracker.gen_pos_rates()
        # info.hist_pos_rates('../../data/self_pos_rate_list.pickle',
        #                     '../../data/forward_pos_rate_list.pickle')

        # # histogram of whole cluster sizes
        # if not path.exists('../../data/cluster_size_list.pickle'):
        #     print('Warning - need to run tracker.calc_sizes first.')
        #     tracker.calc_sizes()
        # info.hist_sizes('../../data/cluster_size_list.pickle')
        
        # # histogram of tracking events
        # info.event_counts('../../data/tracks_protein_pretty.json')

        # show tracking compared to ground truth
        info.tracking_results('../../data/labeled_protein_tracks.pickle',
                              '../../data/test/mapping.json')

    elif args.task == "train":
        print('-------------- Train the Protein Tracker ----------------')
        tracker.train_all()
        print('-------------- Fit the Protein Tracker ------------------')
        f1 = np.load('../../data/f1.npy')
        f2 = np.load('../../data/f2.npy')
        tracker.detector.train_feat(f1[1:, ...], f2[1:, ...])

    elif args.task == 'predict':
        print('-------------- Run Tracker Prediction ------------------')
        # check detector has already been trained
        if not path.exists('../../models/detector.pickle'):
            print('ERROR: You must train the detector first')
            print('\tpython3 deep_tracker.py --task train --type detect')
            exit()

        tracker.predict_all(core_num)
        save_pickle(tracker.tracks, '../../data/labeled_protein_tracks.pickle')
        print('-------------- Tracking Statistics ------------------')
        print('Deaths: ', tracker.DEATH_COUNT)
        print('Births: ', tracker.BIRTH_COUNT)
        print('One to Ones: ', tracker.ONE_TO_ONE_COUNT)
        print('Splits: ', tracker.SPLIT_COUNT)
        print('Merges: ', tracker.MERGE_COUNT)
        print('-------------- Saving Labeled Tracks ------------------')
        save_tracks_as_json(args.model_task,
                            tracker.tracks,
                            tracker.DEATH_COUNT,
                            tracker.ONE_TO_ONE_COUNT,
                            tracker.SPLIT_COUNT,
                            tracker.MERGE_COUNT,
                            tracker.BIRTH_COUNT)
        print('----------------------- Done --------------------------')
