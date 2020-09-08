import argparse
from os import path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pickle
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import math

from util import open_model_json, save_as_json, showTensor, calc_euclidean_dist, has_intersection
from detector import Detector
from track import Track
import torch.nn.functional as F
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
            # print(partial_locs)
            # print(parts_count)
            for index in partial_locs:
                mask[parts_count, :, index[2], index[0], index[1]] = 1

            start = end
            parts_count += 1

        # print(mask.shape)
        # print(torch.unique(mask), (torch.flatten(mask)).sum())
        # exit()
        # print(curr_track.centroid)
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
        # print(center, x_max, x_min, y_max, y_min)
        return tensor[:, :, :, x_min:x_max, y_min:y_max]

    def find_min_crop(self, mask, label):
        # TODO: need to dynamically figure out the min width and height cant be too small
        # find the indexes of all nonzero elements in tensor
        locs = torch.nonzero(mask)
        # print(locs)
        # find the delta, which is the range of each z,x,y dim,
        # to shift the mask by it by its own body length
        x_max = max(locs[:, 3])
        x_min = min(locs[:, 3])
        y_min = min(locs[:, 4]) 
        y_max = max(locs[:, 4])

        x_range = max(abs(x_max - label[1]), abs(x_min - label[1]))
        y_range = max(abs(y_max - label[2]), abs(y_min - label[2]))

        width = math.ceil(x_range + self.params['crop_window'])
        height = math.ceil(y_range + self.params['crop_window'])
        # tqdm.write('width: {} height: {} label: {}'.format(width, height, label))
        # exit()
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
        # print('frame _size_full: ', frame1.shape, frame2.shape)

        # frame1_crop = self.crop_frame(
        #     self.weights_data[self.currSearchFrame], label, 50, 50)
        # frame2_crop = self.crop_frame(
        #     self.weights_data[self.currSearchFrame + 1], label, 50, 50)
        width, height = self.find_min_crop(mask, label)
        frame1_crop = self.crop_frame(frame1, label, width, height)
        frame2_crop = self.crop_frame(frame2, label, width, height)
        mask_crop = self.crop_frame(mask, label, width, height)
        # print(frame1_crop.shape, frame2_crop.shape, mask_crop.shape)
        f1_feature, f2_feature = self.detector(frame1_crop.cuda().float(),
                                               frame2_crop.cuda().float(),
                                               mask_crop.cuda().float())

        tqdm.write('Predictor Epoch: {}, Cluster Num: {}, Frame Num: {}'.format(
            epoch_id, self.trainExample, self.currSearchFrame))

        return f1_feature, f2_feature

    def find_optimal_param(self, f1, f2):
        print(f1.shape, f2.shape)

        f1_pos_rate = []
        f2_pos_rate = []
        partial_predictions = trainer.detector.predict(f1)
        for predictions in partial_predictions:
           
            pos_pred = 0
            print(predictions)
            for label in partial_predictions:
                if label == trainer.params['pos_lbl']:
                    pos_pred += 1

            f1_pos_rate.append(pos_pred / len(partial_predictions))

        
if __name__ == "__main__":
    print('-------------- Initializing the Protein Tracker ------------------')
    # load model params
    model_config = open_model_json('./model_config.json')

    # init the training wrapper
    trainer = Trainer(args, model_config[args.model_task])
    trainer.load_data(track='../../data/tracks.pickle',
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
        axs[0].hist(f1_pos_rate_list, bins=100 )
        axs[0].set_title('Cluster on Own Frame')
        axs[1].hist(f2_pos_rate_list, bins=100 )
        axs[1].set_title('Cluster on Next Frame')
        plt.show()

        # histogram of whole cluster sizes
        if not path.exists('../../data/cluster_size_list.pickle'):
            cluster_size_list = []
            for f in range(1,71):
                for currTrack in trainer.tracks[f]:
                    cluster_size_list.append(len(currTrack.locs))

            with open('../../data/cluster_size_list.pickle', 'wb') as f:
                # Pickle the 'data' dictionary using the highest protocol available.
                pickle.dump(cluster_size_list, f, pickle.HIGHEST_PROTOCOL)
        else:
            with open('../../data/cluster_size_list.pickle', 'rb') as f:
                # Pickle the 'data' dictionary using the highest protocol available.
                cluster_size_list =  pickle.load(f)

        
        plt.hist(cluster_size_list, bins=500)
        plt.title('Whole Cluster Sizes')
        plt.xlabel('Cluster Sizes')
        plt.xticks(np.arange(0,300, 5))
        plt.xlim(left=0, right=300)
        plt.ylabel('Number of Clusters')
        plt.show()

    elif args.task == "train":
        # print('-------------- Train the Protein Tracker ------------------')
        # f1 = np.zeros((1, trainer.params['embedding_len']))
        # f2 = np.zeros((1, trainer.params['embedding_len']))
        # for epoch_id in tqdm(range(0, trainer.params['detect_num'])):
        #     # dynamically calculate the number of training examples
        #     trainNum = trainer.counts[str(trainer.currSearchFrame)]
        #     f1_feature, f2_feature = trainer.predictor_features(epoch_id)
        #     trainer.trainExample += 1
        #     f1 = np.append(f1, f1_feature.cpu().numpy(), axis=0)
        #     f2 = np.append(f2, f2_feature.cpu().numpy(), axis=0)

        #     if trainer.trainExample == trainNum:
        #         trainer.currSearchFrame += 1
        #         trainer.trainExample = 0
        #     # reset the current search frame if all clusters have been searched
        #     if trainer.currSearchFrame == 70:
        #         trainer.currSearchFrame = 0
        #         trainer.trainExample = 0

        # with open('../../data/f1.npy', 'wb') as f:
        #     np.save(f, f1)
        # with open('../../data/f2.npy', 'wb') as f:
        #     np.save(f, f2)

        print('-------------- Fit the Protein Tracker ------------------')
        with open('../../data/f1.npy', 'rb') as f:
            f1 = np.load(f)
        with open('../../data/f2.npy', 'rb') as f:
            f2 = np.load(f)
        
        trainer.detector.train_feat(f1[1:, ...], f2[1:, ...])

    elif args.task == 'predict':
        ''' 
            Will run the tracking algo on the entire dataset
        '''
        # store the posrates
        f1_pos_rate_list = []
        f2_pos_rate_list = []

        # check detector has already been trained
        if not path.exists('../../models/detector.pickle'):
            print('ERROR: You must train the detector first')
            print('\tpython3 deep_tracker.py --task train --type detect')
            exit()

        # retreive the largest id from first frame and increment this will be the next id
        nextID = max(track.id for track in trainer.tracks[1]) + 1
        # iterate through all frames , note we are checking next frame
        for currFrame in tqdm(range(0, 69)):
            # check all tracks in this frame
            for currTrack in trainer.tracks[currFrame + 1]:
                # print(currTrack)
                # get current track info
                currId = currTrack.id
                currCentroid = currTrack.centroid
                currState = currTrack.state
                currOrigin = currTrack.origin
                currLocs = currTrack.locs
                # get the clusters in this frame
                mask, label = trainer.getMask(currTrack)
                # print(label)
                frame1 = trainer.full_data[0, currFrame, 0, ...]
                frame2 = trainer.full_data[0, currFrame + 1, 0, ...]
                frame1 = frame1.reshape(
                    (1, 1, frame1.size(0), frame1.size(1), frame1.size(2)))
                frame2 = frame2.reshape(
                    (1, 1, frame2.size(0), frame2.size(1), frame2.size(2)))
                # print(frame1.shape, frame2.shape)
                width, height = trainer.find_min_crop(mask, label)
                frame1_crop = trainer.crop_frame(frame1, label, width, height)
                frame2_crop = trainer.crop_frame(frame2, label, width, height)
                mask_crop = trainer.crop_frame(mask, label, width, height)

                f1_feature , f2_feature = trainer.detector(frame1_crop.cuda().float(),
                                                 frame2_crop.cuda().float(),
                                                 mask_crop.cuda().float())
                # print(f1_feature, f2_feature)

                partial_predictions = trainer.detector.predict(
                    f2_feature.cpu().numpy())
                pos_pred = 0
                for label in partial_predictions:
                    if label == trainer.params['pos_lbl']:
                        pos_pred += 1
                forward_pos_rate = pos_pred / len(partial_predictions)
                f2_pos_rate_list.append(forward_pos_rate)

                pp1 = trainer.detector.predict(f1_feature.cpu().numpy())
                pos_pred = 0
                for label in pp1:
                    if label == trainer.params['pos_lbl']:
                        pos_pred += 1
                f1_pos_rate = pos_pred / len(pp1)
                f1_pos_rate_list.append(f1_pos_rate)

                if forward_pos_rate >= trainer.params['pos_threshhold']:
                    # this is tooo slow
                    # TODO: refactor min centroid finder algo
                    minDist = 999999
                    minTrack = None
                    for nextTrack in trainer.tracks[currFrame + 2]:

                        if nextTrack.id is None:

                            if calc_euclidean_dist(currCentroid, nextTrack.centroid, [1, 1, 1]) < minDist:
                                minTrack = nextTrack
                                minDist = calc_euclidean_dist(
                                    currCentroid, nextTrack.centroid, [1, 1, 1])
                        elif nextTrack.id == currId:
                            print("should never happen")

                    if minTrack:
                        # print(minTrack)
                        # run the cluster backwards in time to ensure tracking
                        mask, label = trainer.getMask(minTrack)
                        width, height = trainer.find_min_crop(mask, label)
                        frame1_crop = trainer.crop_frame(frame1, label, width, height)
                        frame2_crop = trainer.crop_frame(frame2, label, width, height)
                        mask_crop = trainer.crop_frame(mask, label, width, height)

                        f1_feature , f2_feature = trainer.detector(frame2_crop.cuda().float(),
                                                         frame1_crop.cuda().float(),
                                                         mask_crop.cuda().float())
                        partial_predictions = trainer.detector.predict(
                            f2_feature.cpu().numpy())
                        pos_pred = 0
                        for label in partial_predictions:
                            if label == trainer.params['pos_lbl']:
                                pos_pred += 1
                        backward_pos_rate = pos_pred / len(partial_predictions)
                        f2_pos_rate_list.append(backward_pos_rate)

                        pp1 = trainer.detector.predict(f1_feature.cpu().numpy())
                        pos_pred = 0
                        for label in pp1:
                            if label == trainer.params['pos_lbl']:
                                pos_pred += 1
                        f1_pos_rate = pos_pred / len(pp1)
                        f1_pos_rate_list.append(f1_pos_rate)

                        if backward_pos_rate >= trainer.params['pos_threshhold']:
                            tqdm.write('match')
                            minTrack.id = currId
                            minTrack.state = currState
                            minTrack.origin = currOrigin
                        else:
                            tqdm.write('no reverse match')
                            currTrack.state = 'dead'
                        # tqdm.write('backward pos_rate: {} forward pos_rate: {}'.format(
                        #     backward_pos_rate, forward_pos_rate))
                else:
                    tqdm.write('no forward match')
                    currTrack.state = 'dead'

            # remaining unlabled clusters in next frame are all the result
            # of birth, split, or merge
            for nextTrack in trainer.tracks[currFrame + 2]:
                if nextTrack.id is None:
                    intersections = has_intersection(
                        nextTrack.locs, trainer.tracks[currFrame + 1])
                    if len(intersections) == 1:
                        # split, there is only one intersecting cluster from prev frame
                        nextTrack.id = nextID
                        nextID += 1
                        nextTrack.state = 'active'
                        nextTrack.origin = 'split from: ' + \
                            str([trainer.tracks[currFrame + 1]
                                 [indx].id for indx in intersections])
                    elif len(intersections) > 1:
                        # merge, there is more than one intersecting cluster from prev frame
                        nextTrack.id = nextID
                        nextID += 1
                        nextTrack.state = 'active'
                        nextTrack.origin = 'merge from: ' + \
                            str([trainer.tracks[currFrame + 1]
                                 [indx].id for indx in intersections])
                    else:
                        # this cluster is birthed in this frame
                        nextTrack.id = nextID
                        nextID += 1
                        nextTrack.state = 'active'
                        nextTrack.origin = 'birth'

        with open('../../data/labeled_tracks1.pickle', 'wb') as f:
            # Pickle the 'data' dictionary using the highest protocol available.
            pickle.dump(trainer.tracks, f, pickle.HIGHEST_PROTOCOL)

        with open('../../data/f1_pos_rate_list.pickle', 'wb') as f:
            # Pickle the 'data' dictionary using the highest protocol available.
            pickle.dump(f1_pos_rate_list, f, pickle.HIGHEST_PROTOCOL)
        with open('../../data/f2_pos_rate_list.pickle', 'wb') as f:
            # Pickle the 'data' dictionary using the highest protocol available.
            pickle.dump(f2_pos_rate_list, f, pickle.HIGHEST_PROTOCOL)

        save_as_json(trainer.tracks, 0, 0, 0)
