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

from util import open_model_json, showTensor
from network import Network
from detector import Detector
from track import Track

# Parse arguments
parser = argparse.ArgumentParser(
    description='Edit model hyper-parameters in model_config.json')
# Task
parser.add_argument('--model_task', default='default',
                    help="Choose a model with different hyper-parameters (specified in 'modules/model_config.json')")
parser.add_argument('--task', required=True, choices=['train', 'predict'],
                    help="Choose to train or predict with the model")
parser.add_argument('--type', required=True, choices=['deep', 'detect'],
                    help="Choose whether to train the detector or NN")
parser.add_argument('--init_model', default='',
                    help="Recover training from a checkpoint, e.g., 'latest.pt', '3000.pt'")
args = parser.parse_args()


class Trainer():
    '''
        Wrapper class to train the deep tracker
    '''

    def __init__(self, args, params):
        self.args = args
        self.params = params
        # init network and optimizer
        self.network = Network(self.params).cuda()
        self.detector = Detector(self.params).cuda()
        self.optimizer = torch.optim.Adam(self.network.parameters(),
                                          lr=self.params['lr'],
                                          betas=(0.9, 0.99),
                                          weight_decay=1e-6)
        self.mseLoss = nn.MSELoss().cuda()
        self.trainLossSum = 0
        self.currSearchFrame = 0
        # track the current example
        self.trainExample = 0
        # need to call load_data
        self.full_data = None
        self.tracks = None
        self.counts = None
        # for recording performance of model
        self.benchmark = {'train_loss': [], 'val_loss': []}
        self.save_path = '../../models/'

    def load_data(self, track, labled, count):
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
        data = torch.from_numpy(np.load(labled))
        data = data.permute(3, 2, 0, 1)
        data = data[None, :, None, :, :, :]
        #showTensor(full_data[0, 0, 0, 5, ...])
        # print('Shape of raw full sized data: ', data.shape, type(data))
        self.full_data = data

    def calc_batches(self, time_step):
        '''
            The setting is constant so we dynamically calculate training examples 
            thus each time_step will have a 85% of its clusters used as training examples
            and a dynamic per time step batch size

            Input: time_step, the current frame number 

            Output: train_num, test_num
        '''
        train_num = int(self.counts[str(time_step)]
                        * self.params['train_test_ratio'])
        test_num = self.counts[str(time_step)] - train_num
        return train_num, test_num

    def getMask(self, curr_track):
        '''
            Create initial input weights
        '''
        mask = torch.zeros((13, 280, 512))
        for index in curr_track.locs:
            mask[index[2], index[0], index[1]] = 6
        mask = mask.reshape(
            (1, 1, 1, mask.size(0), mask.size(1), mask.size(2)))
        a = torch.zeros(3)
        a[0] = curr_track.centroid[2]
        a[1] = curr_track.centroid[0]
        a[2] = curr_track.centroid[1]
        return mask, a

    def forward(self, frame1, frame2, mask, label):
        '''
            Forward function return loss

            Input: input_seq, the entire frame sequence
                   mask, the individual cluster

            Output: loss
                    elapsed_time 
                    output, the predictions for each frame
        '''
        start_time = time.time()
        loss, out1, out2 = self.network(frame1.cuda().float(),
                                        frame2.cuda().float(),
                                        mask.cuda().float(),
                                        label.cuda().float())
        elapsed_time = time.time() - start_time
        return loss, out1, out2, elapsed_time

    def backward(self, loss):
        '''
            Backward function

            Input: loss

            Output: elapsed_time
        '''
        start_time = time.time()
        self.optimizer.zero_grad()
        loss.backward()
        # if self.params['grad_clip'] > 0:
        #     nn.utils.clip_grad_norm(self.network.parameters(),
        #                             self.params['grad_clip'])
        self.optimizer.step()
        elapsed_time = time.time() - start_time
        return elapsed_time

    def run_batch(self, frame1, frame2, mask, label, train):
        '''
            The forward and backward passes for an iteration

            Input: full_data, the entire frame sequence
                   mask, the individual cluster
                   train, train or prediction

            Output: loss, output
        '''
        if train == 'train':
            loss, _, _, forward_time = self.forward(
                frame1, frame2, mask, label)
            backward_time = self.backward(loss)
        else:
            with torch.no_grad():
                loss, _, _, forward_time = self.forward(
                    frame1, frame2, mask, label)
            backward_time = 0

        # tqdm.write('Runtime: {}s'.format(forward_time + backward_time))
        return loss.item()

    def run_train_epoch(self, epoch_id):
        '''
            Train one batch we have batch size of one due to the time and size 
            of training one example
        '''
        # configure network for training
        self.network.train()

        # get the clusters in this frame
        frame_tracks = self.tracks[self.currSearchFrame]

        currTrack = frame_tracks[self.trainExample]
        mask = self.getMask(currTrack)

        loss, _ = self.run_batch(
            self.full_data[0, 0:30, :, ...], mask, 'train')

        self.trainLossSum = self.trainLossSum + loss

        tqdm.write('Epoch: {}, batch: {}, loss: {}'.format(epoch_id,
                                                           self.trainExample,
                                                           loss))
        self.trainExample += 1

        # record metrics every 100 epochs
        if self.params['train_log_interval'] > 0 and epoch_id % self.params['train_log_interval'] == 0:
            self.benchmark['train_loss'].append(
                (epoch_id, self.trainLossSum/self.params['train_log_interval']))
            self.trainLossSum = 0

        # if self.params['validate_interval'] > 0 and epoch_id % self.params['validate_interval'] == 0:
        #     # run the newtwork on the test clusters
        #     val_loss, output = self.run_test_epoch(frame_tracks, 5)
        #     self.network.train()

        if self.params['save_interval'] > 0 and epoch_id % self.params['save_interval'] == 0:
            savepoint = {'param': self.params, 'benchmark': self.benchmark}
            savepoint['net_states'] = self.network.state_dict()
            torch.save(savepoint, self.save_path + 'latest.pt')

    def run_train2_epoch(self, epoch_id, testNum):
        '''
            Train one batch we have batch size of one due to the time and size 
            of training one example
        '''
        # configure network for training
        self.network.train()

        # get the clusters in this frame
        frame_tracks = self.tracks[self.currSearchFrame + 1]
        currTrack = frame_tracks[self.trainExample]
        mask, label = self.getMask(currTrack)
        frame1 = self.full_data[0, self.currSearchFrame, 0, ...]
        frame2 = self.full_data[0, self.currSearchFrame + 1, 0, ...]
        frame1 = frame1.reshape(
            (1, 1, 1, frame1.size(0), frame1.size(1), frame1.size(2)))
        frame2 = frame2.reshape(
            (1, 1, 1, frame2.size(0), frame2.size(1), frame2.size(2)))
        # print(frame1.shape, frame2.shape)
        frame1_crop = self.crop_frame(frame1, label, 50, 50)
        frame2_crop = self.crop_frame(frame2, label, 50, 50)
        mask_crop = self.crop_frame(mask, label, 50, 50)

        _, f2_feature = self.detector(frame1_crop.cuda().float(),
                                      frame2_crop.cuda().float(),
                                      mask_crop.cuda().float())
        results = trainer.detector.predict(
            f2_feature.cpu().numpy().reshape(1, -7))

        if results == 2:
            loss = self.run_batch(frame1, frame2, mask, label, 'train')
            self.trainLossSum = self.trainLossSum + loss
            tqdm.write('Deep Epoch: {}, batch: {}, loss: {}'.format(epoch_id,
                                                                    self.trainExample,
                                                                    loss))
        else:
            tqdm.write('Deep Epoch: {}, batch: {}, No Train Not in next frame'.format(epoch_id,
                                                                                      self.trainExample))

        # record metrics every 100 epochs
        if self.params['train_log_interval'] > 0 and epoch_id % self.params['train_log_interval'] == 0:
            self.benchmark['train_loss'].append(
                (epoch_id, self.trainLossSum/self.params['train_log_interval']))
            self.trainLossSum = 0

        if self.params['validate_interval'] > 0 and epoch_id % self.params['validate_interval'] == 0:
            # run the newtwork on the test clusters
            val_loss_avg = self.run_test_epoch(frame_tracks, testNum)
            self.benchmark['val_loss'].append(
                (epoch_id, val_loss_avg))
            tqdm.write('\n\tValidation Loss Average: {}\n'.format(val_loss_avg))
            self.network.train()

        if self.params['save_interval'] > 0 and epoch_id % self.params['save_interval'] == 0:
            savepoint = {'param': self.params, 'benchmark': self.benchmark}
            savepoint['net_states'] = self.network.state_dict()
            torch.save(savepoint, self.save_path + 'latest.pt')

    def run_test_epoch(self, frame_tracks, test_num):
        '''
            The test function
        '''
        self.network.eval()
        val_loss_sum = 0
        num_frame_tracks = len(frame_tracks)
        actual_test_num = 0

        for testId in range(num_frame_tracks - 1, num_frame_tracks - test_num, -1):
            currTrack = frame_tracks[testId]
            mask, label = self.getMask(currTrack)
            frame1 = self.full_data[0, self.currSearchFrame, 0, ...]
            frame2 = self.full_data[0, self.currSearchFrame + 1, 0, ...]
            frame1 = frame1.reshape(
                (1, 1, 1, frame1.size(0), frame1.size(1), frame1.size(2)))
            frame2 = frame2.reshape(
                (1, 1, 1, frame2.size(0), frame2.size(1), frame2.size(2)))

            frame1_crop = self.crop_frame(frame1, label, 50, 50)
            frame2_crop = self.crop_frame(frame2, label, 50, 50)
            mask_crop = self.crop_frame(mask, label, 50, 50)

            _, f2_feature = self.detector(frame1_crop.cuda().float(),
                                          frame2_crop.cuda().float(),
                                          mask_crop.cuda().float())
            results = trainer.detector.predict(
                f2_feature.cpu().numpy().reshape(1, -7))
            if results == 2:
                loss = self.run_batch(frame1, frame2, mask, label, 'val')
                actual_test_num += 1
                val_loss_sum += loss

        return val_loss_sum / actual_test_num

    def predictor_features(self, epoch_id):
        # get the clusters in this frame
        frame_tracks = self.tracks[self.currSearchFrame + 1]
        currTrack = frame_tracks[self.trainExample]
        mask, label = self.getMask(currTrack)
        frame1 = self.full_data[0, self.currSearchFrame, 0, ...]
        frame2 = self.full_data[0, self.currSearchFrame + 1, 0, ...]
        frame1 = frame1.reshape(
            (1, 1, 1, frame1.size(0), frame1.size(1), frame1.size(2)))
        frame2 = frame2.reshape(
            (1, 1, 1, frame2.size(0), frame2.size(1), frame2.size(2)))
        # print(frame1.shape, frame2.shape)

        frame1_crop = self.crop_frame(frame1, label, 50, 50)
        frame2_crop = self.crop_frame(frame2, label, 50, 50)
        mask_crop = self.crop_frame(mask, label, 50, 50)

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
        x_max = int(center[1] + width if center[1] + width < 280 else 280)
        x_min = int(center[1] - width if center[1] - width > 0 else 0)
        y_max = int(center[2] + height if center[2] + height < 512 else 512)
        y_min = int(center[2] - height if center[2] - height > 0 else 0)
        # print(center, x_max, x_min, y_max, y_min)
        return tensor[:, :, :, :, x_min:x_max, y_min:y_max]


if __name__ == "__main__":
    print('-------------- Train the Deep Tracker ------------------')
    # load model params
    model_config = open_model_json('./model_config.json')

    torch.cuda.set_device(0)

    # init the training wrapper
    trainer = Trainer(args, model_config[args.model_task])
    trainer.load_data(track='../../data/tracks.pickle',
                      labled='../../data/raw3data.npy',
                      count='../../data/counts.json')

    # Run the trainer
    if args.task == "train":
        if args.type == 'deep':
            # check detector has already been trained
            if not path.exists('../../models/detector.pickle'):
                print('ERROR: You must train the detector first')
                print('\tpython3 deep_tracker.py --task train --type detect')
                exit()

            for epoch_id in tqdm(range(0, trainer.params['epoch_num'])):
                # dynamically calculate the number of training examples
                trainNum, testNum = trainer.calc_batches(
                    trainer.currSearchFrame + 1)

                trainer.run_train2_epoch(epoch_id, testNum)
                trainer.trainExample += 1

                if trainer.trainExample == trainNum:
                    trainer.currSearchFrame += 1
                    trainer.trainExample = 0

                # reset the current search frame if all clusters have been searched
                if trainer.currSearchFrame == 69:
                    trainer.currSearchFrame = 0
                    trainer.trainExample = 0

        elif args.type == 'detect':
            f1 = np.empty((trainer.params['detect_num'], 7))
            f2 = np.empty((trainer.params['detect_num'], 7))
            for epoch_id in tqdm(range(0, trainer.params['detect_num'])):
                # dynamically calculate the number of training examples
                trainNum = trainer.counts[str(trainer.currSearchFrame + 1)]
                f1_feature, f2_feature = trainer.predictor_features(epoch_id)
                trainer.trainExample += 1
                f1[epoch_id] = f1_feature.cpu().numpy()
                f2[epoch_id] = f2_feature.cpu().numpy()

                if trainer.trainExample == trainNum:
                    trainer.currSearchFrame += 1
                    trainer.trainExample = 0

                # reset the current search frame if all clusters have been searched
                if trainer.currSearchFrame == 69:
                    trainer.currSearchFrame = 0
                    trainer.trainExample = 0

            with open('../../data/f1.npy', 'wb') as f:
                np.save(f, f1)
            with open('../../data/f2.npy', 'wb') as f:
                np.save(f, f2)

            # with open('../../data/f1.npy', 'rb') as f:
            #     f1 = np.load(f)
            # with open('../../data/f2.npy', 'rb') as f:
            #     f2 = np.load(f)

            trainer.detector.train_feat(f1, f2)

    elif args.task == 'predict':
        ''' 
            Will predict the tracking results of the first 2 frames 
            and compare with the results of the naive tracker
        '''
        if args.type == "deep":
            if args.init_model == '':
                print('ERROR: Must specify initial model to predict with it')
                exit()

            savepoint = torch.load(trainer.save_path + args.init_model)
            # print(savepoint['net_states'])
            trainer.network.load_state_dict(savepoint['net_states'])

            benchmark = savepoint['benchmark']

            # plt.plot(*zip(*benchmark['train_loss']), label='train_loss')
            # plt.plot(*zip(*benchmark['val_loss']), label='val_loss')
            # plt.legend()
            # plt.title('Training and Validation Loss')
            # plt.xlabel('Epoch')
            # plt.ylabel('Loss')
            # plt.show()

            print('Model is initialized from ',
                  trainer.save_path + args.init_model)

            param_num = sum([param.data.numel()
                             for param in trainer.network.parameters()])
            print('Parameter number: %.3f M' % (param_num / 1024 / 1024))

            # store the results
            # prediction = np.zeros((4, 270))
            # print(prediction)

            frame_tracks = trainer.tracks[1]
            # for cluster in range(0, len(frame_tracks)):
            currTrack = frame_tracks[0]
            mask, label = trainer.getMask(currTrack)
            frame1 = trainer.full_data[0, 0, 0, ...]
            frame2 = trainer.full_data[0, 0 + 1, 0, ...]
            frame1 = frame1.reshape(
                (1, 1, 1, frame1.size(0), frame1.size(1), frame1.size(2)))
            frame2 = frame2.reshape(
                (1, 1, 1, frame2.size(0), frame2.size(1), frame2.size(2)))

            # TODO: run all clusters through this
            with torch.no_grad():
                start = time.time()
                loss, out1, out2, elapsed_time = trainer.forward(
                    frame1, frame2, mask, label)
                end = time.time()
                def graph_3d( f):
                    f = torch.where(f.cpu() < 1,  torch.tensor([0]), torch.tensor([1]))
                    print(f)
                    fig = plt.figure()
                    ax = fig.add_subplot(111, projection='3d')
                    z, x, y = f[0, 0, ...].cpu().numpy().nonzero()
                    print(z, x, y)
                    ax.set_xlim3d(0, 280)
                    ax.set_ylim3d(0, 512)
                    ax.set_zlim3d(0, 13)
                    ax.scatter(x, y, z, zdir='z')
                    plt.show()
                graph_3d(out1)
                print(torch.where(out1.cpu() < 0.5,  torch.tensor([0]), torch.tensor([1])))
                print(out2)
                print("LOSS ", loss, "TIME: ", end - start)
                # prediction[0, cluster] = out1[0]
                # prediction[1, cluster] = out2

            # with open('../../data/prediction.npy', 'wb') as f:
            #     np.save(f, prediction)
        elif args.type == "detect":
            # load the feature data
            with open('../../data/f2.npy', 'rb') as f:
                f2 = np.load(f)

            # predict if the cluster exists in the next frame
            start = time.time()
            results = trainer.detector.predict(f2[0:270, :])
            print(results)
            end = time.time()
            print("Predict Time: ", end - start)

            # store the results
            prediction = np.zeros((4, 270))
            for i, r in enumerate(results):
                prediction[0, i] = 1
                prediction[1, i] = 1 if r == 2 else 0

            with open('../../data/prediction.npy', 'wb') as f:
                np.save(f, prediction)
