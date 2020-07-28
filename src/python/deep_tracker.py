import argparse
import numpy as np
import torch
import torch.nn as nn
import pickle
import time
from tqdm import tqdm
from util import open_model_json, showTensor
from feature_extractor import FeatureExtractor
from loss_calculator import Loss_Calculator
from network import Network
from track import Track


# Parse arguments
parser = argparse.ArgumentParser(
    description='Edit model hyper-parameters in model_config.json')
# Task
parser.add_argument('--model_task', default='default',
                    help="Choose a model with different hyper-parameters (specified in 'modules/model_config.json')")
parser.add_argument('--train', required=True, choices=['train', 'predict'],
                    help="Choose to train or predict with the model")
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
        self.network = Network(self.params)
        self.optimizer = torch.optim.Adam(self.network.parameters(),
                                          lr=self.params['lr'],
                                          betas=(0.9, 0.99),
                                          weight_decay=1e-6)
        self.trainLossSum = 0
        self.currSearchFrame = 1
        # track the current example
        self.trainExample = 1
        self.testExample = 0
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
        # mask = torch.zeros((13, 280, 512))
        # for index in locs:
        #     mask[index[2], index[0], index[1]] = 1
        # mask = mask.reshape(
        #     (1, 1, 1, mask.size(0), mask.size(1), mask.size(2)))
        # return mask

        a = torch.zeros((6, 1, 4))
        a[:, 0, 0] = 1
        a[:, 0, 1:] = torch.FloatTensor(curr_track.centroid)

        return a

    def forward(self, input_seq, mask):
        '''
            Forward function return loss

            Input: input_seq, the entire frame sequence
                   mask, the individual cluster

            Output: loss
                    elapsed_time 
                    output, the predictions for each frame
        '''
        start_time = time.time()
        loss, output = self.network(
            input_seq.float(), mask.float(), trainer.currSearchFrame)
        elapsed_time = time.time() - start_time
        return loss, elapsed_time, output

    def backward(self, loss):
        '''
            Backward function

            Input: loss

            Output: elapsed_time
        '''
        start_time = time.time()
        self.optimizer.zero_grad()
        loss.backward()
        if self.params['grad_clip'] > 0:
            nn.utils.clip_grad_norm(self.network.parameters(),
                                    self.params['grad_clip'])
        self.optimizer.step()
        elapsed_time = time.time() - start_time
        return elapsed_time

    def run_batch(self, full_data, mask, train):
        '''
            The forward and backward passes for an iteration

            Input: full_data, the entire frame sequence
                   mask, the individual cluster
                   train, train or prediction

            Output: loss, output
        '''
        if train == 'train':
            loss, forward_time, output = self.forward(full_data, mask)
            backward_time = self.backward(loss)
        else:
            with torch.no_grad():
                loss, forward_time, output = self.forward(full_data, mask)
            backward_time = 0

        tqdm.write('Runtime: {}s'.format(forward_time + backward_time))
        return loss.item(), output

    def run_train_epoch(self, epoch_id):
        '''
            Train one batch
        '''
        # configure network for training
        self.network.train()

        # dynamically calculate the number of training examples
        trainNum, testNum = self.calc_batches(self.currSearchFrame)

        # get the clusters in this frame
        frame_tracks = self.tracks[self.currSearchFrame]
        
        # TODO: increment examples
        for batchId in range(self.trainExample, self.trainExample + self.params['batch_size']):
            currTrack = frame_tracks[batchId]
            mask = self.getMask(currTrack)

            loss, output = self.run_batch(self.full_data, mask, 'train')

            self.trainLossSum = self.trainLossSum + loss

            tqdm.write('Epoch: {}, iter: {}/{}, loss: {}'.format(epoch_id,
                                                                 batchId,
                                                                 self.trainExample +
                                                                 self.params['batch_size'],
                                                                 loss))

            # record metrics every 100 epochs
            # if self.params['train_log_interval'] > 0 and epoch_id % self.params['train_log_interval'] == 0:
            #     self.benchmark['train_loss'].append(
            #         (epoch_id, self.trainLossSum/self.params['train_log_interval']))
            #     self.trainLossSum = 0

            # if self.params['validate_interval'] > 0 and epoch_id % self.params['validate_interval'] == 0:
            #     # run the newtwork on the test clusters
            #     val_loss, output = self.run_test_epoch(frame_tracks, testNum)
            #     self.network.train()
            #     self.benchmark['val_loss'].append((epoch_id, val_loss))
            #     savepoint = {'param': self.params, 'benchmark': self.benchmark}
            #     # utils.save_json(savepoint, self.save_path +
            #                     # str(epoch_id) + '_bench.json')
            #     savepoint['net_states'] = self.network.state_dict()
            #     torch.save(savepoint, self.save_path + str(i) + '.pt')

            # if self.params['save_interval'] > 0 and epoch_id % self.params['save_interval'] == 0:
            #     savepoint = {'param': self.params, 'benchmark': self.benchmark}
            #     savepoint['net_states'] = self.network.state_dict()
            #     torch.save(savepoint, self.save_path + 'latest.pt')

    def run_test_epoch(self, frame_tracks, test_num):
        '''
            The test function
        '''
        self.network.eval()
        val_loss_sum = 0
        num_frame_tracks = len(frame_tracks)

        for batchId in range(num_frame_tracks - 1 , num_frame_tracks - test_num, -1):
            currTrack = frame_tracks[batchId]
            mask = self.getMask(currTrack)
            loss, output = self.run_batch(self.full_data, mask, 'train')
            val_loss_sum = val_loss_sum + loss
            tqdm.write('Validation {} / {}, loss = {}'.format
                       (batchId - num_frame_tracks, test_num, loss))

        return val_loss_sum, output

    def run_prediction(self, frame_tracks, cluster_count):
        pass

if __name__ == "__main__":
    print('-------------- Train the Deep Tracker ------------------')
    # load model params
    model_config = open_model_json('./model_config.json')

    # init the training wrapper
    trainer = Trainer(args, model_config[args.model_task])
    trainer.load_data(track='../../data/tracks.pickle',
                      labled='../../data/raw3data.npy',
                      count='../../data/counts.json')


    # print(trainer.getMask( trainer.tracks[3][3]))
    # lc = Loss_Calculator()
    # i = torch.randn(1, 70, 8, requires_grad=True)
    # t = trainer.getMask(trainer.tracks[3][3])[0, 0, ...]
    # print(i)
    # print(t)
    # loss = lc(i, t, 3 - 1)
    # print('loss', loss)
    # loss.backward()



    # trainer.full_data.requires_grad=True
    # print(trainer.full_data[0, 1, 0, ...])



    # Run the trainer
    if args.train == 'train':
        # trainer.params['epoch_num']
        for epoch_id in tqdm(range(0,2)):
            trainer.run_train_epoch(epoch_id)

            # reset the current search frame if all clusters have been searched
            if trainer.currSearchFrame == 70:
                trainer.currSearchFrame = 0

    elif args.train == 'predict':
        # if args.init_model == '':
        #     print('ERROR: Must specify initial model to predict with it')
        #     exit()

        # savepoint = torch.load(trainer.save_path + 'latest.pt')
        # trainer.network.load_state_dict(savepoint['net_states'])

        # benchmark = savepoint['benchmark']
        # print('Model is initialized from ' + f)

        # param_num = sum([param.data.numel() for param in net.parameters()])
        # print('Parameter number: %.3f M' % (param_num / 1024 / 1024))

        # # TODO: run all clusters through this

        # trainer.run_prediction(frame_tracks, cluster_count)
        pass    