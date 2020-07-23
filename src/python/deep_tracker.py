import argparse
import numpy as np
import torch
import pickle
import time
from tqdm import tqdm
from util import open_model_json, showTensor
from feature_extractor import FeatureExtractor
from network import Network
from track import Track


# Parse arguments
parser = argparse.ArgumentParser(description='Edit model hyper-parameters in model_config.json')
# Task
parser.add_argument('--model_task', default='default',
                    help="Choose a model with different hyper-parameters (specified in 'modules/model_config.json')")
parser.add_argument('--train', required=True, choices=['train', 'test'],
                    help="Choose to train or test the model")
parser.add_argument('--init_model', default='',
                    help="Recover training from a checkpoint, e.g., 'sp_latest.pt', 'sp_3000.pt'")
args = parser.parse_args()


class Trainer():
    '''
        Wrapper class to train the deep tracker
    '''
    def __init__(self, args, params):
        self.args = args
        self.params = params
        self.train = args.train
        # init network and optimizer
        self.network = Network(self.params)
        self.optimizer = torch.optim.Adam(self.network.parameters(),
                                          lr=self.params['lr'],
                                          betas=(0.9, 0.99),
                                          weight_decay=1e-6)
        self.trainLossSum = 0
        # need to call load_data
        self.full_data = None
        self.tracks = None
        # for recording performance of model
        self.benchmark = {'train_loss': [], 'val_loss': [], 'iteration': 0}
        self.save_path = '../../models/'



    def load_data(self, track, labled):
        '''
            Load numpy data of unlabeled segemented data and tracks

            Return: numpy array
        '''
        # load the tracks
        with open(track, 'rb') as f:
            # The protocol version used is detected automatically, so we do not
            # have to specify it.
            self.tracks = pickle.load(f)
        # print(self.tracks)

        # load the frames
        data = torch.from_numpy(np.load(labled))
        data = data.permute(3, 2, 0, 1)
        data = data[None, :, None, :, :, :]
        #showTensor(full_data[0, 0, 0, 5, ...])
        print('Shape of raw full sized data: ',
              data.shape, type(data))

        self.full_data = data

        # Split the raw data into a 90% training and 10% test set
        # train_data = full_data[0:60, ...]
        # test_data = full_data[60:, ...]

        # print('train', train_data.shape, 'test', test_data.shape)
        # return train_data, test_data

    def getMask(self, locs):
        '''
            Create a mask that contains just that cluster
        '''
        mask = torch.zeros((13, 280, 512))
        for index in locs:
            mask[index[2], index[0], index[1]] = 1

        return mask

    def forward(self, input_seq):
        '''
            Forward function return loss
        '''
        start_time = time.time()
        loss = self.network(input_seq)
        elapsed_time = time.time() - start_time
        return loss, elapsed_time

    def backward(self, loss):
        '''
            Backward function
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

    def run_batch(self, full_data, train):
        '''
            The forward and backward passes for an iteration
        '''
        if train:
            loss, forward_time = self.forward(full_data)
            backward_time = self.backward(loss)
        else:
            with torch.no_grad():
                loss, forward_time = self.forward(full_data)
            backward_time = 0

        print('Runtime: %.3fs' % (forward_time + backward_time))
        return loss.item()

    def run_train_epoch(self, batch_id_start):
        self.network.train()

        for batch_id in range(batch_id_start, o.train_batch_num):

            o.batch_id = batch_id
            loss = run_batch(batch_id, 'train')
            trainLossSum = trainLossSum + loss
            i = i + 1




            print('Epoch: %.2f/%d, iter: %d/%d, batch: %d/%d, loss: %.3f' % 
            (i/o.train_batch_num, o.epoch_num, i, iter_num, batch_id+1, o.train_batch_num, loss))







            if o.train_log_interval > 0 and i % o.train_log_interval == 0:
                benchmark['train_loss'].append(
                    (i, trainLossSum/o.train_log_interval))
                trainLossSum = 0








            if o.validate_interval > 0 and (i % o.validate_interval == 0 or i == iter_num):
                
                val_loss = run_test_epoch() if o.test_batch_num > 0 else 0
                
                net.train()
                benchmark['val_loss'].append((i, val_loss))
                benchmark['iteration'] = i
                savepoint = {'o': vars(o), 'benchmark': benchmark}
                utils.save_json(
                    savepoint, result_file_header + str(i) + '.json')
                savepoint['net_states'] = net.state_dict()
                # savepoint['optim_states'] = optimizer.state_dict()
                torch.save(savepoint, result_file_header + str(i) + '.pt')








            if o.save_interval > 0 and (i % o.save_interval == 0 or i == iter_num):
                benchmark['iteration'] = i
                savepoint = {'o': vars(o), 'benchmark': benchmark}
                utils.save_json(savepoint, result_file_header + 'latest.json')
                savepoint['net_states'] = net.state_dict()
                # savepoint['optim_states'] = optimizer.state_dict()
                torch.save(savepoint, result_file_header + 'latest.pt')

            print('-' * 80)


    def run_test_epoch(self):
        '''
            The test function
        '''
        torch.save(net.states, result_file_header + 'tmp.pt')
        net.reset_states()
        net.eval()
        val_loss_sum = 0
        for batch_id in range(0, o.test_batch_num):
            o.batch_id = batch_id
            loss = run_batch(batch_id, 'test')
            val_loss_sum = val_loss_sum + loss
            print('Validation %d / %d, loss = %.3f' %
                (batch_id+1, o.test_batch_num, loss))
        print('Final validation loss: %.3f' % (val_loss))
        net.states = torch.load(result_file_header + 'tmp.pt')
        return val_loss


if __name__ == "__main__":
    print('-------------- Train the Deep Tracker ------------------')
    # load model params
    model_config = open_model_json('./model_config.json')
    print(args)

    # init the training wrapper
    trainer = Trainer(args, model_config[args.model_task])

    trainer.load_data(track='../../data/tracks.pickle',
                      labled="../../data/raw3data.npy")

    # print(trainer.tracks[1][0].locs)
    # mask = trainer.getMask(trainer.tracks[1][0].locs)
    # print(mask.shape)
    # showTensor(mask[1, ...])

    # Run the trainer
    if args.train == 1:
        epoch_id_start = 0 
        batch_id_start = 0 
        # for epoch_id in tqdm(range(epoch_id_start, model_config['epoch_num'])):
        #     trainer.run_train_epoch(batch_id_start)
        #     batch_id_start = 0
    else:
        # trainer.run_test_epoch()
        pass
