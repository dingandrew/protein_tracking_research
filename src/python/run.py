import argparse
import numpy as np
import torch
import pickle
from util import open_model_json, showTensor
from feature_extractor import FeatureExtractor
from network import Network
from track import Track


# Parse arguments
parser = argparse.ArgumentParser()
# Task
parser.add_argument('--type', choices=['tracker', 'deep'],
                    help="Choose a method to track with")

parser.add_argument('--model_TASK', default='default',
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


def load_data(labled):
    '''
        Load numpy data of unlabeled segemented data

        Return: numpy array
    '''
    data = np.load(labled)
    print('Shape of raw full sized data: ', data.shape)
    return data


def split_data(full_data):
    '''
        Split the raw data into a 90% training and 10% test set

        Return: train_data, test_data 
    '''
    train_data = full_data[0:60, ...]
    test_data = full_data[60:, ...]

    print('train', train_data.shape, 'test', test_data.shape)
    return train_data, test_data


def getMask(locs):
    '''
        Create a mask that contains just that cluster
    '''
    mask = np.zeros((13, 280, 512))

    # print(mask)
    for index in locs:
        mask[index[2], index[0], index[1]] = 1

    return mask


def forward(input_seq):
    '''
        Forward function return loss
    '''
    start_time = time.time()
    loss = network(input_seq)
    elapsed_time = time.time() - start_time
    return loss, elapsed_time


def backward(loss):
    '''
        Backward function
    '''
    start_time = time.time()
    optimizer.zero_grad()
    loss.backward()
    if args.grad_clip > 0:
        nn.utils.clip_grad_norm(network.parameters(), args.grad_clip)
    optimizer.step()
    elapsed_time = time.time() - start_time
    return elapsed_time


def run_batch(full_data, train):
    '''
        The forward and backward passes for an iteration
    '''
    if train:
        loss, forward_time = forward(full_data)
        backward_time = backward(loss)
    else:
        with torch.no_grad():
            loss, forward_time = forward(X_seq, **kwargs)
        backward_time = 0
   
    print('Runtime: %.3fs' % (forward_time + backward_time))
    return loss.item()


# The training function
i = benchmark['i_start']
iter_num = o.train_batch_num * o.epoch_num
train_loss_sum = 0


def run_train_epoch(batch_id_start):
    net.reset_states()
    net.train()
    global i, train_loss_sum
    for batch_id in range(batch_id_start, o.train_batch_num):
        o.batch_id = batch_id
        loss = run_batch(batch_id, 'train')
        train_loss_sum = train_loss_sum + loss
        i = i + 1
        if ((0 < o.reset_interval < 1 and np.random.random_sample() < o.reset_interval) or
                (o.reset_interval > 1 and i % round(max(o.reset_interval/o.T, 1)) == 0)):
            print('---------- State Reset ----------')
            net.reset_states()
        if o.print_interval > 0 and i % o.print_interval == 0:
            print('Epoch: %.2f/%d, iter: %d/%d, batch: %d/%d, loss: %.3f' %
                  (i/o.train_batch_num, o.epoch_num, i, iter_num, batch_id+1, o.train_batch_num, loss))
        if o.train_log_interval > 0 and i % o.train_log_interval == 0:
            benchmark['train_loss'].append(
                (i, train_loss_sum/o.train_log_interval))
            train_loss_sum = 0
        if o.validate_interval > 0 and (i % o.validate_interval == 0 or i == iter_num):
            val_loss = run_test_epoch() if o.test_batch_num > 0 else 0
            net.train()
            benchmark['val_loss'].append((i, val_loss))
            benchmark['i_start'] = i
            savepoint = {'o': vars(o), 'benchmark': benchmark}
            utils.save_json(savepoint, result_file_header + str(i) + '.json')
            savepoint['net_states'] = net.state_dict()
            # savepoint['optim_states'] = optimizer.state_dict()
            torch.save(savepoint, result_file_header + str(i) + '.pt')
        if o.save_interval > 0 and (i % o.save_interval == 0 or i == iter_num):
            benchmark['i_start'] = i
            savepoint = {'o': vars(o), 'benchmark': benchmark}
            utils.save_json(savepoint, result_file_header + 'latest.json')
            savepoint['net_states'] = net.state_dict()
            # savepoint['optim_states'] = optimizer.state_dict()
            torch.save(savepoint, result_file_header + 'latest.pt')
        print('-' * 80)



if __name__ == "__main__":
    # load model params
    model_config = open_model_json('./model_config.json')
    for k, v in model_config.items():
        vars(args)[k] = v

    print(args)

    network = Network(args.default)

    print(network.parameters())

    optimizer = torch.optim.Adam(
        network.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=1e-6)

    benchmark = {'train_loss': [], 'val_loss': [], 'i_start': 0}

    # #init fe
    # fe = FeatureExtractor(args.default)

    # print(fe)

    # data_dir = '/home/andrew/unsupervised_samples/tracking-by-animation/data/mnist/pt'
    # split = 'train'
    # filename = split + '_' + str(1) + '.pt'
    # X_seq = torch.load(path.join(data_dir, 'input', filename))
    # X_seq = X_seq[:1, :, :, :, :]

    # X_seq = X_seq.float().div_(255)
    # showTensor(X_seq[0, 0, 0, ...])
    # showTensor(X_seq[0, 1, 0, ...])
    # print('start')
    # a = fe(X_seq)
    # print('final', a.shape)

    # showTensor(a[0,:, :, 0])
    full_data = torch.from_numpy(
        load_data(labled="../../data/raw3data.npy"))

    full_data = full_data.permute(3, 2, 0, 1)

    full_data = full_data[None, :, None, :, :, :]
    showTensor(full_data[0, 0, 0, 5, ...])
    print(full_data.shape, type(full_data))

    train_data, test_data = split_data(full_data)

    with open('../../data/tracks.pickle', 'rb') as f:
        # The protocol version used is detected automatically, so we do not
        # have to specify it.
        tracks = pickle.load(f)
        
    # Run the model
    if o.train == 1:
        epoch_id_start = i // o.train_batch_num
        batch_id_start = i % o.train_batch_num
        for epoch_id in range(epoch_id_start, o.epoch_num):
            run_train_epoch(batch_id_start)
            batch_id_start = 0
    else:
        run_test_epoch()

    # print(tracks[69])
    # feed centroids to network in realtime
