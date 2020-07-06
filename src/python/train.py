import os
import matplotlib.pyplot as plt
import sys
import numpy as np
import tensorflow as tf
import copy
from util import calc_centroid
from network import Network as Net

# np.set_printoptions(threshold=sys.maxsize)

class Trainer:
    def __init__(self):
        self.tracks = []
        self.nextID = -1

    def train_model(self):
        net = Net(batch_size=10, time_step=70)

        model = net.build_network()

        model.compile(loss=['mse'])

    def label_initial_frame(self, labled):
        '''
            Need to create a ground truth for training we will use segmentation labels
            to id the clusters in the first frame and use this as our ground truth.

            Output: Will populate self.tracks with track objects of each cluster
        '''
        # load data
        labled3data = np.load(labled)
        print(labled3data.shape)

        # get first frame
        timeSlice1 = labled3data[..., 0]
        print(timeSlice1.shape)

        # find unique clusters
        uniqueClusters = np.unique(timeSlice1)
        # print(uniqueClusters)

        # create an track object for each clsuter and add to tracks list
        for clusterID in uniqueClusters:
            if clusterID == 0:
                continue

            locations = np.argwhere(timeSlice1 == clusterID)
            newTrack = Track(locations, clusterID, calc_centroid(locations), 'active', 'init')
            self.tracks.append(copy.deepcopy(newTrack))

        print(self.tracks[5])
        print('Total tracks: ', len(self.tracks))

    def get_clusters_per_frame(self, labled):
        '''
            Calculate the numbers of clusters in each frame
        '''
        # load data
        labled3data = np.load(labled)

        counts = []
        
        for z in range(70):
            timeSlice = labled3data[..., z]
            uniqueClusters = np.unique(timeSlice)
            counts.append(len(uniqueClusters))

        print(counts)



class Track:
    def __init__(self, locs, id, centroid, state, origin):
        self.locs = locs
        self.id = id
        self.centroid = centroid
        self.state = state
        self.origin = origin

    def __repr__(self):
        return "<Track \nlocs:%s \nid:%s \ncentroid:%s \nstate:%s \norigin:%s>" % (self.locs, self.id, self.centroid, self.state, self.origin)


if __name__ == "__main__":
    print(tf.config.list_physical_devices('GPU'))
    print(tf.test.is_built_with_cuda())
    print(tf.test.is_built_with_gpu_support())

    trainer = Trainer()

    trainer.label_initial_frame(labled="../../data/labled3data.npy")

    trainer.get_clusters_per_frame(labled="../../data/labled3data.npy")
    # train_model()
