import os
import sys
import numpy as np
import tensorflow as tf
import copy
import json
from json import JSONEncoder
from util import calc_centroid
from network import Network as Net

# np.set_printoptions(threshold=sys.maxsize)

class Tracker:
    '''
        Naive implementation of tracking
    '''
    def __init__(self):
        frames = [f for f in range(1, 71)]
        self.tracks = {key: [] for key in frames} #contains tracks for each frame
        self.counts = [] #the total number of tracks for each fram shoun equal these counts
        self.nextID = -1
        self.data = None

    def load_data(self, labled):
        # load data
        self.data = np.load(labled)
        print(self.data.shape)


    def label_initial_frame(self):
        '''
            Need to create a ground truth for tracking we will use segmentation labels
            to id the clusters in the first frame and use this as our ground truth.

            create tracks for the rest of the frames but only init location data

            Output: Will populate self.tracks with track objects of each cluster
        '''

        for t in range(70):
            print("time: ", t)
            # get frame
            timeSlice = self.data[..., t]
            
            # find unique clusters
            uniqueClusters = np.unique(timeSlice)
            # print(uniqueClusters)

            # create an track object for each clsuter and add to tracks list
            for clusterID in uniqueClusters:
                if clusterID == 0:
                    continue
                
                locations = np.argwhere(timeSlice == clusterID)
                
                if t == 0:
                    newTrack = Track(locations, clusterID, calc_centroid(locations), 'active', 'init')
                else:
                    newTrack = Track(locations, None, calc_centroid(locations), None, None)
                
                
                self.tracks[t].append(copy.deepcopy(newTrack))

        print('Total tracks: ', len(self.tracks))
        print(self.tracks)

        # with open('./tracks.json', 'w') as f:
        #     output_json = json.dumps(self.tracks)
        #     f.write(output_json)


    def get_clusters_per_frame(self):
        '''
            Calculate the numbers of clusters in each frame
        '''
        
        for z in range(70):
            timeSlice = self.data[..., z]
            uniqueClusters = np.unique(timeSlice)
            self.counts.append(len(uniqueClusters))

        print(self.counts)
        


    def id_next_frame(curr, next):
        '''
            Use centroid and union of segmentation results to determine id
            Yang did this but extended the borders essentailly

            works well here since no occlusion and clusters dont move much
            but not viable for other datasets
        '''
        




class Track:
    def __init__(self, locs, id, centroid, state, origin):
        self.locs = locs
        self.id = id
        self.centroid = centroid
        self.state = state
        self.origin = origin

    def __repr__(self):
        return "<Track \nlocs:%s \nid:%s \ncentroid:%s \nstate:%s \norigin:%s>" % (len(self.locs), self.id, self.centroid, self.state, self.origin)


# class TrackEncoder(JSONEncoder):
#     def default(self, o):
#         return o.__dict__

if __name__ == "__main__":
    # print(tf.config.list_physical_devices('GPU'))
    # print(tf.test.is_built_with_cuda())
    # print(tf.test.is_built_with_gpu_support())

    tracker = Tracker()

    tracker.load_data(labled="../../data/labled3data.npy")

    tracker.label_initial_frame()

    tracker.get_clusters_per_frame()
    # train_model()

    
