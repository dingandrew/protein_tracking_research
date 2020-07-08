import os
import sys
import numpy as np
import pickle
import tensorflow as tf
import copy
import json
from json import JSONEncoder
from util import calc_centroid, calc_euclidean_dist, save_as_json
from network import Network as Net
import time
from tqdm import tqdm

# np.set_printoptions(threshold=sys.maxsize)


class Tracker:
    '''
        Naive implementation of tracking
    '''

    def __init__(self):
        frames = [f for f in range(1, 71)]
        self.tracks = {key: []
                       for key in frames}  # contains tracks for each frame
        self.counts = []  # the total number of tracks for each fram shoun equal these counts
        self.nextID = 1
        self.data = None
        self.labledTracks = {}

        self.CENTROID_DIST_THRESH = 2
        self.DIST_WEIGHTS = [1, 1, 1]

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

        for t in tqdm(range(1, 71)):
            # get frame
            timeSlice = self.data[..., t-1]

            # find unique clusters
            uniqueClusters = np.unique(timeSlice)
            # print(uniqueClusters)

            # create an track object for each clsuter and add to tracks list
            for clusterID in uniqueClusters:
                if clusterID == 0:
                    continue

                locations = np.argwhere(timeSlice == clusterID)

                if t == 1:
                    newTrack = Track(locations, clusterID, calc_centroid(
                        locations), 'active', 'init')
                else:
                    newTrack = Track(
                        locations, None, calc_centroid(locations), None, None)

                self.tracks[t].append(copy.deepcopy(newTrack))

        print('Total tracks: ', len(self.tracks))
        print(self.tracks)

        with open('../../data/tracks.pickle', 'wb') as f:
            # Pickle the 'data' dictionary using the highest protocol available.
            pickle.dump(self.tracks, f, pickle.HIGHEST_PROTOCOL)

    def get_clusters_per_frame(self):
        '''
            Calculate the numbers of clusters in each frame
        '''

        for z in range(70):
            timeSlice = self.data[..., z]
            uniqueClusters = np.unique(timeSlice)
            self.counts.append(len(uniqueClusters))

    def id_clusters(self, pickled_data=False):
        '''
            Use centroid and union of segmentation results to determine id.
            Very similar to what Yang did.

            Works well here since there is no occlusion and clusters dont move much
            , but may not be viable for other datasets.

            Return: self.labledTracks a track object for each cluster that contains loc data
                    for each frame
        '''

        if pickled_data:
            with open('../../data/tracks.pickle', 'rb') as f:
                # The protocol version used is detected automatically, so we do not
                # have to specify it.
                self.tracks = pickle.load(f)
        
        # retreive the largest id from first frame and increment this will be the next id
        self.nextID = max(track.id for track in self.tracks[1]) + 1
        print(self.nextID)

        # iterate through all frames, note we are checking next frame
        for currFrame in tqdm(range(1, 70)):

            # check centroid distance in the next frame
            for currTrack in self.tracks[currFrame]:

                # get current track info
                currId = currTrack.id
                currCentroid = currTrack.centroid
                currState = currTrack.state
                currOrigin = currTrack.origin
                found = False

                for nextTrack in self.tracks[currFrame + 1]:
                    if nextTrack.id is None:
                        if nextTrack.centroid == currCentroid \
                        or calc_euclidean_dist(currCentroid, nextTrack.centroid, self.DIST_WEIGHTS) < self.CENTROID_DIST_THRESH:

                            nextTrack.id = currId
                            nextTrack.state = currState
                            nextTrack.origin = currOrigin
                            found = True
                            break #found 1to1 can break
                    elif nextTrack.id == currId:
                        print("should never happen")
                    else:
                        pass

                #did not find the 1to1 cluster in next frame this track is dead
                if not found:
                    currTrack.state = 'dead'


            # remaining unlabled clusters in next frame are all the result
            # of birth, split, or merge
            for nextTrack in self.tracks[currFrame + 1]:
                if nextTrack.id is None:
                    intersections = self.has_intersection(nextTrack.locs, self.tracks[currFrame])
                    if len(intersections) == 1:
                        #split, there is only one intersecting cluster from prev frame
                        nextTrack.id = self.nextID
                        self.nextID += 1
                        nextTrack.state = 'active'
                        nextTrack.origin = 'split from: ' + \
                            str([self.tracks[currFrame][indx].id for indx in intersections])
                    elif len(intersections) > 1:
                        #merge, there is more than one intersecting cluster from prev frame
                        nextTrack.id = self.nextID
                        self.nextID += 1
                        nextTrack.state = 'active'
                        nextTrack.origin = 'merge from: ' + \
                            str([self.tracks[currFrame][indx].id for indx in intersections])
                    else:
                        #this cluster is birthed in this frame
                        nextTrack.id = self.nextID
                        self.nextID += 1
                        nextTrack.state = 'active'
                        nextTrack.origin = 'birth'
        
        with open('../../data/labeled_tracks.pickle', 'wb') as f:
            # Pickle the 'data' dictionary using the highest protocol available.
            pickle.dump(self.tracks, f, pickle.HIGHEST_PROTOCOL)

        save_as_json(self.tracks)

    def has_intersection(self, cluster, search_frame):
        '''
            Check if given point clusters have any intersection

            Return: list of indexes of tracks that intersect with the cluster
        '''
        intersections = [] 
        for index, track in enumerate(search_frame):
            # if np.count_nonzero(cluster == track.locs) > 0:
            #     intersections.append(index)
            if len(np.intersect1d(cluster, track.locs)) > 0:
                intersections.append(index)
            
        return intersections


class Track:
    def __init__(self, locs, id, centroid, state, origin):
        self.locs = locs
        self.id = id
        self.centroid = centroid
        self.state = state
        self.origin = origin

    def __repr__(self):
        return "<Track \nlocs:%s \nid:%s \ncentroid:%s \nstate:%s \norigin:%s>" % (len(self.locs), self.id, self.centroid, self.state, self.origin)


if __name__ == "__main__":
    # print(tf.config.list_physical_devices('GPU'))
    # print(tf.test.is_built_with_cuda())
    # print(tf.test.is_built_with_gpu_support())

    tracker = Tracker()
    print("----------- Load labled data set ------------")
    # tracker.load_data(labled="../../data/labled3data.npy")
    print("----------- Label ID's of initial frame ------------")
    # tracker.label_initial_frame()
    print("----------- Number of clusters in each frame ------------")
    # tracker.get_clusters_per_frame()
    print("----------- Tracks clusters of all frames ------------")
    tracker.id_clusters(pickled_data=True)


