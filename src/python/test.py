from tracker import Track
import os
import sys
import numpy as np
import time
import json
import pickle


class Tester:
    '''
        Verify the results of tracking
    '''

    def __init__(self):
        # self.tracker_agent = tracker.Tracker()
        # self.tracker_agent.load_data()
        self.labeledTracks = {}
        self.counts = {}
        self.jsonTracks = {}
        # load results from tracking
        self.load_results()

    def test_verify_active_tracks(self):
        '''
            Verify that each frame has the correct number of active tracks
            and frames do not contain unlabled clusters.

            This is just a litmus test that the algorithm is working doesn't test
            accuracy.

            Return: PASS or FAIL
        '''
        print('Verify that each frame has the correct number of active tracks\nand frames do not contain unlabled clusters')
        noID = 0
        frames = [f for f in range(1, 71)]
        resultCount = {key: 0
                       for key in frames}
        errorFrames = {}

        for frame in range(1, 71):
            for track in self.labeledTracks[frame]:
                if track.id is None:
                    noID += 1

        for clusterID in self.jsonTracks.keys():
            if clusterID != '0':
                for frame in self.jsonTracks[clusterID]:
                    resultCount[frame['Frame']] += 1
        
        for frame in range(1, 71):
            if resultCount[frame] != self.counts[str(frame)]:
                errorFrames[frame] = resultCount[frame] - self.counts[str(frame)]

        if noID > 0 or len(errorFrames.values()) > 0:
            print('\tFAILED')
            print(noID, ' ', errorFrames)
        else:
            print('\tPASS')


    def test_reverse_tracking(self):
        ''' 
            Run the tracker backwards to verify results

        '''        



    def load_results(self, path1='../../data/labeled_tracks.pickle', 
                           path2='../../data/counts.json', 
                           path3='../../data/tracks.json'):
        '''
            Load tracking results
        '''
        with open(path1, 'rb') as f:
            # The protocol version used is detected automatically, so we do not
            # have to specify it.
            self.labeledTracks = pickle.load(f)

        with open(path2, 'r') as f:
            self.counts = json.load(f)

        with open(path3, 'r') as f:
            self.jsonTracks = json.load(f)

    def calc_results(self, TP, TN, FN, FP):
        '''
            Caluculate the results
        '''
        precision = TP / (TP / FP)
        sensitivity = TP / (TP / FN)
        accuracy = (TP + TN) / (TP + FP + TN + FN)
        print('Precision: ', precision)
        print('Sensitivity: ', sensitivity)
        print('Accuracy: ', accuracy)


if __name__ == "__main__":
    test = Tester()
    test.test_verify_active_tracks()
