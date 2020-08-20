from track import Track
import os
import sys
import numpy as np
import time
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

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


    def compare_methods(self):
        methods = ["DEEP: F1", "DEEP: F2", "TRACK: F1", "TRACK: F2"]
        clusters = [id for id in range(1, 41)]
    
        # tracking = np.random.rand(4, 40)

        with open('../../data/prediction.npy', 'rb') as f:
            tracking = np.load(f)
            
        tracking = tracking[:, 0:41]
        
        for frame in range(1, 3):
            for track in self.labeledTracks[frame]:
                # print(track.id)
                if int(track.id) in clusters:
                    tracking[frame + 1, int(track.id) - 1] = 1

        # print(tracking)

        fig, ax = plt.subplots()
        fig.subplots_adjust(left=0.05, right=0.99)
        im = ax.imshow(tracking, norm=colors.Normalize(vmin=0, vmax=1))

        # We want to show all ticks...
        ax.set_xticks(np.arange(len(clusters)))
        ax.set_yticks(np.arange(len(methods)))
        # ... and label them with the respective list entries
        ax.set_xticklabels(clusters)
        ax.set_yticklabels(methods)

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        for y in range(len(methods)):
            for x in range(len(clusters) + 1):
                text = ax.text(x, y, round(tracking[y, x], 3),
                            ha="center", va="center", color="w")

        ax.set_title("Tracking results")
        ax.set_xlabel("Cluster ID")
        plt.show()



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
    test.compare_methods()
