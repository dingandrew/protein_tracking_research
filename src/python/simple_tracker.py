import numpy as np
import pickle
import copy
from util import calc_centroid, calc_euclidean_dist, save_as_json, save_counts_json
from track import Track
from tqdm import tqdm


class Tracker:
    '''
        Naive implementation of tracking
    '''

    def __init__(self):
        self.CENTROID_DIST_THRESH = 7
        self.INTERSECT_NUM_THRESH = 5

    def id_clusters(self, tracks):
        '''
            Use centroid and union of segmentation results to determine id.
            Very similar to what Yang did.

            Works here since there is no occlusion and clusters dont move much
            , but may not be viable for other datasets.

            Return: self.labledTracks a track object for each cluster that contains loc data
                    for each frame
        '''
        with open(tracks, 'rb') as f:
            # The protocol version used is detected automatically, so we do not
            # have to specify it.
            tracks = pickle.load(f)

        # retreive the largest id from first frame and increment this will be the next id
        nextID = max(track.id for track in tracks[1]) + 1
        # iterate through all frames, note we are checking next frame
        for currFrame in tqdm(range(1, 70)):
            # check centroid distance in the next frame
            for currTrack in tracks[currFrame]:
                # get current track info
                currId = currTrack.id
                currCentroid = currTrack.centroid
                currState = currTrack.state
                currOrigin = currTrack.origin
                found = False
                for nextTrack in tracks[currFrame + 1]:
                    if nextTrack.id is None:
                        if nextTrack.centroid == currCentroid \
                                or calc_euclidean_dist(currCentroid, nextTrack.centroid) < self.CENTROID_DIST_THRESH:
                            nextTrack.id = currId
                            nextTrack.state = currState
                            nextTrack.origin = currOrigin
                            found = True
                            break  # found 1to1 can break
                    elif nextTrack.id == currId:
                        print("should never happen")
                    else:
                        pass

                # did not find the 1to1 cluster in next frame this track is dead
                if not found:
                    currTrack.state = 'dead'

            # remaining unlabled clusters in next frame are all the result
            # of birth, split, or merge
            for nextTrack in tracks[currFrame + 1]:
                if nextTrack.id is None:
                    intersections = self.has_intersection(
                        nextTrack.locs, tracks[currFrame])
                    if len(intersections) == 1:
                        # split, there is only one intersecting cluster from prev frame
                        nextTrack.id = nextID
                        nextID += 1
                        nextTrack.state = 'active'
                        nextTrack.origin = 'split from: ' + \
                            str([tracks[currFrame]
                                 [indx].id for indx in intersections])
                    elif len(intersections) > 1:
                        # merge, there is more than one intersecting cluster from prev frame
                        nextTrack.id = nextID
                        nextID += 1
                        nextTrack.state = 'active'
                        nextTrack.origin = 'merge from: ' + \
                            str([tracks[currFrame]
                                 [indx].id for indx in intersections])
                    else:
                        # this cluster is birthed in this frame
                        nextTrack.id = nextID
                        nextID += 1
                        nextTrack.state = 'active'
                        nextTrack.origin = 'birth'

        with open('../../data/labeled_tracks.pickle', 'wb') as f:
            # Pickle the 'data' dictionary using the highest protocol available.
            pickle.dump(tracks, f, pickle.HIGHEST_PROTOCOL)

        save_as_json(tracks, self.CENTROID_DIST_THRESH,
                     self.DIST_WEIGHTS, self.INTERSECT_NUM_THRESH)

    def has_intersection(self, cluster, search_frame):
        '''
            Check if given point clusters have any intersection

            Return: list of indexes of tracks that intersect with the cluster
        '''
        intersections = []
        for index, track in enumerate(search_frame):
            if np.count_nonzero(cluster == track.locs) > self.INTERSECT_NUM_THRESH:
                intersections.append(index)

        return intersections


if __name__ == "__main__":
    tracker = Tracker()
    print("----------- Tracks clusters of all frames ------------")
    tracker.id_clusters('../../data/tracks.pickle')
