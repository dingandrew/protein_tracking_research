import scipy.ndimage as ndimage
import cv2 as cv
import shutil
import os.path as path
import os
import numpy as np
from math import sqrt
import json
import matplotlib.pyplot as plt
import torch
'''
    Some utility functions 

'''

def calc_centroid(arr):
    '''
        Calculate centrioid of a point cluster
    '''
    length = arr.shape[0]

    # print(length)
    # print(arr)
    # print(arr[:, 0])
    # print(arr[:, 1])
    # print(arr[:, 2])

    sumX = np.sum(arr[:, 0])
    sumY = np.sum(arr[:, 1])
    sumZ = np.sum(arr[:, 2])
    return [sumX/length, sumY/length, sumZ/length]


def calc_euclidean_dist(p1, p2, weights):
    '''
        Calculate the weighted euclidean distace between 2 points in 3D
        Need to be weighted since shape is (280, 512, 13)

        Return: float distance
    '''
    return sqrt(weights[0] * (p1[0] - p2[0])**2 + weights[1] * (p1[1] - p2[1])**2 + weights[2] * (p1[2] - p2[2])**2)

def has_intersection(cluster, search_frame):
    '''
        Check if given point clusters have any intersection

        Return: list of indexes of tracks that intersect with the cluster
    '''
    intersections = []
    for index, track in enumerate(search_frame):
        if np.count_nonzero(cluster == track.locs) > 10:
            intersections.append(index)
        # if len(np.intersect1d(cluster, track.locs)) > 0:
        #     intersections.append(index)

    return intersections

def save_as_json(tracks, centroid_thresh, weights, inter_thresh):
    '''
        Save tracking results in a json format
    '''
    json_data = {}
    json_pretty = {}
    json_frame_format = {}

    json_data[0] = {
        'centroid_thresh': centroid_thresh, 'weights': weights, 'inter_thresh': inter_thresh}
    json_pretty[0] = {
        'centroid_thresh': centroid_thresh, 'weights': weights, 'inter_thresh': inter_thresh}
    json_frame_format[0] = {
        'centroid_thresh': centroid_thresh, 'weights': weights, 'inter_thresh': inter_thresh}

    for currFrame in range(1, 71):
        if currFrame not in json_frame_format.keys():
            json_frame_format[currFrame] = []

        for track in tracks[currFrame]:
            if track.id not in json_data.keys():
                json_data[track.id] = []
                json_pretty[track.id] = []
            json_data[track.id].append({'Frame': currFrame, 'locs': track.locs.tolist(),
                                        'centroid': track.centroid, 'state': track.state, 'origin': track.origin})
            json_pretty[track.id].append({'Frame': currFrame, 'locs': len(track.locs),
                                          'centroid': track.centroid, 'state': track.state, 'origin': track.origin})
            json_frame_format[currFrame].append({'id': track.id, 'locs': track.locs.tolist(),
                                                'centroid': track.centroid, 'state': track.state, 'origin': track.origin})

    with open('../../data/tracks.json', 'w') as f:
        json.dump(json_data, f, indent=4)
    with open('../../data/tracks_pretty.json', 'w') as f:
        json.dump(json_pretty, f, indent=4)
    with open('../../data/tracks_frame.json', 'w') as f:
        json.dump(json_frame_format, f, indent=4)

def save_counts_json(counts):
    '''
        Save counts as json
    '''
    with open('../../data/counts.json', 'w') as f:
        json.dump(counts, f, indent=4)


def open_model_json(path):
    '''
        Open the model_config.json file
    '''
    with open(path, 'r') as json_file:
        params = json.load(json_file)

    return params


def save_json(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f)


def load_json(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    return data


def mkdir(dir):
    if not path.exists(dir):
        os.makedirs(dir)


def rmdir(dir):
    if path.exists(dir):
        print('Directory ' + dir + ' is removed.')
        shutil.rmtree(dir)
