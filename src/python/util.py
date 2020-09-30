import scipy.ndimage as ndimage
import cv2 as cv
import shutil
import os.path as path
import os
import numpy as np
from math import sqrt
import pickle
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
    sumX = np.sum(arr[:, 0])
    sumY = np.sum(arr[:, 1])
    sumZ = np.sum(arr[:, 2])
    return [sumX/length, sumY/length, sumZ/length]


def calc_euclidean_dist(p1, p2):
    '''
        Calculate the weighted euclidean distace between 2 points in 3D/2D

        Return: float distance
    '''
    if len(p1) == 3:
        return sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2 + (p1[2] - p2[2])**2)
    elif len(p1) == 2:
        return sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    else:
        return Exception('Invalid Dimensions for Centroid Coordinates')

def save_tracks_as_json(name, tracks, DEATH_COUNT, ONE_TO_ONE_COUNT, SPLIT_COUNT, MERGE_COUNT, BIRTH_COUNT):
    '''
        Save tracking results in a json format
    '''
    json_pretty = {}
    json_frame_format = {}

    json_pretty[0] = {'DEATH_COUNT': DEATH_COUNT, 
                      'ONE_TO_ONE_COUNT': ONE_TO_ONE_COUNT, 
                      'SPLIT_COUNT': SPLIT_COUNT,
                      'MERGE_COUNT': MERGE_COUNT,
                      'BIRTH_COUNT': BIRTH_COUNT}
    json_frame_format[0] = {'DEATH_COUNT': DEATH_COUNT,
                            'ONE_TO_ONE_COUNT': ONE_TO_ONE_COUNT,
                            'SPLIT_COUNT': SPLIT_COUNT,
                            'MERGE_COUNT': MERGE_COUNT,
                            'BIRTH_COUNT': BIRTH_COUNT}

    for currFrame in range(1, 71):
        if currFrame not in json_frame_format.keys():
            json_frame_format[currFrame] = []

        for track in tracks[currFrame]:
            if track.id not in json_pretty.keys():
                json_pretty[track.id] = []

            json_pretty[track.id].append({'Frame': currFrame, 'locs': len(track.locs),
                                          'centroid': track.centroid, 'state': track.state, 'origin': track.origin})
            json_frame_format[currFrame].append({'id': track.id, 'locs': track.locs.tolist(),
                                                 'centroid': track.centroid, 'state': track.state, 'origin': track.origin})

    save_json(json_pretty, '../../data/tracks_{}_pretty.json'.format(name))
    save_json(json_frame_format, '../../data/tracks_{}_frame.json'.format(name))


def save_json(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)


def load_json(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    return data


def save_pickle(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)


def load_pickle(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data


def mkdir(dir):
    if not path.exists(dir):
        os.makedirs(dir)


def rmdir(dir):
    if path.exists(dir):
        print('Directory ' + dir + ' is removed.')
        shutil.rmtree(dir)
