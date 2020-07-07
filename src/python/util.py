import numpy as np

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


