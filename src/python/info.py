import math
import numpy as np
import matplotlib.pyplot as plt

from util import load_json, load_pickle
from track import Track, Status

'''
    Validate the results of tracking and display tracking statistics
'''

def hist_pos_rates(f1_pos_rate_list_path, f2_pos_rate_list_path):
    f1_pos_rate_list = load_pickle(f1_pos_rate_list_path)
    f2_pos_rate_list = load_pickle(f2_pos_rate_list_path)

    # histogram of positive detection rates of partial clusters
    fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)
    axs[0].hist(f1_pos_rate_list, bins=100)
    axs[0].set_title('Cluster on Own Frame')
    axs[0].set_xlabel('Positive Detection Rates')
    axs[0].set_ylabel('Number of Clusters')
    axs[1].hist(f2_pos_rate_list, bins=100)
    axs[1].set_title('Cluster on Next Frame')
    axs[1].set_xlabel('Positive Detection Rates')
    plt.show()


def hist_sizes(cluster_size_list):
    plt.hist(cluster_size_list, bins=500)
    plt.title('Protein Cluster Sizes')
    plt.xlabel('Number of Points')
    plt.xticks(np.arange(0, 300, 5))
    plt.xlim(left=0, right=300)
    plt.ylabel('Number of Clusters')
    plt.show()


def event_counts(labeled_tracks_path):
    tracks = load_json(labeled_tracks_path)
    events = tracks['0']
    plt.title('Total Tracking Event Count')
    plt.bar(events.keys(), events.values(), align='center')
    plt.xlabel('Match Type')
    plt.ylabel('Number of Objects')
    plt.show()

def calc_results(TP, TN, FN, FP):
    '''
        Caluculate the results
    '''
    precision = TP / (TP / FP)
    sensitivity = TP / (TP / FN)
    accuracy = (TP + TN) / (TP + FP + TN + FN)
    print('Precision: ', precision)
    print('Sensitivity: ', sensitivity)
    print('Accuracy: ', accuracy)


