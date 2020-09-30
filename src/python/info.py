import math
import numpy as np
import matplotlib.pyplot as plt

from util import load_json, load_pickle, save_json
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


def hist_sizes(cluster_size_list_path):
    cluster_size_list = load_pickle(cluster_size_list_path)
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


def tracking_results(labeled_tracks_path, mappings_json_path):






    exit()


    

    clusters = [id for id in range(1, 41)]
    tracking = tracking[:, 0:40]

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
        for x in range(len(clusters) ):
            text = ax.text(x, y, round(tracking[y, x], 3),
                        ha="center", va="center", color="w")

    ax.set_title("Tracking results")
    ax.set_xlabel("Cluster ID")
    plt.show()


