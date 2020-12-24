import math
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from util import load_json, load_pickle, save_json, calc_euclidean_dist
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






def get_track_history(id, labeled_tracks_path, frame_track_path):
    labeled_tracks = load_json(labeled_tracks_path)
    frame_tracks = load_json(frame_track_path)
    curr_frame_num = 1
    curr_id = id
    track = None
    results = {key: None for key in range(1, 71)}
    alive = True

    # G = nx.DiGraph()
    # pos = nx.multipartite_layout(G)
    # prev = 'Start'
    # G.add_node(prev)
    while alive:
        tracks = labeled_tracks[str(curr_id)]
        first_track = tracks[0]
        last_track = tracks[-1]
        start_frame = first_track['Frame']
        last_frame_num = tracks[-1]['Frame']

        # print(curr_id)
        # print(start_frame, last_frame_num)
        for frame_num in range(start_frame, last_frame_num + 1):
            if results[frame_num] is None:
                results[frame_num] = 'MATCH ' + str(curr_id)
                # new_node = str(frame_num) + ' : ' + str(curr_id)
                # G.add_edge(prev, new_node)
                # prev = new_node
   

        # check the last object in the tracks list if it has a forward_conf
        # then check if it is a split or merge or death
        if tracks[-1]['forward_conf'] > 0:
            event, curr_id = find_next_track(
                curr_id, frame_tracks[str(last_frame_num + 1)], labeled_tracks)
            results[last_frame_num + 1] = event
            # new_node = str(frame_num + 1) + ' : ' + str(curr_id)
            # G.add_edge(prev, new_node)
            # prev = new_node
        else:
            results[last_frame_num + 1] = 'DEAD'
            alive = False  
            # new_node = str(last_frame_num + 1) + ' : ' + str(curr_id)
            # G.add_edge(prev, new_node)
       
        
    save_json(results, './data/test/{}/{}results.json'.format(int(float(id)), int(float(id))))

    # nx.draw_shell(G, with_labels=True, font_weight='bold')
    # plt.show()

def find_next_track(curr_id, frame_search, labeled_tracks):
    matched = []
    for track in frame_search:
        if str(curr_id) in track['origin'].split():
            matched.append(track)

    
    # the new parent track uses the longest living object
    track = max(matched, key=lambda t: len(labeled_tracks[str(t['id'])]))
    curr_id = track['id']

    event = ''
    if len(matched) == 1:
        event = matched[0]['origin'] + ' => ' + str(curr_id)
    elif len(matched) > 1:
        for t in matched:
            event += t['origin']

        event = event + ' => ' + str(curr_id)
    elif len(matched) == 0:
        print('Error')
        exit()


    return event, curr_id 




if __name__ == "__main__":
    print('Test')
    labeled_tracks_path = './data/tracks_protein_pretty.json'
    frame_track_path = './data/tracks_protein_frame.json'
    # get_track_history('171.0', labeled_tracks_path, frame_track_path)
    # get_track_history('130.0', labeled_tracks_path, frame_track_path)
    # get_track_history('258.0', labeled_tracks_path, frame_track_path)
    # get_track_history('205.0', labeled_tracks_path, frame_track_path)

    # get_track_history('209.0', labeled_tracks_path, frame_track_path)
    # get_track_history('141.0', labeled_tracks_path, frame_track_path)
    # get_track_history('178.0', labeled_tracks_path, frame_track_path)
    # get_track_history('5.0', labeled_tracks_path, frame_track_path)
    # get_track_history('101.0', labeled_tracks_path, frame_track_path)
    # get_track_history('142.0', labeled_tracks_path, frame_track_path)
    # get_track_history('94.0', labeled_tracks_path, frame_track_path)
    # get_track_history('98.0', labeled_tracks_path, frame_track_path)
    # get_track_history('131.0', labeled_tracks_path, frame_track_path)
    # get_track_history('204.0', labeled_tracks_path, frame_track_path)
    # get_track_history('62.0', labeled_tracks_path, frame_track_path)
    # get_track_history('161.0', labeled_tracks_path, frame_track_path)
    # get_track_history('162.0', labeled_tracks_path, frame_track_path)

    get_track_history('4064.0', labeled_tracks_path, frame_track_path)
