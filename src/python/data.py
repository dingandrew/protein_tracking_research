import os
import nibabel as nib
import numpy as np
import torch
import argparse
import copy
import pickle
import json
from track import Track, Status 
from util import save_counts_json, calc_centroid

# Parse arguments
parser = argparse.ArgumentParser(
    description='Create tracks for the segmentation data')
parser.add_argument('--dataset', required=True, choices=['protein', 'sprites', 'optim_param'],
                    help='Specify which dataset to prepare tracks for.')
args = parser.parse_args()


def create_dataset_protein(path):
    """
        Takes the .nii files and convertes it into numpy array, then 
        saves it in the same folder. Also populates the tracks and
        count dictionaries.

        Input: path, to directery containing .nii files
    """
    validData = ["Weights", "Fullsize_", "Fullsize_label_", ".nii"]

    # enter sub-directory of time slices
    for frame_num in os.listdir(path):
        print(frame_num, type(frame_num))
        data_path = os.path.join(path, frame_num)

        # get files from each time slice
        for f in os.listdir(data_path):
            ext = os.path.splitext(f)[1]
            name = os.path.splitext(f)[0]
            if ext.lower() not in validData:
                continue

            full_data_path = os.path.join(data_path, f)
            
            if validData[0] in name:
                print(full_data_path)
                data_seg = nib.load(full_data_path)
                print(data_seg.dataobj[...].shape, os.path.join(data_path, name))
                np.save(os.path.join(data_path, name),
                        data_seg.dataobj[...])

            if validData[1] in name:
                print(full_data_path)
                data_full = nib.load(full_data_path)
                print(data_full.dataobj[...].shape,
                      os.path.join(data_path, name))
                np.save(os.path.join(data_path, name),
                        data_full.dataobj[...])

            if validData[2] in name:
                print(full_data_path)
                data_lbl = nib.load(full_data_path)
                print(data_lbl.dataobj[...].shape,
                      os.path.join(data_path, name))
                np.save(os.path.join(data_path, name),
                        data_lbl.dataobj[...])
        
                tracks[int(frame_num)] = init_frame_tracks(int(frame_num), 
                                                           data_lbl.dataobj[...])

                counts[int(frame_num)] = get_clusters_per_frame(data_lbl.dataobj[...])
        

def create_dataset_sprites(path):
    full_train_c = torch.IntTensor(96, 20, 3, 128, 128)


    for fnum in range(0, 96):
        with open(sprites + 'pt/input/train_{}.pt'.format(fnum), 'rb') as f:
            img = torch.load(f)
            img = img.type(torch.int)
            full_train_c[fnum, ...] = img[0, ...]

    print(full_train_c.shape)
    with open('../../data/sprites-MOT/sprite/pt/full_train_c.pt', 'wb') as f:
        # Pickle the 'data' dictionary using the highest protocol available.
        np.save(f, full_train_c)



    validData = ["Weights", "Fullsize_", "Fullsize_label_", ".nii"]

    # enter sub-directory of time slices
    for frame_num in os.listdir(path):
        print(frame_num, type(frame_num))
        data_path = os.path.join(path, frame_num)

        # get files from each time slice
        for f in os.listdir(data_path):
            ext = os.path.splitext(f)[1]
            name = os.path.splitext(f)[0]
            if ext.lower() not in validData:
                continue

            full_data_path = os.path.join(data_path, f)

            if validData[0] in name:
                print(full_data_path)
                data_seg = nib.load(full_data_path)
                print(data_seg.dataobj[...].shape,
                      os.path.join(data_path, name))
                np.save(os.path.join(data_path, name),
                        data_seg.dataobj[...])

            if validData[1] in name:
                print(full_data_path)
                data_full = nib.load(full_data_path)
                print(data_full.dataobj[...].shape,
                      os.path.join(data_path, name))
                np.save(os.path.join(data_path, name),
                        data_full.dataobj[...])

            if validData[2] in name:
                print(full_data_path)
                data_lbl = nib.load(full_data_path)
                print(data_lbl.dataobj[...].shape,
                      os.path.join(data_path, name))
                np.save(os.path.join(data_path, name),
                        data_lbl.dataobj[...])

                tracks[int(frame_num)] = init_frame_tracks(int(frame_num),
                                                           data_lbl.dataobj[...])

                counts[int(frame_num)] = get_clusters_per_frame(
                    data_lbl.dataobj[...])

def init_frame_tracks(frame_num, frame):
    '''
        Need to create track object for each object we will use segmentation labels
        to find the clusters in each frame.

        Arbitrarily assigns ID's to objects in the first frame, for initial
        tracking.

        Will populate tracks with track objects of each frame.
    '''
    frameTracks = []

    # find unique clusters
    uniqueClusters = np.unique(frame)
    # create an track object for each clsuter and add to tracks list
    for clusterID in uniqueClusters:
        if clusterID == 0:
            continue

        locations = np.argwhere(frame == clusterID)

        if frame_num == 1:
            newTrack = Track(locs=locations,
                            id=clusterID,
                            centroid=calc_centroid(locations),
                            state='active',
                            origin='init ')
        else:
            newTrack = Track(locs=locations,
                            id=None,
                            centroid=calc_centroid(locations),
                            state='',
                            origin='')

        frameTracks.append(copy.deepcopy(newTrack))

    return frameTracks

def get_clusters_per_frame(frame):
    '''
        Calculate the numbers of clusters in the frame.
    '''
    uniqueClusters = np.unique(frame)
    # subtract 1, dont count 0 pixel values
    return len(uniqueClusters) - 1


if __name__ == "__main__":
 
    if args.dataset == 'protein':
        base_dir = '../../data/raw_data/Segmentation_and_result'
        save_dir = '../../data'
        create_dataset = create_dataset_protein
        tracks = {key: [] for key in range(1, 71)}  # contains tracks for each frame

    elif args.dataset == 'sprites':
        base_dir = '../../data/raw_data/sprites-MOT'
        save_dir = '../../data/sprites-MOT/sprite/pt'
        create_dataset = create_dataset_sprites
        tracks = {key: [] for key in range(0, 1920)}

    counts = {}

    print("-------- Creating Data Set from raw data ---------")
    print("\t   !!! Will bottleneck disk !!!")
    create_dataset(base_dir)

    print("----------- Saving the Tracks/Count --------------")
    with open('../../data/tracks_protein.pickle', 'wb') as f:
        pickle.dump(tracks, f, pickle.HIGHEST_PROTOCOL)
    with open('../../data/counts.json', 'w') as f:
        json.dump(counts, f, indent=4)
  

