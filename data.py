import os
import nibabel as nib
import matplotlib.pyplot as plt
from scipy.io import loadmat
import sys
import numpy as np

# np.set_printoptions(threshold=sys.maxsize)

base_dir = "./data/raw_data/Segmentation_and_result"

def create_dataset(path=base_dir):
    """
        Create the 3d data set containing the sementation results and unlabled images
        Takes the raw *.nii format and convertes it into numpy array
        output is indexed as (width, height, z-stack, time slice)

        Input: path, to directery containing .nii files

        Outputs : labled_3data(280, 512, 13, 70), raw_3data(248, 512, 13, 70)
    """
    labled3data = np.ndarray((280, 512, 13, 70))
    raw3data = np.ndarray((280, 512, 13, 70))
    validData = ["Fullsize_label", "Fullsize", ".nii"]

    # enter sub-directory of time slices
    for d in os.listdir(path):
        data_path = os.path.join(path, d)

        # get the time index for the dataset
        # minus 1 since since files numbering start at 1
        timeIndex = int(d) - 1

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
                labled3data[..., timeIndex] = data_seg.dataobj[...]

            elif validData[1] in name:
                print(full_data_path)
                data_full = nib.load(full_data_path)
                raw3data[..., timeIndex] = data_full.dataobj[...]

    print('labled shape: ', labled3data.shape)
    print('raw shape: ', raw3data.shape)

    return labled3data, raw3data


def verify_data(labled, raw, path=base_dir):
    """
        Double check data is still correct after saving and 
        reloading, by comparing to original

        Output true or false
    """
    labled3data = np.load(labled)
    raw3data = np.load(raw)
    print('labled shape: ', labled3data.shape)
    print('raw shape: ', raw3data.shape)

    validData = ["Fullsize_label", "Fullsize", ".nii"]

    # enter sub-directory of time slices
    for d in os.listdir(path):
        data_path = os.path.join(path, d)

        # get the time index for the dataset
        # minus 1 since since files numbering start at 1
        timeIndex = int(d) - 1

        # get files from each time slice
        for f in os.listdir(data_path):
            ext = os.path.splitext(f)[1]
            name = os.path.splitext(f)[0]

            if ext.lower() not in validData:
                continue

            full_data_path = os.path.join(data_path, f)

            if validData[0] in name:
                data_seg = nib.load(full_data_path)
                if not np.array_equal(data_seg.dataobj[...], labled3data[..., timeIndex]):
                    return False
            elif validData[1] in name:
                data_full = nib.load(full_data_path)
                if not np.array_equal(data_full.dataobj[...], raw3data[..., timeIndex]):
                    return False

    return True


def save_data(dataset, path):
    """
        Save the data as .npy
    """
    np.save(path, dataset, allow_pickle=False)
    print('Saved')


def extract_weights_from_nii(path=base_dir):
    """
        Takes the raw weights.nii format and convertes it into numpy array

        Input: path, to directery containing .nii files

        Outputs : saves the weights as numpy array(280, 512, 13, 64)
    """

    validData = ["Weights", ".nii"]

    # enter sub-directory of time slices
    for d in os.listdir(path):
        data_path = os.path.join(path, d)

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
            

if __name__ == "__main__":
    print("--------Creating Data Set from raw data---------")
    labled3data, raw3data = create_dataset()

    print("--------Saving the dataset----------------------")
    save_data(labled3data, "./data/labled3data")
    save_data(raw3data, "./data/raw3data")

    print("--------Verifying the data----------------------")
    if verify_data(labled="./data/labled3data.npy", raw="./data/raw3data.npy") is False:
        print("Very bad")
    else:
        print("OK")
    print("-------- Extracting the weights----------------------")
    print("\tWill bottleneck disk")
    extract_weights_from_nii()
