import os
import nibabel as nib
import matplotlib.pyplot as plt
from scipy.io import loadmat
import sys
import pandas as pd
import numpy as np

# np.set_printoptions(threshold=sys.maxsize)

base_dir = "./data/raw_data/Segmentation_and_result"


def explore_data():
    """
        Some code to explore the data properties
    """
    file_path1 = os.path.join(base_dir, "1/Fullsize_1.nii")
    file_path2 = os.path.join(base_dir, "1/Fullsize_label_1.nii")
    file_path3 = os.path.join(base_dir, "1/Registration_1.nii")
    file_path4 = os.path.join(base_dir, "1/Weights_1.nii")
    file_path5 = os.path.join(base_dir, "1/1_3Dconnection2.fig")

    data1 = nib.load(file_path1)
    label = nib.load(file_path2)
    reg = nib.load(file_path3)
    weight = nib.load(file_path4)

    # print(data1)

    print("full: ", data1.shape)
    print("label: ", label.shape)
    print("registration: ", reg.shape)
    print("weight: ", weight.shape)

    # plt.figure()


def create_dataset(path=base_dir):
    """
        Create the 3d data set containing the sementation results and unlabled images
        Takes the raw *.nii format and convertes it into panda dataframes
        output is indexed as (time slice, width, height, z-stack)

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
