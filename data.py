import os
import nibabel as nib
import matplotlib.pyplot as plt
from scipy.io import loadmat
import sys
import pandas as pd
import numpy
numpy.set_printoptions(threshold=sys.maxsize)

base_dir = './data/raw_data/Segmentation_and_result'



def explore_data():
    '''
        Some code to explore the data properties
    '''
    file_path1 = os.path.join(base_dir, '1/Fullsize_1.nii' )
    file_path2 = os.path.join(base_dir, '1/Fullsize_label_1.nii' )
    file_path3 = os.path.join(base_dir, '1/Registration_1.nii' )
    file_path4 = os.path.join(base_dir, '1/Weights_1.nii' )
    file_path5 = os.path.join(base_dir, '1/1_3Dconnection2.fig' )


    data1 = nib.load(file_path1)
    label = nib.load(file_path2)
    reg = nib.load(file_path3)
    weight = nib.load(file_path4)


    # print(data1)

    print('full: ', vars(data1.dataobj))
    print('label: ', label.shape)
    print('registration: ', reg.shape)
    print('weight: ', weight.shape)

    plt.figure()


    # plt.imshow(data1.dataobj[... , 7] )
    plt.imshow(label.dataobj[... , 12])
    # plt.imshow(reg.dataobj)
    plt.show()

    print(label.dataobj[ ... , 12] )

    print('-------------------------------------')

    # print(reg.dataobj[...])


    # print('-------------------------------------')

    # print(weight.dataobj[..., 7, 7])



def create_dataset(path = base_dir):
    '''
        Create the 3d data set containing the sementation results and unlabled images
        Takes the raw *.nii format and convertes it into panda dataframes

        Input: path, to directery containing .nii files

        Outputs : labled_3data(70, 248, 150, 13), raw_3data(70, 248, 150, 13)
    '''
    image_list = []
    
    valid_images = [".nii"]

    for f in os.listdir(path):
        print( f)
        ext = os.path.splitext(f)[1]
        if ext.lower() not in valid_images:
            continue
        id_img_count = []
        id_img_count.append(f)
        id_img_count.append(cv2.imread(os.path.join(path,f)))
        id_img_count.append(0)
        #image_list.append(Image.open(os.path.join(path,f)))
        #image_list.append(cv2.imread(os.path.join(path,f)))
        image_list.append(id_img_count)

    return pd.DataFrame(image_list, columns=["ID", "IMAGE", "COUNT"])



def save_data(path = '../data/'):
    '''
        Save the data as pickle
    '''
    pass







def plotFig(filename,fignr=1):
    d = loadmat(filename,squeeze_me=True, struct_as_record=False)
    print(d)
    print()
    print(vars(d['hgS_070000']))
    print()
    print('type: ', d['hgS_070000'].type)
    print()
    print('handle: ', d['hgS_070000'].handle)
    print()
    print('properties: ', vars(d['hgS_070000'].properties))
    print()
    print('children: ', vars(d['hgS_070000'].children[0].children[0].properties))
    print()
    print('special: ', d['hgS_070000'].special)
    
    print('\nhgM070000')
    print()
    print(vars(d['hgM_070000']))    


# plotFig(file_path5)


