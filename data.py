import os
import nibabel as nib
import matplotlib.pyplot as plt

base_dir = './data/raw_data'


file_path1 = os.path.join(base_dir, 'Segmentation_and_result' , '1/Fullsize_1.nii' )
file_path2 = os.path.join(base_dir, 'Segmentation_and_result' , '1/Fullsize_label_1.nii' )
file_path3 = os.path.join(base_dir, 'Segmentation_and_result' , '1/Registration_1.nii' )
file_path4 = os.path.join(base_dir, 'Segmentation_and_result' , '1/Weights_1.nii' )


data1 = nib.load(file_path1)
label = nib.load(file_path2)
reg = nib.load(file_path3)
weight = nib.load(file_path4)


# print(data1)

print('full: ', data1.shape)
print('label: ', label.shape)
print('registration: ', reg.shape)
print('weight: ', weight.shape)

plt.imshow(data1.dataobj[... , 8])
plt.show()




