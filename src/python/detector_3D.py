import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import pickle
from collections import Counter

import torch.nn as nn
import torch.nn.functional as F
import torch

from sklearn.cluster import DBSCAN
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import KernelPCA

from util import load_json


class Detector(nn.Module):
    '''
        Detect if the given cluster is in the next frame or not
    '''

    def __init__(self, params):
        '''
            See model_config.json for parameter values
        '''
        super(Detector, self).__init__()
        self.params = params.copy()
        # Will be initialized only when self.predict is called ##
        self.predictor = None
        self.classifier = None
        #########################################################

    def forward(self, frame1, frame2, mask, train=True, fast=True):
        '''
            Get the feature embeddings of the cluster on the frame
        '''
        # (out_channels, in_channels, kZ, kH, kW)
        # 5 filters with same shape as entire frame
        weights = torch.empty((self.params['embedding_len'] * mask.size(0),
                               mask.size(1),
                               mask.size(2),
                               mask.size(3),
                               mask.size(4)))
                            #    .cuda()

        # create custom kernel filters by transforming each partial mask
        start = 0
        end = 0
        mask_count = 0
        while mask_count < mask.size(0):
            end += self.params['embedding_len']
            weights[start:end] = self.mutate_mask(mask[mask_count, ...])
            start = end
            mask_count += 1

        # if fast:
        #     # features are essentially number of intersections, if data is binarized
        #     # this is more efficient then performing convolutions

        #     weight_intersect = torch.nonzero(weights)
        #     print(weight_intersect)
        #     if train:
        #         # only need these feature for fitting the detector
        #         f1_features = F.conv3d(input=frame1, weight=weights)
        #         f1_features = f1_features.reshape((mask.size(0), 
        #                                         self.params['embedding_len']))
        #     else:
        #         f1_features = None

        #     f2_intersec = torch.unique(frame2)
        #     print(f2_intersec)
        #     f2_features = torch.zeroes((mask.size(0),
        #                                self.params['embedding_len'])).cuda()

        #     # for batch in range(mask.size(0)):
        #         # f2_features[batch, ...] = 
                
        #     # f2_features = (weight_intersect == f2_intersec)



        #     print(f2_features)

        #     exit()
        # else:
        if train:
            # only need these feature for fitting the detector
            f1_features = F.conv3d(input=frame1, weight=weights)
            f1_features = f1_features.reshape((mask.size(0), 
                                            self.params['embedding_len']))
        else:
            f1_features = None

        f2_features = F.conv3d(input=frame2, weight=weights)
        f2_features = f2_features.reshape((mask.size(0),
                                        self.params['embedding_len']))

        return f1_features, f2_features

    def mutate_mask(self, mask):
        '''
            For now the mutated partial mask is just the partial mask
            shifted in each direction/dimension by its own respective bodylength.

            Input: mask, a single partial mask of size: 
                   (mask_C, mask_Z, mask_X, mask_Y)

            Return: mutated_mask of size: 
                    (embedding_len, mask_C, mask_Z, mask_X, mask_Y)
        '''
        # find the indexes of all nonzero elements in tensor
        locs = torch.nonzero(mask, as_tuple=False) 
        # print(locs)
        # find the delta, which is the range of each z,x,y dim,
        # to shift the mask by it by its own body length
        z_delta = (max(locs[:, 1]) - min(locs[:, 1])) + 1
        x_delta = (max(locs[:, 2]) - min(locs[:, 2])) + 1
        y_delta = (max(locs[:, 3]) - min(locs[:, 3])) + 1
        
        # create empty mask that will hold the shifted partial mask
        mutated_mask = torch.empty((self.params['embedding_len'],
                                    mask.size(0),
                                    mask.size(1),
                                    mask.size(2),
                                    mask.size(3)))
                                    # .cuda()

        # create custom kernel filters by transforming mask
        mutated_mask[0, ...] = mask  # original mask mask
        mutated_mask[1, ...] = torch.roll(mask,
                                          shifts=(0, x_delta, 0),
                                          dims=(1, 2, 3))  # roll x by x_delta
        mutated_mask[2, ...] = torch.roll(mask,
                                          shifts=(0, -x_delta, 0),
                                          dims=(1, 2, 3))  # roll x by -x_delta
        mutated_mask[3, ...] = torch.roll(mask,
                                          shifts=(0, 0, y_delta),
                                          dims=(1, 2, 3))  # roll y by y_delta
        mutated_mask[4, ...] = torch.roll(mask,
                                          shifts=(0, 0, -y_delta),
                                          dims=(1, 2, 3))  # roll y by -y_delta
        mutated_mask[5, ...] = torch.roll(mask,
                                          shifts=(z_delta, 0, 0),
                                          dims=(1, 2, 3))  # roll z by z_delta
        mutated_mask[6, ...] = torch.roll(mask,
                                          shifts=(-z_delta, 0, 0),
                                          dims=(1, 2, 3))  # roll z by -z_delta
        
        return mutated_mask

    def predict(self, feature):
        # load the model
        if self.predictor is None or self.classifier is None:
            with open('../../models/detector.pickle', 'rb') as f:
                self.predictor = pickle.load(f)
                tqdm.write('Loaded the trained predictor model')

            # init classifier
            self.classifier = KNeighborsClassifier(
                n_neighbors=len(np.unique(self.predictor['DBSCAN'].labels_)))
            self.classifier.fit(self.predictor['DBSCAN'].components_,
                                self.predictor['DBSCAN'].labels_[self.predictor['DBSCAN'].core_sample_indices_])

        pca2 = self.predictor['PCA1']
        p2 = pca2.transform(feature)

        predictions = self.classifier.predict(p2)
        return predictions

    def train_feat(self, f1, f2, graph=True):
        midpt = len(f1)
        print(midpt)
        data = np.concatenate((f1, f2), axis=0)
        # need to reduce the dimesionality of the data
        pca1 = KernelPCA(n_components=2, kernel='sigmoid', gamma=0.7)
        transformed_data = pca1.fit_transform(data)
      
        # dbscan will label the clusters
        dbscan = DBSCAN(eps=0.02, min_samples=10)
        dbscan.fit(transformed_data)

        if graph:
            fig = plt.figure()
            ax = fig.add_subplot()
            f1_points = ax.scatter(transformed_data[:midpt, 0],
                                   transformed_data[:midpt, 1], c='r')
            f2_points = ax.scatter(transformed_data[midpt:, 0],
                                   transformed_data[midpt:, 1], marker='x')
            plt.legend([f1_points, f2_points], ['Frame 1', 'Frame 2'])
            plt.show()
            print('Labels: ', Counter(dbscan.labels_).keys())
            print('Label Counts: ', Counter(dbscan.labels_).values())
            self.plot_dbscan(dbscan, transformed_data, size=100)
        
        # save the models
        self.predictor = {'PCA1': pca1, 'DBSCAN': dbscan}
        with open('../../models/detector.pickle', 'wb') as f:
            pickle.dump(self.predictor, f, pickle.HIGHEST_PROTOCOL)

    def plot_dbscan(self, dbscan, X, size, show_xlabels=True, show_ylabels=True):
        core_mask = np.zeros_like(dbscan.labels_, dtype=bool)
        core_mask[dbscan.core_sample_indices_] = True
        anomalies_mask = dbscan.labels_ == -1
        non_core_mask = ~(core_mask | anomalies_mask)

        cores = dbscan.components_
        anomalies = X[anomalies_mask]
        non_cores = X[non_core_mask]

        plt.scatter(cores[:, 0], cores[:, 1],
                    c=dbscan.labels_[core_mask], marker='o', s=size, cmap="Paired")
        plt.scatter(cores[:, 0], cores[:, 1], marker='*',
                    s=20, c=dbscan.labels_[core_mask])
        plt.scatter(anomalies[:, 0], anomalies[:, 1],
                    c="r", marker="x", s=100)
        plt.scatter(non_cores[:, 0], non_cores[:, 1],
                    c=dbscan.labels_[non_core_mask], marker=".")

        # plt.legend()
        
        if show_xlabels:
            plt.xlabel("$x_1$", fontsize=14)
        else:
            plt.tick_params(labelbottom=False)
        if show_ylabels:
            plt.ylabel("$x_2$", fontsize=14, rotation=0)
        else:
            plt.tick_params(labelleft=False)
        plt.title("eps={:.2f}, min_samples={}".format(
            dbscan.eps, dbscan.min_samples), fontsize=14)
        plt.show()

    def graph_3d(self, f):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        z, x, y = f[0,  0, ...].cpu().numpy().nonzero()
        ax.set_xlim3d(0, 280)
        ax.set_ylim3d(0, 512)
        ax.set_zlim3d(0, 13)
        ax.scatter(x, y, z, zdir='z')
        plt.show()


if __name__ == "__main__":
    # test
    model_config = load_json('./model_config.json')
    model = Detector(model_config['default'])
    print(model)
    param_num = sum([param.data.numel()
                     for param in model.parameters()])
    print('Parameter number: %.3f ' % (param_num))
