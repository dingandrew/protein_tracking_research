import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from tqdm import tqdm
from sklearn.cluster import DBSCAN
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import KernelPCA
import math
import pickle
from util import open_model_json, showTensor
from collections import Counter


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
        self.predictor = None
        self.classifier = None
        # self.gauss_kernel = self.get_gaussian_kernel()

    def forward(self, frame1, frame2, target, train=True):
        '''
            1. input_seq -> fetaure_extractor => features
            2. features -> RNN(bidrectional=true) => forwards and backwards predictions
            3. predictions -> loss_calculator => loss
            4. return loss

            Input: input_seq has shape [batch, time_step, depth, z, x, y]
                   target, the object we are trying to track through time
                           it is h_0 

        '''
        # print(frame2.shape, frame1.shape, target.shape)
        # self.graph_2d(frame1)
        # self.graph_2d(frame2)
        # self.graph_2d(target)
        f1 = frame1.view(-1,
                         frame1.size(2),
                         frame1.size(3),
                         frame1.size(4))  # (batch * time_frame), D, H, W
        f2 = frame2.view(-1,
                         frame2.size(2),
                         frame2.size(3),
                         frame2.size(4))  # (batch * time_frame), D, H, W
        targ = target.view(target.size(3),
                           target.size(4))  # H, W
        # print(f2.shape, f1.shape, targ.shape)

        # (out_channels, in_channels, kZ, kH, kW)
        # 5 filters with same shape as entire frame
        weights = torch.empty((5, 1, target.size(3), target.size(4))).cuda()

        # create custom kernel filters by transforming target
        weights[0, ...] = targ  # original target mask
        weights[1, ...] = torch.roll(targ,
                                     shifts=(5, 0),
                                     dims=(0, 1))  # roll x by 3
        weights[2, ...] = torch.roll(targ,
                                     shifts=(-5, 0),
                                     dims=(0, 1))  # roll x by -3
        weights[3, ...] = torch.roll(targ,
                                     shifts=(0, 5),
                                     dims=(0, 1))  # roll y by 3
        weights[4, ...] = torch.roll(targ,
                                     shifts=(0, -5),
                                     dims=(0, 1))  # roll y by -3

        if train:
            # only need these feature for fitting the detector
            f1_features = F.conv2d(input=f1, weight=weights)
            f1_features = torch.flatten(f1_features)
        else:
            f1_features = None

        f2_features = F.conv2d(input=f2, weight=weights)
        f2_features = torch.flatten(f2_features)
        return f1_features, f2_features

    def predict(self, feature):
        # print(feature)
        # load the model
        if self.predictor is None or self.classifier is None:
            with open('../../models/sprite_detector.pickle', 'rb') as f:
                self.predictor = pickle.load(f)
                tqdm.write('Loaded the trained predictor model')

            # init classifier
            self.classifier = KNeighborsClassifier(
                n_neighbors=len(np.unique(self.predictor['DBSCAN'].labels_)))
            self.classifier.fit(self.predictor['DBSCAN'].components_,
                                self.predictor['DBSCAN'].labels_[self.predictor['DBSCAN'].core_sample_indices_])

        pca2 = self.predictor['PCA2']
        p2 = pca2.transform(feature)

        # print(p2)
        predictions = self.classifier.predict(p2)
        return predictions

    def train_feat(self, f1, f2, graph=True):
        midpt = len(f1)
        # need to reduce the dimesionality of the data
        pca1 = KernelPCA(n_components=2, kernel='sigmoid', gamma=0.7)
        pca2 = KernelPCA(n_components=2, kernel='sigmoid', gamma=0.7)
        t1 = pca1.fit_transform(f1)
        t2 = pca2.fit_transform(f2)

        transformed_data = np.concatenate((t1, t2), axis=0)

        # dbscan will label the clusters
        dbscan = DBSCAN(eps=0.05, min_samples=10)
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
            # print(dbscan.labels_[0:20])
            # print(dbscan.core_sample_indices_[0:20] + 1)
            # print(dbscan.labels_[10000:10020])
            print('Labels: ', Counter(dbscan.labels_).keys())
            print('Label Counts: ', Counter(dbscan.labels_).values())
            self.plot_dbscan(dbscan, transformed_data, size=100)

        # save the models
        self.predictor = {'PCA1': pca1, 'PCA2': pca2, 'DBSCAN': dbscan}
        with open('../../models/sprite_detector.pickle', 'wb') as f:
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

    def graph_2d(self, f):
        plt.figure()
        plt.imshow(f[0, 0, 0, ...].cpu())
        plt.show()
    


if __name__ == "__main__":
    # test
    model_config = open_model_json('./model_config.json')
    model = Detector(model_config['default'])
    print(model)
    param_num = sum([param.data.numel()
                     for param in model.parameters()])
    print('Parameter number: %.3f ' % (param_num))
