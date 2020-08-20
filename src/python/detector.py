import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import KNeighborsClassifier
from util import open_model_json, showTensor
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pickle


class Detector(nn.Module):
    '''
        End to end tracker on segmented input

    '''

    def __init__(self, params):
        '''
            See model_config.json for parameter values

            Input: args, for the task to set up the network for
        '''
        super(Detector, self).__init__()
        self.params = params.copy()

    def forward(self, frame1, frame2, target):
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
        # self.graph_3d(frame1)
        # self.graph_3d(frame2)
        # self.graph_3d(target)

        f1 = frame1.view(-1,
                         frame1.size(2),
                         frame1.size(3),
                         frame1.size(4),
                         frame1.size(5))  # (batch * time_frame), D, Z , H, W
        f2 = frame2.view(-1,
                         frame2.size(2),
                         frame2.size(3),
                         frame2.size(4),
                         frame2.size(5))  # (batch * time_frame), D, Z , H, W
        targ = target.view(target.size(3),
                           target.size(4),
                           target.size(5))  # Z , H, W

        # print(f2.shape, f1.shape, targ.shape)

        # (out_channels, in_channels, kZ, kH, kW)
        # 5 filters with same shape as entire frame
        weights = torch.empty(
            (7, 1, target.size(3), target.size(4), target.size(5))).cuda()

        # create custom kernel filters by transforming target
        weights[0, ...] = targ  # original target mask
        weights[1, ...] = torch.roll(targ, 
                                     shifts=(0, 3, 0), 
                                     dims=(0, 1, 2))  # roll x by 3
        weights[2, ...] = torch.roll(targ,
                                     shifts=(0, -3, 0),
                                     dims=(0, 1, 2))  # roll x by -3
        weights[3, ...] = torch.roll(targ,
                                     shifts=(0, 0, 3),
                                     dims=(0, 1, 2))  # roll y by 3
        weights[4, ...] = torch.roll(targ,
                                     shifts=(0, 0, -3),
                                     dims=(0, 1, 2))  # roll y by -3
        weights[5, ...] = torch.roll(targ,
                                     shifts=(2, 0, 0),
                                     dims=(0, 1, 2))  # roll z by 2
        weights[6, ...] = torch.roll(targ,
                                     shifts=(-2, 0, 0),
                                     dims=(0, 1, 2))  # roll z by -2

        f1_features = F.conv3d(input=f1, weight=weights)
        f2_features = F.conv3d(input=f2, weight=weights)
        f1_features = torch.flatten(f1_features)
        f2_features = torch.flatten(f2_features)

        return f1_features, f2_features

    def fit_kmeans(self, results):
        pass
        




    def plot_feat(self, f1, f2):
        # need to reduce the dimesionality of the data
        # and cluster the data

        # with open('../../models/tsne1_init.pickle', 'rb') as f:
        #     init1 = pickle.load(f)
        # with open('../../models/tsne2_init.pickle', 'rb') as f:
        #     init2 = pickle.load(f)


        # tsne1 = TSNE(n_components=2, init=init1)
        # tsne2 = TSNE(n_components=2, init=init2)

        tsne1 = TSNE(n_components=2, init='random')
        tsne2 = TSNE(n_components=2, init='random')

        t = tsne1.fit_transform(f1)
        t2 = tsne2.fit_transform(f2)
        
        fig = plt.figure()
        ax = fig.add_subplot()
        ax.scatter(t[:, 0], t[:, 1], c='r')
        ax.scatter(t2[:, 0], t2[:, 1])
        plt.show()

        # with open('../../models/tsne1_init.pickle', 'wb') as f:
        #     # Pickle the 'data' dictionary using the highest protocol available.
        #     pickle.dump(tsne1.embedding_, f, pickle.HIGHEST_PROTOCOL)

        # with open('../../models/tsne2_init.pickle', 'wb') as f:
        #     # Pickle the 'data' dictionary using the highest protocol available.
        #     pickle.dump(tsne2.embedding_, f, pickle.HIGHEST_PROTOCOL)

        
        cluster_data = np.concatenate((t, t2), axis=0)

        dbscan = DBSCAN(eps=1.5, min_samples=10)

        dbscan.fit(cluster_data)
        print(dbscan.labels_[:10])
        print(dbscan.core_sample_indices_[:10])
        print(dbscan.components_[:3])
        print(np.unique(dbscan.labels_))
        plt.figure()
        plt.subplot()
        self.plot_dbscan(dbscan, cluster_data, size=100)
        plt.show()

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




    def graph_3d(self, f):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        z, x, y = f[0, 0, 0, ...].cpu().numpy().nonzero()
        ax.set_xlim3d(0, 280)
        ax.set_ylim3d(0, 512)
        ax.set_zlim3d(0, 13)
        ax.scatter(x, y, z, zdir='z')
        plt.show()


if __name__ == "__main__":
    # test
    model_config = open_model_json('./model_config.json')
    model = Detector(model_config['default'])
    print(model)
    param_num = sum([param.data.numel()
                     for param in model.parameters()])
    print('Parameter number: %.3f ' % (param_num))
