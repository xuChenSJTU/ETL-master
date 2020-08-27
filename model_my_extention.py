from __future__ import print_function
import argparse
import numpy as np
import torch
import torch.utils.data
from torch import nn, optim
import torch.nn.functional as F

# class Cluster_layer(nn.Module):
#     def __init__(self, emb_size, num_cluster=2, iters=4, tau=1.0, **kwargs):
#         super(Cluster_layer, self).__init__()
#         self.n_cluster = num_cluster
#         self.iters = iters
#         self.tau = tau
#         self.centers = nn.Parameter(torch.nn.init.xavier_uniform_(torch.empty(self.n_cluster, emb_size)))
#
#     def forward(self, u_vecs):
#         for i in range(self.iters):
#             distance = torch.matmul(u_vecs.unsqueeze(1), F.normalize(self.centers, p=1, dim=1).T)
#             assigns = F.softmax(distance*self.tau)
#
#
#         return o, c


class CDAE(nn.Module):
    def __init__(self, NUM_USER, NUM_MOVIE, NUM_BOOK, EMBED_SIZE, dropout, x_cluster_num=10, y_cluster_num=10, is_sparse=False):
        super(CDAE, self).__init__()
        self.NUM_MOVIE = NUM_MOVIE
        self.NUM_BOOK = NUM_BOOK
        self.NUM_USER = NUM_USER
        self.emb_size = EMBED_SIZE

        self.user_embeddings = nn.Embedding(self.NUM_USER, EMBED_SIZE, sparse=is_sparse)
        self.user_embeddings.weight.data = torch.from_numpy(
            np.random.normal(0, 0.01, size=[self.NUM_USER, EMBED_SIZE])).float()

        self.encoder_x = nn.Sequential(
            nn.Linear(self.NUM_MOVIE, EMBED_SIZE),
            nn.ReLU(),
            nn.Linear(EMBED_SIZE, EMBED_SIZE)
        )
        self.decoder_x = nn.Sequential(
            nn.Linear(EMBED_SIZE, EMBED_SIZE),
            nn.ReLU(),
            nn.Linear(EMBED_SIZE, self.NUM_MOVIE)
        )
        self.encoder_y = nn.Sequential(
            nn.Linear(self.NUM_BOOK, EMBED_SIZE),
            nn.ReLU(),
            nn.Linear(EMBED_SIZE, EMBED_SIZE)
        )
        self.decoder_y = nn.Sequential(
            nn.Linear(EMBED_SIZE, EMBED_SIZE),
            nn.ReLU(),
            nn.Linear(EMBED_SIZE, self.NUM_BOOK)
        )

        self.x_clustering = Cluster_layer(num_cluster=x_cluster_num)
        self.y_clustering = Cluster_layer(num_cluster=y_cluster_num)
        self.epsilon = torch.tensor(1e-10).type(torch.FloatTensor)  # .cuda()

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU


    def forward(self, batch_user, batch_user_x, batch_user_y):
        # encoding for x and y domain
        h_user_x = self.encoder_x(self.dropout(batch_user_x))
        h_user_y = self.encoder_y(self.dropout(batch_user_y))

        # disentangling for user representations in x and y domains
        x_centers, x_assigns = self.x_clustering(h_user_x.unsqueeze(1))
        y_centers, y_assigns = self.y_clustering(h_user_y.unsqueeze(1))

        x_centers_ = torch.sum(x_centers ** 2, dim=2, keepdim=True)
        y_centers_ = torch.sum(y_centers ** 2, dim=2, keepdim=True)

        distance_ = torch.sqrt(torch.max(x_centers_ - 2 * torch.bmm(x_centers, y_centers.permute(0, 2, 1)) + y_centers_.permute(0, 2, 1),
                                self.epsilon))
        distance = torch.min(distance_, dim=1)
        distance = distance.values

        # adding the ont-hot user encoding for user representations in x and y domains
        # h_user_x_ =
        h_user = self.user_embeddings(batch_user)
        feature_x = torch.add(h_user_x, h_user)
        feature_y = torch.add(h_user_y, h_user)
        z_x = F.relu(feature_x)
        z_y = F.relu(feature_y)

        # decoding for x and y domain
        preds_x = self.decoder_x(z_x)
        preds_y = self.decoder_y(z_y)

        return preds_x, preds_y, feature_x, feature_y, distance

    def get_user_embedding(self, batch_user_x, batch_user_y):
        # this is for SIGIR's experiment
        h_user_x = self.encoder_x(self.dropout(batch_user_x))
        h_user_y = self.encoder_y(self.dropout(batch_user_y))
        return h_user_x, h_user_y

    def get_latent_z(self, batch_user, batch_user_x, batch_user_y):
        # this is for clustering visualization
        h_user_x = self.encoder_x(self.dropout(batch_user_x))
        h_user_y = self.encoder_y(self.dropout(batch_user_y))
        h_user = self.user_embeddings(batch_user)
        feature_x = torch.add(h_user_x, h_user)
        feature_y = torch.add(h_user_y, h_user)
        z_x = F.relu(feature_x)
        z_y = F.relu(feature_y)

        return z_x, z_y


class Discriminator(nn.Module):
    def __init__(self, n_fts, dropout):
        super(Discriminator, self).__init__()
        self.dropout = dropout
        self.training = True

        self.disc = nn.Sequential(
            nn.Linear(n_fts, int(n_fts / 2)),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(int(n_fts / 2), 1))

    def forward(self, x):
        # make mlp for discriminator
        h = self.disc(x)
        return h


def save_embedding_process(model, save_loader, feed_data, is_cuda):
    fts1 = feed_data['fts1']
    fts2 = feed_data['fts2']

    user_embedding1_list = []
    user_embedding2_list = []
    model.eval()
    for batch_idx, data in enumerate(save_loader):
        data = data.reshape([-1])
        val_user_arr = data.numpy()
        v_item1 = fts1[val_user_arr]
        v_item2 = fts2[val_user_arr]
        if is_cuda:
            v_user = torch.LongTensor(val_user_arr).cuda()
            v_item1 = torch.FloatTensor(v_item1).cuda()
            v_item2 = torch.FloatTensor(v_item2).cuda()
        else:
            v_user = torch.LongTensor(val_user_arr)
            v_item1 = torch.FloatTensor(v_item1)
            v_item2 = torch.FloatTensor(v_item2)

        res = model.get_user_embedding(v_item1, v_item2)
        user_embedding1 = res[0]
        user_embedding2 = res[1]
        if is_cuda:
            user_embedding1 = user_embedding1.detach().cpu().numpy()
            user_embedding2 = user_embedding2.detach().cpu().numpy()
        else:
            user_embedding1 = user_embedding1.detach().numpy()
            user_embedding2 = user_embedding2.detach().numpy()

        user_embedding1_list.append(user_embedding1)
        user_embedding2_list.append(user_embedding2)

    return np.concatenate(user_embedding1_list, axis=0), np.concatenate(user_embedding2_list, axis=0)


def save_z_process(model, save_loader, feed_data, is_cuda):
    fts1 = feed_data['fts1']
    fts2 = feed_data['fts2']

    user_embedding1_list = []
    user_embedding2_list = []
    model.eval()
    for batch_idx, data in enumerate(save_loader):
        data = data.reshape([-1])
        val_user_arr = data.numpy()
        v_item1 = fts1[val_user_arr]
        v_item2 = fts2[val_user_arr]
        if is_cuda:
            v_user = torch.LongTensor(val_user_arr).cuda()
            v_item1 = torch.FloatTensor(v_item1).cuda()
            v_item2 = torch.FloatTensor(v_item2).cuda()
        else:
            v_user = torch.LongTensor(val_user_arr)
            v_item1 = torch.FloatTensor(v_item1)
            v_item2 = torch.FloatTensor(v_item2)

        res = model.get_latent_z(v_user, v_item1, v_item2)
        user_embedding1 = res[0]
        user_embedding2 = res[1]
        if is_cuda:
            user_embedding1 = user_embedding1.detach().cpu().numpy()
            user_embedding2 = user_embedding2.detach().cpu().numpy()
        else:
            user_embedding1 = user_embedding1.detach().numpy()
            user_embedding2 = user_embedding2.detach().numpy()

        user_embedding1_list.append(user_embedding1)
        user_embedding2_list.append(user_embedding2)

    return np.concatenate(user_embedding1_list, axis=0), np.concatenate(user_embedding2_list, axis=0)