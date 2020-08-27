from __future__ import print_function
import argparse
import numpy as np
import torch
import torch.utils.data
from torch import nn, optim
import torch.nn.functional as F

class CDAE(nn.Module):
    def __init__(self, NUM_USER, NUM_MOVIE, NUM_BOOK,  EMBED_SIZE, dropout, is_sparse=False):
        super(CDAE, self).__init__()
        self.NUM_MOVIE = NUM_MOVIE
        self.NUM_BOOK = NUM_BOOK
        self.NUM_USER = NUM_USER
        self.emb_size = EMBED_SIZE

        self.user_embeddings = nn.Embedding(self.NUM_USER, EMBED_SIZE, sparse=is_sparse)
        self.user_embeddings.weight.data = torch.from_numpy(np.random.normal(0, 0.01, size=[self.NUM_USER, EMBED_SIZE])).float()

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

        self.orthogonal_w1 = nn.Parameter(nn.init.xavier_uniform(torch.Tensor(EMBED_SIZE, EMBED_SIZE).type(
            torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor), gain=np.sqrt(2.0)),
                              requires_grad=True)

        self.orthogonal_w2 = nn.Parameter(nn.init.xavier_uniform(torch.Tensor(EMBED_SIZE, EMBED_SIZE).type(
            torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor), gain=np.sqrt(2.0)),
            requires_grad=True)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU

    def orthogonal_map(self, z_x, z_y):
        mapped_z_x = torch.matmul(z_x, self.orthogonal_w1)
        mapped_z_y = torch.matmul(z_y, self.orthogonal_w2)
        return mapped_z_x, mapped_z_y

    def forward(self, batch_user, batch_user_x, batch_user_y):
        h_user_x = self.encoder_x(self.dropout(batch_user_x))
        h_user_y = self.encoder_y(self.dropout(batch_user_y))
        h_user = self.user_embeddings(batch_user)
        feature_x = torch.add(h_user_x, h_user)
        feature_y = torch.add(h_user_y, h_user)
        z_x = F.relu(feature_x)
        z_y = F.relu(feature_y)
        preds_x = self.decoder_x(z_x)
        preds_y = self.decoder_y(z_y)
        mapped_z_x, mapped_z_y = self.orthogonal_map(z_x, z_y)
        preds_x2y = self.decoder_y(mapped_z_x)
        preds_y2x = self.decoder_x(mapped_z_y)

        # # define orthogonal constraint loss
        z_x_ = torch.matmul(mapped_z_x, self.orthogonal_w2)
        z_y_ = torch.matmul(mapped_z_y, self.orthogonal_w1)
        z_x_reg_loss = torch.norm(z_x - z_x_, p=1, dim=1)
        z_y_reg_loss = torch.norm(z_y - z_y_, p=1, dim=1)

        return preds_x, preds_y, preds_x2y, preds_y2x, feature_x, feature_y, z_x_reg_loss, z_y_reg_loss
    def get_user_embedding(self, batch_user_x, batch_user_y):
        h_user_x = self.encoder_x(self.dropout(batch_user_x))
        h_user_y = self.encoder_y(self.dropout(batch_user_y))
        return h_user_x, h_user_y

# class MI_Map(nn.Module):
#     def __init__(self, n_input):
#         super(MI_Map, self).__init__()
#
#         self.fc_x = nn.Linear(n_input, n_input, bias=False)
#         # self.fc_y = nn.Linear(n_input, n_input, bias=False)
#
#     def forward(self, z_x, z_y):
#         # make mlp for discriminator
#         logits = torch.sum(self.fc_x(z_x)*z_y, dim=1)
#         return logits

class Discriminator(nn.Module):
    def __init__(self, n_fts, dropout):
        super(Discriminator, self).__init__()
        self.dropout = dropout
        self.training = True

        self.disc = nn.Sequential(
            nn.Linear(n_fts, int(n_fts/2)),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(int(n_fts/2), 1))

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