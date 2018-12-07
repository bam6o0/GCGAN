import math

import torch

import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class BipertiteGraphConvolution(nn.Module):
    """Graph convolution layer for bipartite graphs"""
    def __init__(self, in_features_u, in_features_v, out_features_u, out_features_v, rating):
        super(BipertiteGraphConvolution, self).__init__()
        self.rating = rating
        self.weight_u = nn.Parameter(torch.FloatTensor(in_features_v, out_features_u))
        self.weight_v = nn.Parameter(torch.FloatTensor(in_features_u, out_features_v))
    
    def forward(self, adj, u_feature, v_feature):
        M_r = torch.FloatTensor(self.rating, adj.shape[0], adj.shape[1]).cuda()
        for i in range(self.rating):
            M_r[i] = adj == i+1
        M_r_t = torch.transpose(M_r, 1, 2).cuda()
        D_u = torch.diag(torch.reciprocal(torch.sum(adj > 0, dim=1, dtype=torch.float))).cuda()
        D_v = torch.diag(torch.reciprocal(torch.sum(adj > 0, dim=0, dtype=torch.float))).cuda()
        output_u = torch.zeros(D_u.shape[0], self.weight_u.shape[1], dtype=torch.float).cuda()
        output_v = torch.zeros(D_v.shape[0], self.weight_v.shape[1], dtype=torch.float).cuda()
        for i in range(self.rating):
            output_u = output_u + torch.mm(torch.mm(torch.mm(D_u, M_r[i]), v_feature), self.weight_u)
            output_v = output_v + torch.mm(torch.mm(torch.mm(D_v, M_r_t[i]), u_feature), self.weight_v)

        return F.relu(output_u), F.relu(output_v)



class BipertiteDense(nn.Module):
    """Dense layer for two types of nodes in a bipartite graph. """
    def __init__(self, in_features_u, in_features_v, out_features_u, out_features_v):
        super(BipertiteDense, self).__init__()
        self.weight_u = nn.Parameter(torch.FloatTensor(in_features_u, out_features_u))
        self.weight_v = nn.Parameter(torch.FloatTensor(in_features_v, out_features_v))

    def forward(self, u_feature, v_feature):
        output_u = torch.mm(u_feature, self.weight_u)
        output_v = torch.mm(v_feature, self.weight_v)

        return F.relu(output_u), F.relu(output_v)
