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
        self.weight_u = nn.Parameter(torch.Tensor(self.rating, in_features_v, out_features_u))
        self.weight_v = nn.Parameter(torch.Tensor(self.rating, in_features_u, out_features_v))

        self.weight_u.data.normal_(0, 0.02)
        self.weight_v.data.normal_(0, 0.02)
    
    def forward(self, adj, u_feature, v_feature):
        M_r = torch.zeros(self.rating, adj.shape[0], adj.shape[1], dtype=torch.float, device=adj.device)
        for i in range(self.rating):
            M_r[i] = adj == i+1
        M_r_t = torch.transpose(M_r, 1, 2)
        D_u = torch.diag(torch.reciprocal(torch.sum(adj > 0, dim=1, dtype=torch.float)))
        D_v = torch.diag(torch.reciprocal(torch.sum(adj > 0, dim=0, dtype=torch.float)))
        D_u[D_u == float("Inf")] = 0
        D_v[D_v == float("Inf")] = 0
        output_u = torch.zeros(D_u.shape[0], self.weight_u.shape[2], dtype=torch.float, device=adj.device)
        output_v = torch.zeros(D_v.shape[0], self.weight_v.shape[2], dtype=torch.float, device=adj.device)
        for i in range(self.rating):
            output_u = output_u + torch.mm(torch.mm(torch.mm(D_u, M_r[i]), v_feature), self.weight_u[i])
            output_v = output_v + torch.mm(torch.mm(torch.mm(D_v, M_r_t[i]), u_feature), self.weight_v[i])
            
        return F.relu(output_u), F.relu(output_v)



class BipertiteDense(nn.Module):
    """Dense layer for two types of nodes in a bipartite graph. """
    def __init__(self, in_features_u, in_features_v, out_features_u, out_features_v):
        super(BipertiteDense, self).__init__()
        self.weight_u = nn.Parameter(torch.Tensor(in_features_u, out_features_u))
        self.weight_v = nn.Parameter(torch.Tensor(in_features_v, out_features_v))

        self.weight_u.data.normal_(0, 0.02)
        self.weight_v.data.normal_(0, 0.02)

    def forward(self, u_feature, v_feature):
        output_u = torch.mm(u_feature, self.weight_u)
        output_v = torch.mm(v_feature, self.weight_v)

        return F.relu(output_u), F.relu(output_v)



'''
class GraphConvolution(nn.Module):
    """Graph convolution layer for bipartite graphs"""
    def __init__(self, in_features_u, in_features_v, out_features_u, out_features_v, rating):
        super(GraphConvolution, self).__init__()
        self.rating = rating
        self.weight_r = nn.Parameter(torch.Tensor(self.rating, in_features_v, out_features_u))

        self.weight_r.data.normal_(0, 0.02)
    
    def forward(self, adj, u_feature, v_feature):
        M_r = torch.zeros(self.rating, adj.shape[0], adj.shape[1], dtype=torch.float, device=adj.device)
        for i in range(self.rating):
            M_r[i] = adj == i+1
        D = torch.diag(torch.reciprocal(torch.sum(adj > 0, dim=1, dtype=torch.float)))
        D[D == float("Inf")] = 0
        output = torch.zeros(D.shape[0], self.weight_r.shape[2], dtype=torch.float, device=adj.device)
        for i in range(self.rating):
            output = output + torch.mm(torch.mm(torch.mm(D, M_r[i]), v_feature), self.weight_r[i])
            
        return F.relu(output)

'''
'''
class GraphConvolution(nn.Module):
    """Graph convolution layer for bipartite graphs"""
    def __init__(self, in_features_u, in_features_v, out_features_u, out_features_v, rating):
        super(GraphConvolution, self).__init__()
        self.rating = rating
        self.weight = nn.Parameter(torch.Tensor(in_features_v, out_features_u))

        self.weight.data.normal_(0, 0.02)
    
    def forward(self, adj, u_feature, v_feature):
        D = torch.diag(torch.reciprocal(torch.sum(adj > 0, dim=1, dtype=torch.float)))
        D[D == float("Inf")] = 0
        output = torch.mm(torch.mm(torch.mm(D, adj), v_feature), self.weight)
            
        return F.relu(output)

'''

class GraphConvolution(nn.Module):
    """Graph convolution layer for bipartite graphs"""
    def __init__(self, in_features_u, in_features_v, out_features_u, out_features_v, rating):
        super(GraphConvolution, self).__init__()
        self.rating = rating
        self.weight = nn.Parameter(torch.Tensor(in_features_u+in_features_v, out_features_u))

        self.weight.data.normal_(0, 0.02)
    
    def forward(self, adj, u_feature, v_feature):
        D = torch.diag(torch.reciprocal(torch.sum(adj > 0, dim=1, dtype=torch.float)))
        D[D == float("Inf")] = 0
        feature = torch.cat([torch.mm(torch.mm(D, adj), v_feature), u_feature], 1)
        output = torch.mm(feature, self.weight)
            
        return F.relu(output)


class Dense(nn.Module):
    """Dense layer for two types of nodes in a bipartite graph. """
    def __init__(self, in_features_u, in_features_v, out_features_u, out_features_v):
        super(Dense, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_features_u, out_features_u))

        self.weight.data.normal_(0, 0.02)

    def forward(self, u_feature):
        output = torch.mm(u_feature, self.weight)

        return F.relu(output)