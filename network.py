import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils
from layers import BipertiteGraphConvolution

'''
# Generator
class generator(nn.Module):
    def __init__(self, input_dim=100, feature_num=10, output_dim=10, layer_num=2, hidden_num=100):
        super(generator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layer_num = layer_num
        self.hidden_num = hidden_num
        self.feature_num = feature_num

        fc_in = nn.Linear(self.input_dim + self.feature_num, hidden_num)
        fc_hidden = nn.Linear(hidden_num, hidden_num)
        self.fc_out = nn.Linear(hidden_num, output_dim)

        layer_list = []
        layer_list.append(fc_in)
        for _ in range(self.layer_num-2):
            layer_list.append(fc_hidden)

        self.layer_list = nn.ModuleList(layer_list)
        utils.initialize_weights(self)

    def forward(self, noise, feature):
        x = torch.cat([noise, feature], -1)
        for f in self.layer_list:
            x = torch.sigmoid(f(x))

        return self.fc_out(x)
'''

# Generator
class generator(nn.Module):
    def __init__(self, input_dim=100, feature_num=10, output_dim=10, layer_num=2, hidden_num=100):
        super(generator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.feature_num = feature_num

        self.fc = nn.Sequential(
            nn.Linear(self.input_dim + self.feature_num, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 64 * input_dim),
            nn.BatchNorm1d(64 * input_dim),
            nn.ReLU(),
            nn.Linear(64 * input_dim, output_dim)
        )

        utils.initialize_weights(self)

    def forward(self, noise, feature):
        x = torch.cat([noise, feature], -1)
        x = self.fc(x)

        return x

# Discriminator
class discriminator(nn.Module):
    def __init__(self, num_item, in_features_u, in_features_v, rating, output_dim, layer_num):
        super(discriminator, self).__init__()
        self.in_features_u = in_features_u
        self.in_features_v = in_features_v
        self.num_item = num_item
        self.output_dim = output_dim
        self.rating = rating
        
        self.bgc = BipertiteGraphConvolution(in_features_u=self.in_features_u, 
                                    in_features_v=self.in_features_v, 
                                    out_features_u=self.in_features_u,
                                    out_features_v=self.in_features_v,
                                    rating=self.rating)
        layer_list = []
        for _ in range(layer_num-1):
            layer_list.append(self.bgc)
            
        self.layer_list = nn.ModuleList(layer_list)               
        self.fc = nn.Linear(self.num_item+self.in_features_u, self.output_dim)

        utils.initialize_weights(self)

    def forward(self, adj, u_feature, v_feature):
        x_u, x_v = self.bgc(adj, u_feature, v_feature)
        for f in self.layer_list:
            x_u, x_v = self.bgc(adj, x_u, x_v)
        x = torch.cat([adj, x_u], -1)
        x = self.fc(x)
        return torch.sigmoid(x)
