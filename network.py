import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils
from layers import BipertiteGraphConvolution


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


# Discriminator
class discriminator(nn.Module):
    def __init__(self, num_item, in_features_u, in_features_v, rating, output_dim=1):
        super(discriminator, self).__init__()
        self.in_features_u = in_features_u
        self.in_features_v = in_features_v
        self.num_item = num_item
        self.output_dim = output_dim
        self.rating = rating

        self.bgc1 = BipertiteGraphConvolution(in_features_u=self.in_features_u, 
                                            in_features_v=self.in_features_v, 
                                            out_features_u=32, 
                                            out_features_v=32, 
                                            rating=self.rating)
        self.bgc2 = BipertiteGraphConvolution(in_features_u=32, 
                                            in_features_v=32, 
                                            out_features_u=64, 
                                            out_features_v=64, 
                                            rating=self.rating)
        self.fc = nn.Linear(self.num_item+64, self.output_dim)

        utils.initialize_weights(self)

    def forward(self, adj, u_feature, v_feature):
        x_u, x_v = self.bgc1(adj, u_feature, v_feature)
        x_u, x_v = self.bgc2(adj, x_u, x_v)
        x = torch.cat([adj, x_u], -1)
        x = self.fc(x)
        return torch.sigmoid(x)

'''
class discriminator(nn.Module):
    def __init__(self, num_user, num_item, in_features_u, in_features_v, rating, hidden_num=100, output_dim=1):
        super(discriminator, self).__init__()
        self.in_features_u = in_features_u
        self.in_features_v = in_features_v
        self.output_dim = output_dim
        self.hidden_num = hidden_num
        self.rating = rating

        self.bgc = BipertiteGraphConvolution(self.in_features_u, self.in_features_v, self.hidden_num, self.hidden_num, self.rating)
        self.bdense = BipertiteDense(self.hidden_num, self.hidden_num, 1, 1)
        self.fc = nn.Linear(num_user + num_item, self.output_dim)

        utils.initialize_weights(self)

    def forward(self, adj, u_feature, v_feature):
        x_u, x_v = self.bgc(adj, u_feature, v_feature)
        x_u, x_v = self.bdense(x_u, x_v)
        x = torch.cat([x_u, x_v], 0)
        x = torch.transpose(x, 0, 1)
        x = self.fc(x)

        return torch.sigmoid(x)
'''

#-----------------------------------------------------------------

'''
class Graph_discriminator(nn.Module):
    def __init__(self, num_user, num_item, in_features_u, in_features_v, rating, hidden_num=100, output_dim=1):
        super(Graph_discriminator, self).__init__()
        self.in_features_u = in_features_u
        self.in_features_v = in_features_v
        self.output_dim = output_dim
        self.hidden_num = hidden_num
        self.rating = rating

        self.gc = GraphConvolution(self.in_features_u, self.in_features_v, self.hidden_num, self.hidden_num, self.rating)
        self.dense = Dense(self.hidden_num, self.hidden_num, 1, 1)
        self.fc = nn.Linear(num_user, self.output_dim)

        utils.initialize_weights(self)

    def forward(self, adj, u_feature, v_feature):
        x = self.gc(adj, u_feature, v_feature)
        x = self.dense(x)
        x = torch.transpose(x, 0, 1)
        x = self.fc(x)

        return torch.sigmoid(x)
'''

