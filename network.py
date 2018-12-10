import numpy as np
import torch
import torch.nn as nn

import utils
from layers import BipertiteDense, BipertiteGraphConvolution

'''
Generator Code
-----------------------------------------
palameter

input_dim       input noize dim
feature_num     input feature dim
layer_num       number of layer
hidden_num      number of node on hidden layer
output_dim      purchase vector dim

------------------------------------------
'''
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

    def forward(self, input, feature):
        x = torch.cat([input, feature], 1)
        for f in self.layer_list:
            x = torch.sigmoid(f(x))
        return self.fc_out(x)


'''
Discriminator Code
-----------------------------------------
palameter
input_dim           perchase vector dim
feature_num         input feature dim
layer_num           number of hidden layer
hidden_num          number of node on hidden layer
output_dim          real or fake
------------------------------------------
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
