import os, gzip, torch
import torch.nn as nn
import numpy as np
import scipy.misc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import datasets, transforms



def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)



def loss_plot(hist, path = 'Train_hist.png', model_name = ''):
    x = range(len(hist['D_loss']))

    y1 = hist['D_loss']
    y2 = hist['G_loss']

    plt.plot(x, y1, label='D_loss')
    plt.plot(x, y2, label='G_loss')

    plt.xlabel('Iter')
    plt.ylabel('Loss')

    plt.legend(loc=4)
    plt.grid(True)
    plt.tight_layout()

    path = os.path.join(path, model_name + '_loss.png')

    plt.savefig(path)

    plt.close()

def predict_plot(matrix, path, model_name=''):
    plt.figure(figsize=(25, 15)) 
    sns.heatmap(matrix)
    path = os.path.join(path, model_name + '_heatmap.png')
    plt.savefig(path)
    plt.close()

def initialize_weights(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.ConvTranspose2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            #m.weight.data.normal_(0, 0.02)
            #m.weight.data.xavier_normal_()
            torch.nn.init.xavier_normal_(m.weight)
            m.bias.data.zero_()