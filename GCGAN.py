import torch, time, os, pickle
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, utils

from dataloader import MovieLensDataset, ToTensor
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
        self.bdense_1 = BipertiteDense(self.hidden_num, self.hidden_num, self.hidden_num, self.hidden_num)
        self.bdense_2 = BipertiteDense(self.hidden_num, self.hidden_num, 1, 1)
        self.fc_1 = nn.Linear(num_user + num_item, self.hidden_num)
        self.fc_2 = nn.Linear(self.hidden_num, self.output_dim)
        
        utils.initialize_weights(self)

    def forward(self, adj, u_feature, v_feature):
        x_u, x_v = self.bgc(adj, u_feature, v_feature)
        x_u, x_v = self.bdense_1(x_u, x_v)
        x_u, x_v = self.bdense_2(x_u, x_v)
        x = torch.cat([x_u, x_v], 0)
        x = torch.transpose(x, 0, 1)
        x = self.fc_1(x)
        x = self.fc_2(x)
        
        return torch.sigmoid(x)


class GCGAN(object):
    def __init__(self, args):
        # parameters
        self.epoch = args.epoch
        self.batch_size = args.batch_size
        self.save_dir = args.save_dir
        self.result_dir = args.result_dir
        self.dataset = args.dataset
        self.log_dir = args.log_dir
        self.gpu_mode = args.gpu_mode
        self.model_name = "GCGAN"
        self.Glayer_num = args.Glayer
        self.Dlayer_num = args.Dlayer
        self.hidden_num = args.hidden
        self.z_dim = 0
        self.num_worker = args.num_worker

        dataset = MovieLensDataset(dataset=self.dataset,
                                    transform=transforms.Compose([
                                        ToTensor()
                                    ]))

        self.u_feature_num = dataset[0]['u_feature'].shape[0]
        self.v_feature_num = dataset[0]['v_feature'].shape[1]

        # load dataset
        self.data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_worker)
        data = dataset[0]['u_perchase']
    
        # networks init
        self.G = generator(input_dim=self.z_dim, feature_num=self.u_feature_num, output_dim=data.shape[0], layer_num=self.Glayer_num, hidden_num=self.hidden_num)
        self.D = discriminator(num_user=self.batch_size, num_item=data.shape[0], in_features_u=self.u_feature_num, in_features_v=self.v_feature_num, rating=5, hidden_num=100, output_dim=1)
        self.G_optimizer = optim.SGD(self.G.parameters(), lr=args.lrG)
        self.D_optimizer = optim.SGD(self.D.parameters(), lr=args.lrD)

        if self.gpu_mode:
            self.G.cuda()
            self.D.cuda()
            self.D.bgc.cuda()
            self.D.bdense_1.cuda()
            self.D.bdense_2.cuda()
            self.BCE_loss = nn.BCELoss().cuda()
        else:
            self.BCE_loss = nn.BCELoss()
        

        print('---------- Networks architecture -------------')
        utils.print_network(self.G)
        utils.print_network(self.D)
        print('-----------------------------------------------')



    def train(self):
        self.train_hist = {}
        self.train_hist['D_loss'] = []
        self.train_hist['G_loss'] = []
        self.train_hist['per_epoch_time'] = []
        self.train_hist['total_time'] = []

        #flag in Discriminator 
        #self.y_real_, self.y_fake_ = torch.ones(self.batch_size, 1), torch.zeros(self.batch_size, 1)
        self.y_real_, self.y_fake_ = torch.ones(1, 1), torch.zeros(1, 1)
        if self.gpu_mode:
            self.y_real_, self.y_fake_ = self.y_real_.cuda(), self.y_fake_.cuda()

        self.D.train()
        print('training start!!')
        start_time = time.time()
        for epoch in range(self.epoch):
            self.G.train()
            epoch_start_time = time.time()
            for iter, sample in enumerate(self.data_loader):
                perchase = sample['u_perchase']
                u_feature = sample['u_feature']
                v_feature = sample['v_feature']

                if iter == self.data_loader.dataset.__len__() // self.batch_size:
                    break

                if(self.z_dim == 0):
                    z_ = torch.zeros(0, 0)

                if self.gpu_mode:
                    perchase, u_feature, v_feature, z_ = perchase.cuda(), u_feature.cuda(), v_feature.cuda(), z_.cuda()

                # update D network
                self.D_optimizer.zero_grad()
                D_real = self.D(perchase, u_feature, v_feature[0])
                D_real_loss = self.BCE_loss(D_real, self.y_real_)

                G_ = self.G(z_, u_feature)

                # masking fake perchase vector
                mask = (perchase > 0).float()

                D_fake = self.D(G_*mask, u_feature, v_feature[0])
                D_fake_loss = self.BCE_loss(D_fake, self.y_fake_)

                D_loss = D_real_loss + D_fake_loss
                self.train_hist['D_loss'].append(D_loss.item())

                D_loss.backward()
                self.D_optimizer.step()

                # update G network
                self.G_optimizer.zero_grad()

                G_ = self.G(z_, u_feature)
                D_fake = self.D(G_*mask, u_feature, v_feature[0])
                G_loss = self.BCE_loss(D_fake, self.y_real_)
                self.train_hist['G_loss'].append(G_loss.item())

                G_loss.backward()
                self.G_optimizer.step()


                if ((iter + 1) % 10) == 0:
                    print("Epoch: [%2d] [%4d/%4d] D_loss: %.8f, G_loss: %.8f" %
                        ((epoch + 1), (iter + 1), self.data_loader.dataset.__len__() // self.batch_size, D_loss.item(), G_loss.item()))
                    print(torch.min(G_), torch.max(G_))
                    

            self.train_hist['per_epoch_time'].append(time.time() - epoch_start_time)

        self.train_hist['total_time'].append(time.time() - start_time)
        print("Avg one epoch time: %.2f, total %d epochs time: %.2f" % (np.mean(self.train_hist['per_epoch_time']),
                self.epoch, self.train_hist['total_time'][0]))
        print("Training finish!... save training results")

        self.save()
        utils.loss_plot(self.train_hist, os.path.join(self.save_dir, self.dataset, self.model_name), self.model_name)


    # Evaluation
    def eval(self):
        self.load()
        self.mse = 0
        self.rmse = 0
        for iter, sample in enumerate(self.data_loader):
            real_perchase = sample['u_perchase']
            feature = sample['u_feature']

            if iter == self.data_loader.dataset.__len__() // self.batch_size:
                break
            if(self.z_dim == 0):
                z_ = torch.zeros(0, 0)
            if self.gpu_mode:
                real_perchase, feature, z_ = real_perchase.cuda(), feature.cuda(), z_.cuda()

            # Generate Fake Prechase Vector
            fake_perchase = self.G(z_, feature)

            '''RMES'''
            # masking perchase vector
            mask_r = (real_perchase > 0).float()
            mask_f = (fake_perchase > 0).float()

            mse = nn.MSELoss()
            self.mse += mse(fake_perchase*mask_r, real_perchase*mask_f)
            self.rmse += torch.sqrt(mse(fake_perchase*mask_r, real_perchase*mask_f))
        
        print("MSE:{}\nRMSE:{}".format(self.mse, self.rmse))


    def save(self):
        save_dir = os.path.join(self.save_dir, self.dataset, self.model_name)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        torch.save(self.G.state_dict(), os.path.join(save_dir, self.model_name + '_G.pkl'))
        torch.save(self.D.state_dict(), os.path.join(save_dir, self.model_name + '_D.pkl'))

        with open(os.path.join(save_dir, self.model_name + '_history.pkl'), 'wb') as f:
            pickle.dump(self.train_hist, f)


    def load(self):
        save_dir = os.path.join(self.save_dir, self.dataset, self.model_name)

        self.G.load_state_dict(torch.load(os.path.join(save_dir, self.model_name + '_G.pkl')))
        self.D.load_state_dict(torch.load(os.path.join(save_dir, self.model_name + '_D.pkl')))
