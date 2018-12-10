import torch, time, os, pickle
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, utils

import utils
from dataloader import MovieLensDataset, ToTensor
from network import generator, discriminator


class GCGAN(object):
    def __init__(self, args, device):
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
        self.Ghidden_num = args.Ghidden
        self.Dhidden_num = args.Dhidden
        self.z_dim = 0
        self.num_worker = args.num_worker
        self.device = device

        dataset = MovieLensDataset(dataset=self.dataset,
                                    transform=transforms.Compose([
                                        ToTensor()
                                    ]))

        self.u_feature_num = dataset[0]['u_feature'].shape[0]
        self.v_feature_num = dataset[0]['v_feature'].shape[1]

        # load dataset
        self.data_loader = DataLoader(dataset,
                                    batch_size=self.batch_size,
                                    shuffle=True,
                                    num_workers=self.num_worker)
        data = dataset[0]['u_perchase']
    
        # networks init
        self.G = generator(input_dim=self.z_dim,
                            feature_num=self.u_feature_num,
                            output_dim=data.shape[0],
                            layer_num=self.Glayer_num,
                            hidden_num=self.Ghidden_num).to(self.device)

        self.D = discriminator(num_user=self.batch_size,
                                num_item=data.shape[0],
                                in_features_u=self.u_feature_num,
                                in_features_v=self.v_feature_num,
                                rating=5, hidden_num=self.Dhidden_num,
                                output_dim=1).to(self.device)
        
        self.G_optimizer = optim.SGD(self.G.parameters(), lr=args.lrG)
        self.D_optimizer = optim.SGD(self.D.parameters(), lr=args.lrD)
        self.BCE_loss = nn.BCELoss().to(self.device)


        print('---------- Networks architecture -------------')
        utils.print_network(self.G)
        utils.print_network(self.D)
        print('-----------------------------------------------')

    def train(self):
        print('training start!!')

        self.train_hist = {}
        self.train_hist['D_loss'] = []
        self.train_hist['G_loss'] = []
        self.train_hist['per_epoch_time'] = []
        self.train_hist['total_time'] = []

        self.D.train()
        start_time = time.time()
        y_real_ = torch.ones(1, 1).to(self.device)
        y_fake_ = torch.zeros(1, 1).to(self.device)

        for epoch in range(self.epoch):
            self.G.train()
            epoch_start_time = time.time()

            for iter, sample in enumerate(self.data_loader):
                if iter == self.data_loader.dataset.__len__() // self.batch_size:
                    break
                
                perchase = sample['u_perchase'].to(self.device)
                u_feature = sample['u_feature'].to(self.device)
                v_feature = sample['v_feature'].to(self.device)
                z_ = torch.zeros(0, 0).to(self.device)
                # masking fake perchase vector
                mask = (perchase > 0).float().to(self.device)

                # -----------------update D network-------------------------
                self.D_optimizer.zero_grad()
                G_perchase = self.G(z_, u_feature)

                D_real = self.D(perchase, u_feature, v_feature[0])
                D_fake = self.D(G_perchase*mask, u_feature, v_feature[0])

                D_real_loss = self.BCE_loss(D_real, y_real_)
                D_fake_loss = self.BCE_loss(D_fake, y_fake_)
                D_loss = D_real_loss + D_fake_loss
                D_loss.backward()

                self.train_hist['D_loss'].append(D_loss.item())
                self.D_optimizer.step()
                # -----------------update D network-------------------------

                # -----------------update G network------------------------
                self.G_optimizer.zero_grad()
                G__perchase = self.G(z_, u_feature)

                D_fake = self.D(G__perchase*mask, u_feature, v_feature[0])
                G_loss = self.BCE_loss(D_fake, y_real_)
                G_loss.backward()

                self.train_hist['G_loss'].append(G_loss.item())
                self.G_optimizer.step()
                # -----------------update G network------------------------

                # -----------------output status---------------------------
                if ((iter + 1) % 10) == 0:
                    print("Epoch: [%2d] [%4d/%4d] D_loss: %.8f, G_loss: %.8f" %
                        ((epoch + 1), (iter + 1), self.data_loader.dataset.__len__() // self.batch_size, D_loss.item(), G_loss.item()))
                    print(torch.min(G__perchase), torch.max(G__perchase))
                    

            self.train_hist['per_epoch_time'].append(time.time() - epoch_start_time)

        self.train_hist['total_time'].append(time.time() - start_time)
        print("Avg one epoch time: %.2f, total %d epochs time: %.2f" % (np.mean(self.train_hist['per_epoch_time']),
                self.epoch, self.train_hist['total_time'][0]))


        print("Training finish!... save training results")

        utils.loss_plot(self.train_hist, os.path.join(self.save_dir, self.dataset, self.model_name), self.model_name)
        self.save()
        

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
