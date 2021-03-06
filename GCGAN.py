import torch, time, os, pickle
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms, utils

import utils
from dataloader import MovieLensDataset, ToTensor, random_split
from network import generator, discriminator

from sklearn.metrics import precision_score, recall_score, f1_score


class GCGAN(object):
    def __init__(self, args, device):
        # parameters
        self.epoch = args.epochs
        self.batch_size = args.batch_size
        self.save_dir = args.save_dir
        self.result_dir = args.result_dir
        self.dataset = args.dataset
        self.log_dir = args.log_dir
        self.model_name = "GCGAN"
        self.Glayer_num = args.Glayer
        self.Dlayer_num = args.Dlayer
        self.Ghidden_num = args.Ghidden
        self.z_dim = args.z_dim
        self.num_worker = args.num_worker
        self.device = device

        dataset = MovieLensDataset(dataset=self.dataset,
                                    transform=transforms.Compose([
                                        ToTensor()
                                    ]))
        dataset_num = len(dataset)
        train_num = int(dataset_num*0.8)
        train_dataset, test_dataset = random_split(dataset, [train_num, dataset_num-train_num])

        # load dataset
        self.train_loader = DataLoader(train_dataset,
                                    batch_size=self.batch_size,
                                    shuffle=True,
                                    num_workers=self.num_worker)
        self.test_loader = DataLoader(test_dataset,
                                    batch_size=len(dataset),
                                    shuffle=True,
                                    num_workers=self.num_worker)

        data = dataset[0]['u_perchase']
        self.u_feature_num = dataset[0]['u_feature'].shape[0]
        self.v_feature_num = dataset[0]['v_feature'].shape[1]
    
        # networks init
        
        self.G = generator(input_dim=self.z_dim,
                            feature_num=self.u_feature_num,
                            output_dim=data.shape[0],
                            layer_num=self.Glayer_num,
                            hidden_num=self.Ghidden_num).to(self.device)

        self.D = discriminator(in_features_u=self.u_feature_num,
                                num_item=data.shape[0],
                                in_features_v=self.v_feature_num,
                                rating=5,
                                output_dim=1,
                                layer_num=self.Dlayer_num).to(self.device)
        
        
        self.G_optimizer = optim.SGD(self.G.parameters(), lr=args.lrG)
        self.D_optimizer = optim.SGD(self.D.parameters(), lr=args.lrD)
        self.BCE_loss = nn.BCELoss().to(self.device)
        self.MSE_loss = nn.MSELoss().to(self.device)


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
        self.train_hist['precision'] = []
        self.train_hist['recall'] = []

        self.D.train()
        start_time = time.time()
        y_real_ = torch.ones(self.batch_size, 1).to(self.device)
        y_fake_ = torch.zeros(self.batch_size, 1).to(self.device)

        for epoch in range(self.epoch):
            self.G.train()
            epoch_start_time = time.time()

            for iter, sample in enumerate(self.train_loader):
                if iter == self.train_loader.dataset.__len__() // self.batch_size:
                    break
                
                perchase = sample['u_perchase'].to(self.device)
                u_feature = sample['u_feature'].to(self.device)
                v_feature = sample['v_feature'].to(self.device)
                #z_ = torch.zeros(0, 0).to(self.device)
                z_ = Variable(torch.FloatTensor(np.random.normal(0, 1, (self.batch_size, self.z_dim)))).to(self.device)
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
            
            
            # ---------validation -----------------------------------------
            self.G.eval()
            with torch.no_grad():
                for iter, sample in enumerate(self.test_loader):
                    if iter == self.test_loader.dataset.__len__() // self.batch_size:
                        break

                    real_perchase = sample['u_perchase'].to(self.device)
                    u_feature = sample['u_feature'].to(self.device)
                    #z_ = torch.zeros(0, 0).to(self.device)
                    z_ = Variable(torch.FloatTensor(np.random.normal(0, 1, (self.test_loader.dataset.__len__(), self.z_dim)))).to(self.device)

                    # Generate Fake Prechase Vector
                    fake_perchase = self.G(z_, u_feature)
                    # Top-N Recommend
                    for N in [5]:
                        sorted, indices = torch.topk(fake_perchase, N)
                        # しきい値でバイナリ化
                        true = real_perchase.gather(1, indices)
                        mask = (true > 0).float().to(self.device)
                        pred = (sorted*true) > 3
                        true = true > 3
                        precision = precision_score(true, pred, average='samples')
                        # Recall
                        recall = recall_score(true, pred, average='samples')
                        # f1
                        #f1 = f1_score(true, pred, average="samples")

            self.train_hist['precision'].append(precision)
            #self.train_hist['recall'].append(recall)
            #----------validation -----------------------------------------

            # -----------------output status---------------------------  
            print("Epoch: [%2d] D_loss: %.8f, G_loss: %.8f" %
                    ((epoch + 1), D_loss.item(), G_loss.item()))

            self.train_hist['per_epoch_time'].append(time.time() - epoch_start_time)

        self.train_hist['total_time'].append(time.time() - start_time)
        print("Avg one epoch time: %.2f, total %d epochs time: %.2f" % (np.mean(self.train_hist['per_epoch_time']),
                self.epoch, self.train_hist['total_time'][0]))

        #utils.loss_plot(self.train_hist, os.path.join(self.save_dir, self.dataset, self.model_name), self.model_name)
        #utils.valid_plot(self.train_hist, os.path.join(self.save_dir, self.dataset, self.model_name), self.model_name)
        self.save()
    

    # Evaluation
    def eval(self):
        self.result = {}
        self.result['f1'] = []
        self.result['precision'] = []
        self.result['recall'] = []

        self.G.eval()
        self.load()
        mse = 0
        rmse = 0
        with torch.no_grad():
            for iter, sample in enumerate(self.test_loader):
                if iter == self.test_loader.dataset.__len__() // self.batch_size:
                    break

                real_perchase = sample['u_perchase'].to(self.device)
                u_feature = sample['u_feature'].to(self.device)
                #z_ = torch.zeros(0, 0).to(self.device)
                z_ = Variable(torch.FloatTensor(np.random.normal(0, 1, (self.test_loader.dataset.__len__(), self.z_dim)))).to(self.device)

                # Generate Fake Prechase Vector
                fake_perchase = self.G(z_, u_feature)
                # Top-N Recommend
                for N in [5, 10, 15, 20, 25, 30, 35]:
                    sorted, indices = torch.topk(fake_perchase, N)
                    # しきい値でバイナリ化
                    true = real_perchase.gather(1, indices)
                    mask = (true > 0).float().to(self.device)
                    pred = (sorted*true) > 3
                    true = true > 3
                    f1 = f1_score(true, pred, average="samples")
                    precision = precision_score(true, pred, average="samples")
                    recall = recall_score(true, pred, average="samples")
                    print(precision)

                    #self.result['f1'].append(f1)
                    self.result['precision'].append(precision)
                    #self.result['recall'].append(recall)

                    #---------------------save--------------------------------------
                    #result_df = pd.read_csv('exp.csv')
                    #tmp_se = pd.Series( [self.dataset, self.batch_size, self.Dlayer_num, N, precision, recall, f1], index=result_df.columns )
                    #result_df = result_df.append( tmp_se, ignore_index=True )
                    #result_df.to_csv("exp.csv", index=False)
                    #---------------------save--------------------------------------

                
                # MSE & RMSE
                # masking perchase vector
                #mask_r = (real_perchase > 0).float().to(self.device)
                #mask_f = (fake_perchase > 0).float().to(self.device)
                #mse += self.MSE_loss(fake_perchase*mask_r, real_perchase*mask_f)
                #rmse += torch.sqrt(self.MSE_loss(fake_perchase*mask_r, real_perchase*mask_f))

                #utils.predict_plot(fake_perchase.cpu().numpy(), os.path.join(self.save_dir, self.dataset, self.model_name), self.model_name+'_fake')
                #utils.predict_plot(real_perchase.cpu().numpy(), os.path.join(self.save_dir, self.dataset, self.model_name), self.model_name+'_real')
        
            #print("MSE:{}\nRMSE:{}".format(mse, rmse))
        

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

        self.G.load_state_dict(torch.load(os.path.join(save_dir, self.model_name + '_G.pkl'), map_location=self.device))
        self.D.load_state_dict(torch.load(os.path.join(save_dir, self.model_name + '_D.pkl'), map_location=self.device))
