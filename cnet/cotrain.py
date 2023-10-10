import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import copy


from .full_body import adapt_net
from core.cnet_dataset import cnet_pose_dataset, hflip_keypoints

class CoTrainer():
    ''' 
    CoTrainer class for co-training of CNet & RCNet
    '''

    def __init__(self, cnet, rcnet) -> None:        
        self.cnet = cnet
        self.rcnet = rcnet

        self.config = self.cnet.config
        self.device = self.config.device

        self.cnet_lr = 5e-5
        self.rcnet_lr = 1e-3

        self.cotrain_eps = 6

    def get_dataloaders(self, pretrain_AMASS=False):
        data_train, data_val = self.cnet._load_data_files(pretrain_AMASS)

        data_train = data_train.permute(1,0,2)
        data_val = data_val.permute(1,0,2)
        data_train = data_train.reshape(data_train.shape[0], 3, -1, 3) # batch, 2, kpts, xyz)
        data_val = data_val.reshape(data_val.shape[0], 3, -1, 3)

        # data_train = data_train[:5]

        train_transform = hflip_keypoints()
        train_set = cnet_pose_dataset(data_train, transform=train_transform)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=self.config.batch_size, shuffle=True, 
                                                     num_workers=8, drop_last=False, pin_memory=True)
                                                    # drop_last=False, pin_memory=True)
        val_set = cnet_pose_dataset(data_val)
        val_loader = torch.utils.data.DataLoader(val_set, batch_size=self.config.batch_size, shuffle=True, 
                                                   num_workers=8, drop_last=False, pin_memory=True)
                                                # drop_last=False, pin_memory=True)
        
        return train_loader, val_loader

    def train(self, ):
        train_loader, val_loader = self.get_dataloaders()

        self.cnet_optim = torch.optim.Adam(self.cnet.cnet.parameters(), lr=self.cnet_lr)
        self.rcnet_optim = torch.optim.Adam(self.rcnet.cnet.parameters(), lr=self.rcnet_lr)

        self.criterion = nn.MSELoss(reduction='none')
        self.cnet.criterion = self.criterion
        self.rcnet.criterion = self.criterion

        for ep in tqdm(range(self.cotrain_eps), dynamic_ncols=True):
            cnet_train_losses, rcnet_train_losses = [], []
            self.cnet.cnet.train()
            self.rcnet.cnet.train()
            # Train
            for batch_idx, data in enumerate(train_loader):
                backbone_pred = data[:,0,:].to(self.device)
                target_xyz_17 = data[:,1,:].to(self.device)

                # CNET
                # get cnet & rcnet corrected samples: Corr_sample, RCorr_sample
                self.cnet_optim.zero_grad()
                cnet_in = torch.flatten(backbone_pred[:,self.cnet.in_kpts], start_dim=1)
                cnet_out = self.cnet.cnet(cnet_in)
                cnet_corr = backbone_pred.detach().clone()
                cnet_corr[:,self.cnet.target_kpts] -= cnet_out.reshape(-1, len(self.cnet.target_kpts), 3) / self.config.err_scale

                rcnet_in = torch.flatten(cnet_corr[:,self.rcnet.in_kpts], start_dim=1)
                rcnet_out = self.rcnet.cnet(rcnet_in)
                rcnet_corr = cnet_corr.detach().clone()
                rcnet_corr[:,self.rcnet.target_kpts] -= rcnet_out.reshape(-1, len(self.rcnet.target_kpts), 3) / self.config.err_scale
                # update cnet: Loss = (Corr_sample distal GT MSE) + (RCorr_sample proximal GT MSE)
                loss_cnet_cnet = self.cnet._loss(backbone_pred, cnet_out, target_xyz_17)
                loss_cnet_rcnet = self.rcnet._loss(backbone_pred, rcnet_out, target_xyz_17)
                cnet_loss = loss_cnet_cnet + loss_cnet_rcnet
                # step
                cnet_loss.backward()
                self.cnet_optim.step()
                cnet_train_losses.append(loss_cnet_cnet.item())
                rcnet_train_losses.append(loss_cnet_rcnet.item())

                # RCNET
                # get cnet & rcnet corrected samples: Corr_sample, RCorr_sample
                self.rcnet_optim.zero_grad()
                cnet_in = torch.flatten(backbone_pred[:,self.cnet.in_kpts], start_dim=1)
                cnet_out = self.cnet.cnet(cnet_in)
                cnet_corr = backbone_pred.detach().clone()
                cnet_corr[:,self.cnet.target_kpts] -= cnet_out.reshape(-1, len(self.cnet.target_kpts), 3) / self.config.err_scale

                rcnet_in = torch.flatten(cnet_corr[:,self.rcnet.in_kpts], start_dim=1)
                rcnet_out = self.rcnet.cnet(rcnet_in)
                rcnet_corr = cnet_corr.detach().clone()
                rcnet_corr[:,self.rcnet.target_kpts] -= rcnet_out.reshape(-1, len(self.rcnet.target_kpts), 3) / self.config.err_scale
                # update rcnet: Loss = (Corr_sample distal GT MSE) + (RCorr_sample proximal GT MSE)
                loss_cnet_cnet = self.cnet._loss(backbone_pred, cnet_out, target_xyz_17)
                loss_cnet_rcnet = self.rcnet._loss(backbone_pred, rcnet_out, target_xyz_17)
                rcnet_loss = loss_cnet_cnet + loss_cnet_rcnet
                # step
                rcnet_loss.backward()
                self.rcnet_optim.step()
                cnet_train_losses.append(loss_cnet_cnet.item())
                rcnet_train_losses.append(loss_cnet_rcnet.item())

            mean_cnet_train_loss = np.mean(cnet_train_losses)
            mean_rcnet_train_loss = np.mean(rcnet_train_losses)
            out_str = f"EP {ep:3d}:    cnet_t_loss: {mean_cnet_train_loss:12.5f} rcnet_t_loss: {mean_rcnet_train_loss:12.5f}"
            print(out_str)

            # save
            self.cnet.save()
            self.rcnet.save()




