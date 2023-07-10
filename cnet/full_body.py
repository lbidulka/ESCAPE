import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import copy

from .residual import BaselineModel
import utils.errors as errors

class adapt_net():
    '''
    Survive, adapt, overcome
    '''
    def __init__(self, config, 
                target_kpts=[3, 6, 13, 16,], 
                in_kpts=[0,1,2, 4,5, 7,8,9,10,11,12, 14,15, ],
                # pred_errs = True,
                ) -> None:
        self.config = copy.copy(config)
        self.device = self.config.device

        self.pred_errs = config.pred_errs   # True: predict distal joint errors, False: predict 3d-joints directly
        self.corr_dims = [0,1,2]  # 3d-joint dims to correct   0,1,2

        # Paths
        self.config.ckpt_name = self.config.hybrIK_version + '_cnet_all.pth'

        # Kpt definition
        # self.distal_kpts = target_kpts # [3, 6, 13, 16,]  # L_Ankle, R_Ankle, L_Wrist, R_Wrist
        # self.exclude_in_kpts = [] #self.distal_kpts
        # self.in_kpts = in_kpts #[val for val in range(17) if val not in self.exclude_in_kpts]
        self.in_kpts = in_kpts # [0, 1, 2, 3]
        self.distal_kpts = target_kpts # [3,]

        self.in_kpts.sort()

        # Nets
        # self.cnet = BaselineModel(linear_size=512, num_stages=2, p_dropout=0.5,).to(self.device)
        self.cnet = BaselineModel(linear_size=1024, num_stages=4, p_dropout=0.5, 
                                  num_in_kpts=len(self.in_kpts), num_out_kpts=len(self.distal_kpts)).to(self.device)
        
        # Training
        self.config.train_split = 0.8   # 0.85
        self.config.err_scale = 1000    # 100

        self.config.lr = 1e-2
        # self.config.weight_decay = 1e-3
        self.config.batch_size = 1024
        self.config.cnet_train_epochs = 50  # 200
        self.config.ep_print_freq = 5

        self.optimizer = torch.optim.Adam(self.cnet.parameters(), lr=self.config.lr)#, weight_decay=self.config.weight_decay)
        self.criterion = nn.MSELoss()
        # self.criterion = nn.L1Loss()

    def _loss(self, backbone_pred, cnet_out, target_xyz_17):
        '''
        Loss function
        '''
        if self.pred_errs: 
            cnet_target = backbone_pred - target_xyz_17  # predict errors
        else:
            cnet_target = target_xyz_17.clone() # predict kpts
        cnet_target = torch.flatten(cnet_target[:, self.distal_kpts, :], start_dim=1)
        loss = self.criterion(cnet_out, cnet_target*self.config.err_scale)
        return loss

    def _corr(self, input):
        backbone_pred = input
        '''
        Correct the backbone predictions, no Feedback
        '''
        corr_pred = backbone_pred.detach().clone()
        inp = torch.flatten(backbone_pred[:,self.in_kpts], start_dim=1)
        pred_errs = self.cnet(inp) / self.config.err_scale
        pred_errs = pred_errs.reshape(-1, len(self.distal_kpts), 3) # net output is 4x3 (4 distal joints, 3d) errors

        if self.pred_errs:
            # corr_pred[:, self.distal_kpts, self.corr_dims_s:self.corr_dims_e] -= pred_errs[..., self.corr_dims_s:self.corr_dims_e] # subtract from distal joints

            for dim in self.corr_dims:
                corr_pred[:, self.distal_kpts, dim] -= pred_errs[..., dim]
        else: 
            # corr_pred[:, self.distal_kpts, self.corr_dims_s:self.corr_dims_e] = pred_errs[..., self.corr_dims_s:self.corr_dims_e] # predict kpts directly
            for dim in self.corr_dims:
                corr_pred[:, self.distal_kpts, dim] = pred_errs[:, self.distal_kpts, dim]
        return corr_pred

    def __call__(self, input):
        return self._corr(input)

    def train(self,):
        # data_path = self.config.cnet_dataset_path + 'cnet_hybrik_train.npy'
        if self.config.train_datalim is not None:
            data_all = torch.from_numpy(np.load(self.config.cnet_trainset_path)).float()[:, :self.config.train_datalim]
        else:
            data_all = torch.from_numpy(np.load(self.config.cnet_trainset_path)).float()

        len_train = int(len(data_all[0]) * self.config.train_split)
        data_train, data_val = data_all[:, :len_train, :], data_all[:, len_train:, :]

        data_train = data_train.permute(1,0,2)
        data_val = data_val.permute(1,0,2)
        data_train = data_train.reshape(data_train.shape[0], 3, -1, 3) # batch, 2, kpts, xyz)
        data_val = data_val.reshape(data_val.shape[0], 3, -1, 3)

        gt_trainset = torch.utils.data.TensorDataset(data_train)
        gt_trainloader = torch.utils.data.DataLoader(gt_trainset, batch_size=self.config.batch_size, shuffle=True, 
                                                     num_workers=16, drop_last=False, pin_memory=True)
        gt_valset = torch.utils.data.TensorDataset(data_val)
        gt_valloader = torch.utils.data.DataLoader(gt_valset, batch_size=self.config.batch_size, shuffle=True, 
                                                   num_workers=16, drop_last=False, pin_memory=True)

        # Train
        eps = self.config.cnet_train_epochs
        best_val_loss = 1e10
        best_val_ep = 0
        for ep in tqdm(range(eps), dynamic_ncols=True):
            cnet_train_losses = []
            self.cnet.train()
            for batch_idx, data in enumerate(gt_trainloader):
                backbone_pred = data[0][:, 0, :].to(self.device)
                target_xyz_17 = data[0][:, 1, :].to(self.device)
                inp = torch.flatten(backbone_pred[:,self.in_kpts], start_dim=1)
                out = self.cnet(inp)
                loss = self._loss(backbone_pred, out, target_xyz_17)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                cnet_train_losses.append(loss.item())
            # Val
            self.cnet.eval()
            with torch.no_grad():
                cnet_val_losses = []
                for batch_idx, data in enumerate(gt_valloader):
                    backbone_pred = data[0][:, 0, :].to(self.device)
                    target_xyz_17 = data[0][:, 1, :].to(self.device)
                    inp = torch.flatten(backbone_pred[:,self.in_kpts], start_dim=1)
                    out = self.cnet(inp)
                    loss = self._loss(backbone_pred, out, target_xyz_17)
                    cnet_val_losses.append(loss.item())
                
            mean_train_loss = np.mean(cnet_train_losses)
            mean_val_loss = np.mean(cnet_val_losses)
            # print on some epochs
            print_ep = ep % self.config.ep_print_freq == 0
            if print_ep:
                print(f"EP {ep}:    t_loss: {mean_train_loss:.5f}    v_loss: {mean_val_loss:.5f}")#, end='')
            
            if mean_val_loss < best_val_loss:
                if not print_ep:
                    print(f"EP {ep}:    t_loss: {mean_train_loss:.5f}    v_loss: {mean_val_loss:.5f}", end='')
                print("    ---> best val loss so far, ", end='')
                self.save(self.config.cnet_ckpt_path + self.config.ckpt_name)
                best_val_loss = mean_val_loss
                best_val_ep = ep

        print(f"EP {ep}:    t_loss: {mean_train_loss:.5f}    v_loss: {mean_val_loss:.5f}")
        print("|| Best val loss: {:.5f} at ep: {} ||".format(best_val_loss, best_val_ep))
        return

    def load_cnets(self, print_str=True):
        '''
        Load the best validation checkpoints
        '''
        load_path = self.config.cnet_ckpt_path + self.config.ckpt_name
        if print_str: print("\nLoading cnet from: ", load_path)
        all_net_ckpt_dict = torch.load(load_path)
        self.cnet.load_state_dict(all_net_ckpt_dict) 
    
    def save(self, path):
        print("saving cnet to: ", path)
        torch.save(self.cnet.state_dict(), path)
    
    def eval(self,):
        self.cnet.eval()

    