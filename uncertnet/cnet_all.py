from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
from tqdm import tqdm

from types import SimpleNamespace

from uncertnet.distal_cnet import distal_err_net, distal_cnet
from utils import errors

class Linear(nn.Module):
    def __init__(self, linear_size, p_dropout):
        """

        Args:
            linear_size (int): Number of nodes in the linear layers.
            p_dropout (float): Dropout probability.
        """
        super(Linear, self).__init__()

        self.relu = nn.ReLU(inplace=True)
        self.l_relu = nn.LeakyReLU(inplace=True)
        self.dropout = nn.Dropout(p_dropout)

        self.w1 = nn.Linear(linear_size, linear_size)
        self.bn1 = nn.BatchNorm1d(linear_size)

        self.w2 = nn.Linear(linear_size, linear_size)
        self.bn2 = nn.BatchNorm1d(linear_size)

    def forward(self, x):
        """Forward operations of the linear block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            y (torch.Tensor): Output tensor.
        """
        h = self.w1(x)
        h = self.bn1(h)
        # h = self.relu(h)
        h = self.l_relu(h)
        h = self.dropout(h)

        h = self.w2(h)
        h = self.bn2(h)
        # h = self.relu(h)
        h = self.l_relu(h)
        h = self.dropout(h)

        y = x + h
        return y

class BaselineModel(nn.Module):
    def __init__(self, linear_size=1024, num_stages=2, p_dropout=0.5, 
                 use_FF=False, num_p_stages=2, p_linear_size=1024):
        """

        Args:
            linear_size (int, optional): Number of nodes in the linear layers. Defaults to 1024.
            num_stages (int, optional): Number to repeat the linear block. Defaults to 2.
            p_dropout (float, optional): Dropout probability. Defaults to 0.5.
            predict_14 (bool, optional): Whether to predict 14 3d-joints. Defaults to False.
        """
        super(BaselineModel, self).__init__()
        self.use_FF = use_FF
        self.num_p_stages = num_p_stages

        input_size = 17 * 3          # Input 3d-joints.
        if use_FF: # Fitness Feedback?
            FF_size = 17 #1
            input_size += FF_size
        output_size = 4 * 3          # Output distal joint errs

        if use_FF:
            embed_size = 128
            # FF 
            # embedding
            self.w_FF_embd = nn.Linear(FF_size, embed_size)
            self.bn_FF_embd = nn.BatchNorm1d(embed_size)
            # linear blocks
            self.w1_FF = nn.Linear(embed_size, p_linear_size)
            self.bn1_FF = nn.BatchNorm1d(p_linear_size)
            # if num_p_stages != 0:
            #     self.linear_FF_stages = [Linear(p_linear_size, p_dropout) for _ in range(num_p_stages)]
            #     self.linear_FF_stages = nn.ModuleList(self.linear_FF_stages)
            # Input 
            # embedding
            self.w_in_embd = nn.Linear(input_size-FF_size, embed_size)
            self.bn_in_embd = nn.BatchNorm1d(embed_size)
            # linear blocks
            self.w1_in = nn.Linear(embed_size, p_linear_size)
            self.bn1_in = nn.BatchNorm1d(p_linear_size)
            # if num_p_stages != 0:
            #     self.linear_in_stages = [Linear(p_linear_size, p_dropout) for _ in range(num_p_stages)]
            #     self.linear_in_stages = nn.ModuleList(self.linear_in_stages)

            # self.w1 = nn.Linear(embed_size*2, linear_size)
            self.w1 = nn.Linear(p_linear_size*2, linear_size)
        else:
            self.w1 = nn.Linear(input_size, linear_size)
        self.bn1 = nn.BatchNorm1d(linear_size)

        self.linear_stages = [Linear(linear_size, p_dropout) for _ in range(num_stages)]
        self.linear_stages = nn.ModuleList(self.linear_stages)

        self.w2 = nn.Linear(linear_size, output_size)

        self.relu = nn.ReLU(inplace=True)
        self.l_relu = nn.LeakyReLU(inplace=True)
        self.dropout = nn.Dropout(p_dropout)

    def forward(self, x):
        """Forward operations of the linear block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            y (torch.Tensor): Output tensor.
        """
        if self.use_FF:
            # pose and FF streams
            FF = x[:, :17]
            FF = self.bn_FF_embd(self.w_FF_embd(FF))
            FF = self.w1_FF(FF)
            FF = self.bn1_FF(FF)
            FF = self.relu(FF)
            # if self.num_p_stages != 0:
            #     for linear in self.linear_FF_stages:
            #         FF = linear(FF)            

            x = x[:, 17:]
            x = self.bn_in_embd(self.w_in_embd(x))
            x = self.w1_in(x)
            x = self.bn1_in(x)
            x = self.relu(x)
            # if self.num_p_stages != 0:
            #     for linear in self.linear_in_stages:
            #         x = linear(x)            

            # Concatenate embedded inputs
            x = torch.cat([x, FF], dim=1)

        y = self.w1(x)
        y = self.bn1(y)
        # y = self.relu(y)
        y = self.l_relu(y)
        y = self.dropout(y)

        # linear blocks
        for linear in self.linear_stages:
            y = linear(y)

        y = self.w2(y)

        return y

class adapt_net():
    '''
    Survive, adapt, overcome
    '''
    def __init__(self, config) -> None:
        self.config = config
        self.device = self.config.device

        # Paths
        self.config.ckpt_name = self.config.hybrIK_version + '_cnet_all.pth'
        if config.use_FF:
            self.config.ckpt_name = self.config.ckpt_name[:-4] + '_FF' + self.config.ckpt_name[-4:]
        # self.config.data_train_path = '{}{}/{}_cnet_hybrik_train.npy'.format(
        #                                                         config.cnet_dataset_path, 
        #                                                         config.trainset,
        #                                                         config.hybrIK_version,)

        # Nets
        self.cnet = BaselineModel(linear_size=512, num_stages=2, p_dropout=0.5,
                                  use_FF=config.use_FF, num_p_stages=1, p_linear_size=512,
                                  ).to(self.device)
        
        # Training
        self.config.train_split = 0.85
        self.config.err_scale = 1000    # 100

        if config.trainset == 'PW3D':
            self.config.lr = 1e-4           # 1e-2, 3e-4
            self.config.weight_decay = 1e-3
        elif config.trainset == 'HP3D':
            self.config.lr = 1e-3          
            self.config.weight_decay = 1e-3
        self.config.batch_size = 1024
        self.config.cnet_train_epochs = 40
        self.config.ep_print_freq = 5

        self.optimizer = torch.optim.Adam(self.cnet.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)
        self.criterion = nn.MSELoss()
        # self.criterion = nn.L1Loss()

        # Misc
        self.distal_kpts = [3, 6, 13, 16,]  # L_Ankle, R_Ankle, L_Wrist, R_Wrist

    def _corr(self, input):
        backbone_pred = input
        '''
        Correct the backbone predictions, no Feedback
        '''
        corr_pred = backbone_pred.detach().clone()
        inp = torch.flatten(backbone_pred, start_dim=1)
        pred_errs = self.cnet(inp) / self.config.err_scale
        pred_errs = pred_errs.reshape(-1, 4, 3) # net output is 4x3 (4 distal joints, 3d) errors
        # subtract from distal joints
        corr_pred[:, self.distal_kpts, :] -= pred_errs
        return corr_pred

    def _corr_FF(self, input):
        '''
        Correct the backbone predictions, with 2D reproj FF
        '''
        poses_2d, backbone_pred = input

        corr_pred = backbone_pred.detach().clone()
        FF_errs = errors.loss_weighted_rep_no_scale(poses_2d, backbone_pred)
        inp = torch.cat([FF_errs, torch.flatten(backbone_pred, start_dim=1)], 1)
        pred_errs = self.cnet(inp) / self.config.err_scale
        pred_errs = pred_errs.reshape(-1, 4, 3) # net output is 4x3 (4 distal joints, 3d) errors

        # subtract from distal joints
        corr_pred[:, self.distal_kpts, :] -= pred_errs
        return corr_pred

    def __call__(self, input):
        if self.config.use_FF:
            return self._corr_FF(input)
        else:
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
            for batch_idx, data in enumerate(gt_trainloader):
                backbone_pred = data[0][:, 0, :].to(self.device)
                target_xyz_17 = data[0][:, 1, :].to(self.device)
                cnet_target = backbone_pred - target_xyz_17
                cnet_target = torch.flatten(cnet_target[:, self.distal_kpts, :], start_dim=1)

                if self.config.use_FF:
                    FF_errs = errors.loss_weighted_rep_no_scale(target_xyz_17[:,:,:2], backbone_pred)
                    inp = torch.cat([FF_errs, torch.flatten(backbone_pred, start_dim=1)], 1)
                else:
                    inp = torch.flatten(backbone_pred, start_dim=1)
                
                out = self.cnet(inp)

                loss = self.criterion(out, cnet_target*self.config.err_scale)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                cnet_train_losses.append(loss.item())
            # Val
            with torch.no_grad():
                cnet_val_losses = []
                for batch_idx, data in enumerate(gt_valloader):
                    backbone_pred = data[0][:, 0, :].to(self.device)
                    target_xyz_17 = data[0][:, 1, :].to(self.device)
                    cnet_target = backbone_pred - target_xyz_17
                    cnet_target = torch.flatten(cnet_target[:, self.distal_kpts, :], start_dim=1)
                    
                    if self.config.use_FF:
                        FF_errs = errors.loss_weighted_rep_no_scale(target_xyz_17[:,:,:2], backbone_pred)
                        inp = torch.cat([FF_errs, torch.flatten(backbone_pred, start_dim=1)], 1)
                    else:
                        inp = torch.flatten(backbone_pred, start_dim=1)

                    out = self.cnet(inp)

                    loss = self.criterion(out, cnet_target*self.config.err_scale)
                    cnet_val_losses.append(loss.item())
                
            mean_train_loss = np.mean(cnet_train_losses)
            mean_val_loss = np.mean(cnet_val_losses)
            # print on some epochs
            print_ep = ep % self.config.ep_print_freq == 0
            losses_str, best_val_str = '', ''
            if print_ep:
                losses_str = f"EP {ep}:    t_loss: {mean_train_loss:.5f}    v_loss: {mean_val_loss:.5f}"
                print(f"EP {ep}:    t_loss: {mean_train_loss:.5f}    v_loss: {mean_val_loss:.5f}")#, end='')
            
            if mean_val_loss < best_val_loss:
                best_val_str = "    ---> best val loss so far, "
                if not print_ep:
                    print(f"EP {ep}:    t_loss: {mean_train_loss:.5f}    v_loss: {mean_val_loss:.5f}", end='')
                print("    ---> best val loss so far, ", end='')
                self.save(self.config.cnet_ckpt_path + self.config.ckpt_name)
                best_val_loss = mean_val_loss
                best_val_ep = ep
            # print()

            # losses_str = f"EP {ep}:    t_loss: {mean_train_loss:.5f}    v_loss: {mean_val_loss:.5f}"
            # best_val_str = "    ---> best val loss so far, " if mean_val_loss < best_val_loss else ""

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

    