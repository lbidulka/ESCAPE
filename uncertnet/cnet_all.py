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

torch.backends.cudnn.benchmark = True

class Linear(nn.Module):
    def __init__(self, linear_size, p_dropout):
        """

        Args:
            linear_size (int): Number of nodes in the linear layers.
            p_dropout (float): Dropout probability.
        """
        super(Linear, self).__init__()

        self.relu = nn.ReLU(inplace=True)
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
        h = self.relu(h)
        h = self.dropout(h)

        h = self.w2(h)
        h = self.bn2(h)
        h = self.relu(h)
        h = self.dropout(h)

        y = x + h
        return y

class BaselineModel(nn.Module):
    def __init__(self, linear_size=1024, num_stages=2, p_dropout=0.5, use_FF=False):
        """

        Args:
            linear_size (int, optional): Number of nodes in the linear layers. Defaults to 1024.
            num_stages (int, optional): Number to repeat the linear block. Defaults to 2.
            p_dropout (float, optional): Dropout probability. Defaults to 0.5.
            predict_14 (bool, optional): Whether to predict 14 3d-joints. Defaults to False.
        """
        super(BaselineModel, self).__init__()

        input_size = 17 * 3          # Input 3d-joints.
        if use_FF: input_size += 1   # Fitness Feedback?
        output_size = 4 * 3          # Output distal joint errs

        self.w1 = nn.Linear(input_size, linear_size)
        self.bn1 = nn.BatchNorm1d(linear_size)

        self.linear_stages = [Linear(linear_size, p_dropout) for _ in range(num_stages)]
        self.linear_stages = nn.ModuleList(self.linear_stages)

        self.w2 = nn.Linear(linear_size, output_size)

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p_dropout)

    def forward(self, x):
        """Forward operations of the linear block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            y (torch.Tensor): Output tensor.
        """
        y = self.w1(x)
        y = self.bn1(y)
        y = self.relu(y)
        y = self.dropout(y)

        # linear blocks
        for linear in self.linear_stages:
            y = linear(y)

        y = self.w2(y)

        return y
    
class dumb_mlp(nn.Module):
    '''
    debug simple mlp
    '''
    def __init__(self):
        super().__init__()
        input_size = 17 * 3  # Input 3d-joints.
        output_size = 4 * 3  # Output distal joint errs

        self.fc1 = nn.Linear(input_size, output_size)

        self.relu = nn.ReLU(inplace=True)
        # self.fc1 = nn.Linear(input_size, 32)
        # self.out = nn.Linear(32, output_size)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.out(x)
        return x
    

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
        self.config.data_path = '{}{}_cnet_hybrik_train.npy'.format(config.cnet_dataset_path, 
                                                                 config.hybrIK_version,)

        # Nets
        self.cnet = BaselineModel(linear_size=128, num_stages=2, use_FF=config.use_FF).to(self.device)
        # self.cnet = dumb_mlp().to(self.device)
        # self.cnet = torch.ones(1, 1).to(self.device)
        
        # Training
        self.config.train_split = 0.85
        self.config.err_scale = 1000    # 100

        self.config.lr = 3e-4           # 1e-2
        self.config.weight_decay = 1e-5
        self.config.batch_size = 1024
        self.config.cnet_train_epochs = 60
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
        corr_pred = copy.deepcopy(backbone_pred)
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

        corr_pred = copy.deepcopy(backbone_pred)
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
            data_all = torch.from_numpy(np.load(self.config.data_path)).float()[:, :self.config.train_datalim]
        else:
            data_all = torch.from_numpy(np.load(self.config.data_path)).float()

        len_train = int(len(data_all[0]) * self.config.train_split)
        data_train, data_val = data_all[:, :len_train, :], data_all[:, len_train:, :]

        data_train = data_train.permute(1,0,2)
        data_val = data_val.permute(1,0,2)
        data_train = data_train.reshape(data_train.shape[0], 2, -1, 3) # batch, 2, kpts, xyz)
        data_val = data_val.reshape(data_val.shape[0], 2, -1, 3)

        gt_trainset = torch.utils.data.TensorDataset(data_train)
        gt_trainloader = torch.utils.data.DataLoader(gt_trainset, batch_size=self.config.batch_size, shuffle=True, 
                                                     num_workers=16, drop_last=False)#, pin_memory=True)
        gt_valset = torch.utils.data.TensorDataset(data_val)
        gt_valloader = torch.utils.data.DataLoader(gt_valset, batch_size=self.config.batch_size, shuffle=True, 
                                                   num_workers=16, drop_last=False)#, pin_memory=True)

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

    def load_cnets(self,):
        '''
        Load the best validation checkpoints
        '''
        load_path = self.config.cnet_ckpt_path + self.config.ckpt_name
        print("\nLoading cnet from: ", load_path)
        all_net_ckpt_dict = torch.load(load_path)
        self.cnet.load_state_dict(all_net_ckpt_dict) 
    
    def save(self, path):
        print("saving cnet to: ", path)
        torch.save(self.cnet.state_dict(), path)
    
    def eval(self,):
        self.cnet.eval()

    