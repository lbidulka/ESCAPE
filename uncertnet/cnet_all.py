from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
from tqdm import tqdm

from types import SimpleNamespace

from uncertnet.distal_cnet import distal_err_net, distal_cnet

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
    def __init__(self, linear_size=1024, num_stages=2, p_dropout=0.5,):
        """

        Args:
            linear_size (int, optional): Number of nodes in the linear layers. Defaults to 1024.
            num_stages (int, optional): Number to repeat the linear block. Defaults to 2.
            p_dropout (float, optional): Dropout probability. Defaults to 0.5.
            predict_14 (bool, optional): Whether to predict 14 3d-joints. Defaults to False.
        """
        super(BaselineModel, self).__init__()

        input_size = 17 * 3  # Input 3d-joints.
        output_size = 4 * 3  # Output distal joint errs

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

        self.relu = nn.ReLU(inplace=True)

        self.fc1 = nn.Linear(input_size, 1024)

        self.out = nn.Linear(1024, output_size)

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
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Paths
        self.config.ckpt_name = self.config.hybrIK_version + 'cnet_all.pth'
        self.config.data_path = '{}{}_cnet_hybrik_train.npy'.format(config.cnet_dataset_path, 
                                                                 config.hybrIK_version,)

        # Net
        self.cnet = BaselineModel(linear_size=128, num_stages=3).to(self.device)
        # self.cnet = dumb_mlp().to(self.device)        
        
        # Training
        self.config.ep_print_freq = 1
        self.config.err_scale = 1000    # 100
        self.config.lr = 1e-3           # 1e-2
        self.config.cnet_train_epochs = 15

        self.optimizer = torch.optim.Adam(self.cnet.parameters(), lr=self.config.lr)
        self.criterion = nn.MSELoss()
        # self.criterion = nn.L1Loss()

        # Misc
        self.distal_kpts = [3, 6, 13, 16,]  # L_Ankle, R_Ankle, L_Wrist, R_Wrist
    
    def __call__(self, backbone_pred):
        '''
        Correct the backbone predictions
        '''
        corr_pred = copy.deepcopy(backbone_pred)
        inp = torch.flatten(corr_pred, start_dim=1)
        pred_errs = self.cnet(inp) / self.config.err_scale
        pred_errs = pred_errs.reshape(-1, 4, 3) # net output is 4x3 (4 distal joints, 3d) errors
        # subtract from distal joints
        corr_pred[:, self.distal_kpts, :] -= pred_errs
        return corr_pred               

    def train(self,):
        # data_path = self.config.cnet_dataset_path + 'cnet_hybrik_train.npy'
        if self.config.train_datalim is not None:
            data_all = torch.from_numpy(np.load(self.config.data_path)).float()[:, :self.config.train_datalim]
        else:
            data_all = torch.from_numpy(np.load(self.config.data_path)).float()

        len_train = int(len(data_all[0]) * 0.7)
        data_train, data_val = data_all[:, :len_train, :], data_all[:, len_train:, :]

        data_train = data_train.permute(1,0,2)
        data_val = data_val.permute(1,0,2)
        data_train = data_train.reshape(data_train.shape[0], 2, -1, 3) # batch, 2, kpts, xyz)
        data_val = data_val.reshape(data_val.shape[0], 2, -1, 3)

        # Data
        batch_size = 1024
        gt_trainset = torch.utils.data.TensorDataset(data_train)
        gt_trainloader = torch.utils.data.DataLoader(gt_trainset, batch_size=batch_size, shuffle=True, num_workers=16, drop_last=False, pin_memory=True)
        gt_valset = torch.utils.data.TensorDataset(data_val)
        gt_valloader = torch.utils.data.DataLoader(gt_valset, batch_size=batch_size, shuffle=True, num_workers=16, drop_last=False, pin_memory=True)

        # Train
        eps = self.config.cnet_train_epochs
        best_val_loss = 1e10
        for ep in tqdm(range(eps), dynamic_ncols=True):
            cnet_train_losses = []
            for batch_idx, data in enumerate(gt_trainloader):
                backbone_pred = data[0][:, 0, :].to('cuda')
                target_xyz_17 = data[0][:, 1, :].to('cuda')
                cnet_target = backbone_pred - target_xyz_17
                cnet_target = torch.flatten(cnet_target[:, self.distal_kpts, :], start_dim=1)
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
                    backbone_pred = data[0][:, 0, :].to('cuda')
                    target_xyz_17 = data[0][:, 1, :].to('cuda')
                    cnet_target = backbone_pred - target_xyz_17
                    cnet_target = torch.flatten(cnet_target[:, self.distal_kpts, :], start_dim=1)
                    inp = torch.flatten(backbone_pred, start_dim=1)
                    out = self.cnet(inp)

                    loss = self.criterion(out, cnet_target*self.config.err_scale)
                    cnet_val_losses.append(loss.item())
                
            mean_train_loss = np.mean(cnet_train_losses)
            mean_val_loss = np.mean(cnet_val_losses)
            # print on some epochs
            if ep % self.config.ep_print_freq == 0:
                print(f"EP {ep}:    t_loss: {mean_train_loss:.5f}    v_loss: {mean_val_loss:.5f}")
            
            if mean_val_loss < best_val_loss:
                print(" ---> best val loss so far, saving model...")
                self.save(self.config.cnet_ckpt_path + self.config.ckpt_name)
                best_val_loss = mean_val_loss
        return

    def load_cnets(self,):
        '''
        Load the best validation checkpoints
        '''
        load_path = self.config.cnet_ckpt_path + self.config.ckpt_name
        print("Loading cnet from: ", load_path)
        all_net_ckpt_dict = torch.load(load_path)
        self.cnet.load_state_dict(all_net_ckpt_dict) 
    
    def save(self, path):
        print("Saving cnet to: ", path)
        torch.save(self.cnet.state_dict(), path)
    
    def eval(self,):
        self.cnet.eval()

    