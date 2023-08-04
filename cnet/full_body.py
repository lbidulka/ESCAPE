import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import copy

from .residual import BaselineModel
import utils.errors as errors
from core.cnet_dataset import cnet_pose_dataset, hflip_keypoints

class adapt_net():
    '''
    Survive, adapt, overcome
    '''
    def __init__(self, config, 
                target_kpts=[3, 6, 13, 16,], 
                in_kpts=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16],
                R=False,    # R-CNet ?
                ) -> None:
        self.config = copy.copy(config)
        self.device = self.config.device
        self.R = R
        self.pred_errs = config.pred_errs   # True: predict distal joint errors, False: predict 3d-joints directly
        self.corr_dims = [0,1,2]  # 3d-joint dims to correct   0,1,2

        # Paths
        self.config.ckpt_name = '' #self.config.hybrIK_version 
        if R:
            self.config.ckpt_name += '_rcnet'
        else:
            self.config.ckpt_name += '_cnet'
        if self.config.use_multi_distal:
            self.config.ckpt_name += '_md.pth'
        else:
            self.config.ckpt_name += '_all.pth'

        # Kpt definition
        self.in_kpts = in_kpts
        self.target_kpts = target_kpts

        self.in_kpts.sort()

        # Nets
        if R:
            # self.cnet = BaselineModel(linear_size=512, num_stages=5, p_dropout=0.5, 
            #                       num_in_kpts=len(self.in_kpts), num_out_kpts=len(self.target_kpts)).to(self.device)
            self.cnet = BaselineModel(linear_size=1024, num_stages=4, p_dropout=0.5, 
                                  num_in_kpts=len(self.in_kpts), num_out_kpts=len(self.target_kpts)).to(self.device)
            self.config.cnet_train_epochs = 100  # 200
            self.config.lr = 1e-3
        else:
            # self.cnet = BaselineModel(linear_size=1024, num_stages=4, p_dropout=0.5, 
            #                         num_in_kpts=len(self.in_kpts), num_out_kpts=len(self.target_kpts)).to(self.device)
            self.cnet = BaselineModel(linear_size=1024, num_stages=4, p_dropout=0.5, 
                                    num_in_kpts=len(self.in_kpts), num_out_kpts=len(self.target_kpts)).to(self.device)
            self.config.cnet_train_epochs = 50  # 200
            self.config.lr = 1e-3
        
        # Training
        self.config.train_split = 0.75   # 0.85 
        self.config.err_scale = 1000    # 100

        # self.config.weight_decay = 1e-3
        self.config.batch_size = 4096

        self.config.ep_print_freq = 5

        self.optimizer = torch.optim.Adam(self.cnet.parameters(), lr=self.config.lr)#, weight_decay=self.config.weight_decay)
        self.criterion = nn.MSELoss()

    def _loss(self, backbone_pred, cnet_out, target_xyz_17):
        '''
        Loss function
        '''
        if self.pred_errs: 
            cnet_target = backbone_pred - target_xyz_17  # predict errors
        else:
            cnet_target = target_xyz_17.clone() # predict kpts
        cnet_target = torch.flatten(cnet_target[:, self.target_kpts, :], start_dim=1)
        loss = self.criterion(cnet_out, cnet_target*self.config.err_scale)
        return loss

    def _corr(self, backbone_pred):
        '''
        Correct the backbone predictions, no Feedback
        '''
        inp = torch.flatten(backbone_pred[:,self.in_kpts], start_dim=1)
        pred_errs = self.cnet(inp) / self.config.err_scale
        pred_errs = pred_errs.reshape(-1, len(self.target_kpts), 3) # net output is 4x3 (4 distal joints, 3d) errors
        corr_pred = backbone_pred.detach().clone()
        if self.pred_errs:
            for dim in self.corr_dims:
                corr_pred[:, self.target_kpts, dim] -= self.config.corr_step_size*pred_errs[..., dim]
        else:
            for dim in self.corr_dims:
                # TODO: ADD STEP SIZE TO THIS CORRECTION
                corr_pred[:, self.target_kpts, dim] = pred_errs[:, self.target_kpts, dim]
        return corr_pred

    def __call__(self, cnet_in):
        for i in range(self.config.corr_steps):
            corr_pred = self._corr(cnet_in)
            cnet_in = corr_pred
        return corr_pred

    def _load_data_files(self,):
        '''
        Fetch and load the training files
        '''
        data_all = []
        for i, (trainset_path, backbone_scale) in enumerate(zip(self.config.cnet_trainset_paths,
                                                                self.config.cnet_trainset_scales)):
            if self.R:
                trainset_path = trainset_path.replace('cnet', 'rcnet')
            
            data = torch.from_numpy(np.load(trainset_path)).float()
            
            if ('MPii' in trainset_path) and ('mmlab' in trainset_path):
                # samples with all 0 labels need to be removed
                idx = np.where(data[1,:,:].sum(axis=1) != 0)
                data = data[:, idx, :]
                data = data.squeeze()

            if self.config.train_datalims[i] is not None:
                # get random subset of data
                data = data[:, np.random.choice(data.shape[1], min(self.config.train_datalims[i], data.shape[1]), replace=False), :]               

            # scale according to the backbone
            data[:2] *= backbone_scale
            data_all.append(data[:3])   # TEMP: don't include image paths, if they are present
        data_all = torch.cat(data_all, dim=1)
        return data_all

    def train(self,):
        data_all = self._load_data_files()

        # shuffle & split data
        idx = torch.randperm(data_all.shape[1])
        data_all = data_all[:, idx, :]
        len_train = int(len(data_all[0]) * self.config.train_split)
        data_train, data_val = data_all[:, :len_train, :], data_all[:, len_train:, :]

        data_train = data_train.permute(1,0,2)
        data_val = data_val.permute(1,0,2)
        data_train = data_train.reshape(data_train.shape[0], 3, -1, 3) # batch, 2, kpts, xyz)
        data_val = data_val.reshape(data_val.shape[0], 3, -1, 3)

        train_transform = hflip_keypoints()
        gt_trainset = cnet_pose_dataset(data_train, transform=train_transform)
        gt_trainloader = torch.utils.data.DataLoader(gt_trainset, batch_size=self.config.batch_size, shuffle=True, 
                                                     num_workers=8, drop_last=False, pin_memory=True)
                                                    # drop_last=False, pin_memory=True)
        gt_valset = cnet_pose_dataset(data_val)
        gt_valloader = torch.utils.data.DataLoader(gt_valset, batch_size=self.config.batch_size, shuffle=True, 
                                                   num_workers=8, drop_last=False, pin_memory=True)
                                                # drop_last=False, pin_memory=True)

        # Train
        print('\n--- Training: {} ---'.format('R-CNet' if self.R else 'CNet'))
        eps = self.config.cnet_train_epochs
        best_val_loss = 1e10
        best_val_ep = 0
        for ep in tqdm(range(eps), dynamic_ncols=True):
            cnet_train_losses = []
            self.cnet.train()
            for batch_idx, data in enumerate(gt_trainloader):
                backbone_pred = data[:, 0, :].to(self.device)
                target_xyz_17 = data[:, 1, :].to(self.device)
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
                    backbone_pred = data[:, 0, :].to(self.device)
                    target_xyz_17 = data[:, 1, :].to(self.device)
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
    
    def write_train_preds(self,):
        '''
        Get preds on training data and saves them to npy file, for
        training RCNet
        '''
        self.load_cnets()
        self.cnet.eval()
        # Save preds for each trainset
        for trainset_path in self.config.cnet_trainset_paths:
            data_train = torch.from_numpy(np.load(trainset_path)).float()
            data_train = data_train.permute(1,0,2)
            data_train = data_train.reshape(data_train.shape[0], data_train.shape[1], -1, 3) # batch, 2, kpts, xyz)
            gt_trainset = torch.utils.data.TensorDataset(data_train)
            gt_trainloader = torch.utils.data.DataLoader(gt_trainset, batch_size=self.config.batch_size, 
                                                         shuffle=False, num_workers=8, drop_last=False, 
                                                         pin_memory=True)
            # Iterate over train data
            print('\n--- Getting Preds with {} on {} ---'.format(('R-CNet' if self.R else 'CNet'),
                                                                 trainset_path))
            with torch.no_grad():
                for batch_idx, data in enumerate(tqdm(gt_trainloader)):
                    backbone_pred = data[0][:, 0, :].to(self.device)
                    corr_pred = self(backbone_pred)
                    if batch_idx == 0:
                        train_preds = corr_pred.cpu().numpy()
                    else:
                        train_preds = np.concatenate((train_preds, corr_pred.cpu().numpy()), axis=0)

            data_out = np.load(trainset_path)
            data_out[0] = train_preds.reshape(train_preds.shape[0], -1)
            np.save(trainset_path.replace('cnet', 'rcnet'), data_out)

    def load_cnets(self, print_str=True):
        '''
        Load the best validation checkpoints
        '''
        load_path = self.config.cnet_ckpt_path + self.config.ckpt_name
        if print_str: 
            print("Loading {} from: {}".format('R-CNet' if self.R else 'CNet', load_path))
        all_net_ckpt_dict = torch.load(load_path)
        self.cnet.load_state_dict(all_net_ckpt_dict) 
    
    def save(self, path):
        print('saving {} to: {}'.format('R-CNet' if self.R else 'CNet', path))
        torch.save(self.cnet.state_dict(), path)
    
    def eval(self,):
        self.cnet.eval()

    