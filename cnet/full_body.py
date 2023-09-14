import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import copy

from .residual import BaselineModel
import utils.errors as errors
from utils.convert_pose_2kps import get_smpl_l2ws
from core.cnet_dataset import cnet_pose_dataset, hflip_keypoints

class adapt_net():
    '''
    Survive, adapt, overcome
    '''
    def __init__(self, config, 
                target_kpts=[3, 6, 13, 16,], 
                in_kpts=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16],
                R=False,    # R-CNet ?
                lr=None, eps=None, num_stages=None, lin_size=None,
                ) -> None:
        self.config = copy.copy(config)
        self.device = self.config.device
        self.R = R
        self.pred_errs = config.pred_errs   # True: predict distal joint errors, False: predict 3d-joints directly
        self.corr_dims = [0,1,2]  # 3d-joint dims to correct   0,1,2
        # self.corr_dims = config.RCNet_corr_dims if R else config.CNet_corr_dims

        # Paths
        self.config.ckpt_name = '' 
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
            if self.config.pretrain_AMASS:
                net_num_stages = 4 if not num_stages else num_stages
                net_lin_size = 1024 if not lin_size else lin_size
                self.config.cnet_train_epochs = 10 if not eps else eps #3
                self.config.lr = 1e-4 if not lr else lr # 5e-4
            else:
                if self.config.cotrain:
                    net_num_stages = 4
                    net_lin_size = 1024
                else:
                    net_num_stages = 4 if not num_stages else num_stages
                    net_lin_size = 1024 if not lin_size else lin_size
                self.config.cnet_train_epochs = 6 if not eps else eps #6
                self.config.lr = 3e-4 if not lr else lr # 1e-4

            self.config.pretrain_epochs = 5
            self.config.pretrain_lr = 1e-5
        else:
            if self.config.pretrain_AMASS:
                net_num_stages = 4 if not num_stages else num_stages
                net_lin_size = 1024 if not lin_size else lin_size
                if self.config.loss_pose_scaling:
                    self.config.cnet_train_epochs = 5 if not eps else eps
                    self.config.lr = 5e-4 if not lr else lr
                else:
                    self.config.cnet_train_epochs = 7 if not eps else eps
                    self.config.lr = 1e-4 if not lr else lr
            else:
                if self.config.cotrain:
                    net_num_stages = 4
                    net_lin_size = 1024
                else:
                    net_num_stages = 4 if not num_stages else num_stages
                    net_lin_size = 1024 if not lin_size else lin_size

                if self.config.only_hard_samples:
                    self.config.cnet_train_epochs = 6 if not eps else eps
                    self.config.lr = 1e-4 if not lr else lr 
                else:
                    self.config.cnet_train_epochs = 6 if not eps else eps
                    self.config.lr = 5e-5 if not lr else lr 

            # AMASS pretraining params
            self.config.pretrain_epochs = 5
            self.config.pretrain_lr = 2e-5

        self.continue_train = True if self.config.continue_train_CNet else False
        self.cnet = BaselineModel(linear_size=net_lin_size, num_stages=net_num_stages, p_dropout=0.5, 
                                num_in_kpts=len(self.in_kpts), num_out_kpts=len(self.target_kpts)).to(self.device)
        
        # Training
        self.config.train_split = 1.0
        self.config.err_scale = 1000

        self.use_valset = False if self.config.train_split == 1.0 else True

        # self.config.weight_decay = 1e-3
        self.config.batch_size = 4096

        self.config.ep_print_freq = 5

    def _loss(self, backbone_pred, cnet_out, target_xyz_17):
        '''
        Loss function
        '''
        # scale cnet targets to match backbone scale
        if self.config.loss_pose_scaling:
            scale_target = torch.sqrt((target_xyz_17**2).sum(dim=(1,2), keepdim=True))
            scale_backbone = torch.sqrt((backbone_pred**2).sum(dim=(1,2), keepdim=True))
            # target_xyz_17 /= (scale_target / scale_backbone)
            backbone_pred *= (scale_target / scale_backbone)

        if self.pred_errs: 
            cnet_target = backbone_pred - target_xyz_17  # predict errors
        else:
            cnet_target = target_xyz_17.clone() # predict kpts

        cnet_target = torch.flatten(cnet_target[:, self.target_kpts, :], start_dim=1)
        loss = self.criterion(cnet_out, cnet_target*self.config.err_scale).mean(dim=1)
        if self.config.sample_weighting:
            sample_ws = torch.sigmoid((cnet_target).norm(2, dim=1, keepdim=True).mean(dim=1))   # cnet_target.norm(2, dim=1, keepdim=True).mean(dim=1)
            loss *= sample_ws
        return loss.mean()

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
                diff = backbone_pred[:, self.target_kpts, dim] - pred_errs[..., dim]
                # corr_pred[:, self.target_kpts, dim] = pred_errs[:, self.target_kpts, dim]
                corr_pred[:, self.target_kpts, dim] -= self.config.corr_step_size*diff
        return corr_pred

    def __call__(self, cnet_in):
        for i in range(self.config.corr_steps):
            corr_pred = self._corr(cnet_in)
            cnet_in = corr_pred
        return corr_pred

    def _load_data_files(self, load_AMASS=False):
        '''
        Fetch and load the training files
        '''
        self.config.train_data_ids = []
        data_train, data_val = [], []
        if load_AMASS:
            self.use_valset = True
            amass_kpt_data = np.load(self.config.amass_kpts_path)['pose3d']    
            data = torch.from_numpy(amass_kpt_data).float()

            # add noise to kpts, to emulate backbone prediction errs
            noise_mean = 0.0
            distal_noise_std = 0.05
            body_noise_std = 0.03

            noisy_preds = data[0].reshape(data.shape[1], -1, 3).clone().detach()
            body_noise = torch.empty_like(noisy_preds).normal_(mean=noise_mean, std=body_noise_std)
            body_noise[:, :, 2] *= 2
            distal_noise = torch.empty_like(noisy_preds[:,self.config.distal_kpts]).normal_(mean=noise_mean, 
                                                                                            std=distal_noise_std)
            distal_noise[:, :, 2] *= 2

            noisy_preds[:, [kpt for kpt in range(1,17) if kpt not in self.config.distal_kpts]] += body_noise[:,[kpt for kpt in range(1,17) if kpt not in self.config.distal_kpts]]
            noisy_preds[:,self.config.distal_kpts] += distal_noise
            
            # Update "backbone" predictions with the noisy preds
            data[0] = noisy_preds.reshape(data.shape[1], data.shape[2])

            # shuffle & slice data
            idx = torch.randperm(data.shape[1])
            data = data[:, idx, :]

            TRAINSPLIT = 0.95
            len_train = int(len(data[0]) * TRAINSPLIT)
            data_t, data_v = data[:, :len_train, :], data[:, len_train:, :]

            # Check scale is correct (all coords should be < 10)
            for data in [data_t, data_v]:
                wrong_scale_idxs = (torch.where(data[:2].abs() > 10, 1, 0)).nonzero()
                if wrong_scale_idxs.sum() != 0:
                    idx = wrong_scale_idxs[...,:-1].unique()
                    data[idx[0], idx[1]] /= 1e3
            
            data_train = data_t
            data_val = data_v
        else:
            self.use_valset = False if self.config.train_split == 1.0 else True
            for i, (trainset_path, backbone_scale, data_lim) in enumerate(zip(self.config.cnet_trainset_paths,
                                                                    self.config.cnet_trainset_scales,
                                                                    self.config.train_datalims)):
                if self.R:
                    trainset_path = trainset_path.replace('cnet', 'rcnet')
                data = torch.from_numpy(np.load(trainset_path)).float()
                if ('MPii' in trainset_path) and ('mmlab' in trainset_path):
                    # samples with all 0 labels need to be removed
                    idx = np.where(data[1,:,:].sum(axis=1) != 0)
                    data = data[:, idx, :]
                    data = data.squeeze()
                if data_lim is not None:
                    # get random subset of data
                    self.config.train_data_ids.append(np.random.choice(data.shape[1], 
                                                                        min(data_lim, data.shape[1]), 
                                                                        replace=False))
                    data = data[:, self.config.train_data_ids[-1], :]
                # scale according to the backbone
                data[:2] *= backbone_scale

                # Check scale is correct (all coords should be < 10)
                wrong_scale_idxs = (torch.where(data[:2].abs() > 10, 1, 0)).nonzero()
                if wrong_scale_idxs.sum() != 0:
                    idx = wrong_scale_idxs[...,:-1].unique()
                    data[idx[0], idx[1]] /= 1e3

                # only keep back backbone pred samples if specified
                if self.config.only_hard_samples:
                    # get mse error
                    mse_err = (((data[0] - data[1])*1000)**2).mean(1)
                    # keep where error is above thresh
                    keep_idxs = (mse_err > self.config.hard_sample_thresh).nonzero().squeeze()
                    data = data[:, keep_idxs]
                
                # shuffle & slice data
                idx = torch.randperm(data.shape[1])
                data = data[:, idx, :]

                len_train = int(len(data[0]) * self.config.train_split)
                data_t, data_v = data[:, :len_train, :], data[:, len_train:, :]

                data_train.append(data_t[:3])   # TEMP: don't include image paths, if they are present
                data_val.append(data_v[:3])

            data_train = torch.cat(data_train, dim=1)
            data_val = torch.cat(data_val, dim=1) 

            if self.use_valset == False:
                # single dummy sample
                data_val = torch.ones_like(data_train[:, -1:])*-99

        return data_train, data_val

    def train(self, pretrain_AMASS=False, continue_train=False):
        data_train, data_val = self._load_data_files(pretrain_AMASS)

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

        lr = self.config.lr if not pretrain_AMASS else self.config.pretrain_lr
        self.optimizer = torch.optim.Adam(self.cnet.parameters(), lr=lr)
        self.criterion = nn.MSELoss(reduction='none')

        ckpt_name = self.config.ckpt_name if not pretrain_AMASS else 'pretrain' + self.config.ckpt_name

        # Train
        print('\n--- Training: {} ---'.format('R-CNet' if self.R else 'CNet'))
        if continue_train:
            print(" WARNING: CONTINUING TRAINING FROM PREVIOUS CHECKPOINT")
            if self.config.pretrain_AMASS:
                self.load_cnets(load_from=self.config.cnet_ckpt_path+'pretrain'+self.config.ckpt_name)
            else:    
                self.load_cnets()
        eps = self.config.cnet_train_epochs if not pretrain_AMASS else self.config.pretrain_epochs
        best_val_loss = 1e10
        best_val_ep = 0
        for ep in tqdm(range(eps), dynamic_ncols=True):
            cnet_train_losses = []
            self.cnet.train()
            for batch_idx, data in enumerate(gt_trainloader):
                backbone_pred = data[:, 0, :].to(self.device)
                target_xyz_17 = data[:, 1, :].to(self.device)
                self.optimizer.zero_grad()
                inp = torch.flatten(backbone_pred[:,self.in_kpts], start_dim=1)
                out = self.cnet(inp)
                loss = self._loss(backbone_pred, out, target_xyz_17)
                loss.backward()
                self.optimizer.step()
                cnet_train_losses.append(loss.item())

                # print every 500 batches
                if (batch_idx % 500 == 0) and (batch_idx != 0):
                    print(" EP {:3d} | Batch {:5d}/{:5d} | Avg Loss: {:12.5f}".format(ep, batch_idx, 
                                                                          len(gt_trainloader), 
                                                                          np.mean(cnet_train_losses)))
            mean_train_loss = np.mean(cnet_train_losses)
            # Val
            if self.use_valset:
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
                mean_val_loss = np.mean(cnet_val_losses)
            else:
                mean_val_loss = best_val_loss - 1
            # print on some epochs
            print_ep = ep % self.config.ep_print_freq == 0
            out_str = f"EP {ep:3d}:    t_loss: {mean_train_loss:12.5f} "
            if self.use_valset:
                out_str += f"   v_loss: {mean_val_loss:12.5f}"

            if print_ep:
                print(out_str)#, end='')
            
            if mean_val_loss < best_val_loss:
                if not print_ep:
                    print(out_str, end='')
                if self.use_valset: 
                    print("    ---> best val loss so far, ", end='')
                self.save(self.config.cnet_ckpt_path + ckpt_name)
                best_val_loss = mean_val_loss
                best_val_ep = ep

        out_str = f"EP {ep:3d}:    t_loss: {mean_train_loss:12.5f} "
        if self.use_valset:
            out_str += f"   v_loss: {mean_val_loss:12.5f}"
        print(out_str)
        print("|| Best val loss: {:12.5f} at ep: {:3d} ||".format(best_val_loss, best_val_ep))
        return
    
    def write_train_preds(self,):
        '''
        Get preds on training data and saves them to npy file, for
        training RCNet
        '''
        self.load_cnets()
        self.cnet.eval()
        # Save preds for each trainset
        for i, trainset_path in enumerate(self.config.cnet_trainset_paths):
            data_train = torch.from_numpy(np.load(trainset_path)).float()
             
            # # Get same subset used for training
            # if self.config.train_datalims[i] is not None:
            #     data_train = data_train[:, self.config.train_data_ids[i], :]

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

    def load_cnets(self, load_from=None, print_str=True):
        '''
        Load the best validation checkpoints
        '''
        load_path = self.config.cnet_ckpt_path + self.config.ckpt_name if load_from is None else load_from
        if print_str: 
            print("Loading {} from: {}".format('R-CNet' if self.R else 'CNet', load_path))
        all_net_ckpt_dict = torch.load(load_path)
        self.cnet.load_state_dict(all_net_ckpt_dict) 
    
    def save(self, path=None):
        if path == None:
            path = self.config.cnet_ckpt_path + self.config.ckpt_name
        print('saving {} to: {}'.format('R-CNet' if self.R else 'CNet', path))
        torch.save(self.cnet.state_dict(), path)
    
    def eval(self,):
        self.cnet.eval()

    