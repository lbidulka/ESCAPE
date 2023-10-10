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
                    net_num_stages = 2
                    net_lin_size = 512
                else:
                    net_num_stages = 2 if not num_stages else num_stages
                    net_lin_size = 512 if not lin_size else lin_size
                self.config.cnet_train_epochs = 6 if not eps else eps #6
                self.config.lr = 5e-4 if not lr else lr # 1e-4

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
                    net_num_stages = 2
                    net_lin_size = 512
                else:
                    # net_num_stages = 4 if not num_stages else num_stages
                    # net_lin_size = 1024 if not lin_size else lin_size
                    net_num_stages = 2 if not num_stages else num_stages
                    net_lin_size = 512 if not lin_size else lin_size

                if self.config.only_hard_samples:
                    self.config.cnet_train_epochs = 6 if not eps else eps
                    self.config.lr = 1e-4 if not lr else lr 
                else:
                    # self.config.cnet_train_epochs = 6 if not eps else eps
                    # self.config.lr = 5e-5 if not lr else lr 
                    self.config.cnet_train_epochs = 6 if not eps else eps
                    self.config.lr = 5e-4 if not lr else lr 

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
    
    def __call__(self, cnet_in, ret_corr_idxs=False):
        out_pred = cnet_in.clone().detach()
        if self.config.use_cnet_energy and self.R == False:
            E_in = self._energy(cnet_in)
            if self.config.energy_lower_thresh:
                # dont correct samples with energy above threshold
                dont_corr_idxs = E_in > self.config.energy_thresh
            else:
                dont_corr_idxs = torch.zeros(cnet_in.shape[0]).bool().to(self.device)

            # init step sizes
            step_sizes = torch.ones_like(cnet_in[:,:1,0]) * self.config.corr_step_size
            
            if self.config.energy_scaled_corr:
                # scale step sizes based on energy
                # med_thresh = 600
                # hard_thresh = 400

                # med_idxs = ((E_in < med_thresh) & (E_in > hard_thresh)).nonzero()
                # hard_idxs = (E_in < hard_thresh).nonzero()

                # step_sizes[med_idxs] *= 2.0
                # step_sizes[hard_idxs] *= 3.0

                step_sizes *= torch.clamp(1.0 - (E_in - 900)/(200),
                                          min=0.0, max=10.0).reshape(-1,1)
        else:
            dont_corr_idxs = torch.zeros(cnet_in.shape[0]).bool().to(self.device)
            step_sizes = torch.ones_like(cnet_in[:,:1,0]) * self.config.corr_step_size
        corr_idxs = ~dont_corr_idxs.cpu()
        step_sizes = step_sizes[corr_idxs]

        # during TTT we might only have one sample, and so we can return if we dont need to do anything
        if corr_idxs.sum() == 0:
            if ret_corr_idxs:
                return cnet_in, corr_idxs
            else:
                return cnet_in

        # rotate poses to zero orientation
        if self.config.zero_orientation:
            cnet_in, rot, i_rot = self._zero_orientation(cnet_in[corr_idxs])

        # correct as required
        for i in range(self.config.corr_steps):
            corr_pred = self._corr(cnet_in, step_sizes=step_sizes)
            cnet_in = corr_pred
        
        # restore corr poses to original orientation
        if self.config.zero_orientation:
            corr_pred = torch.bmm(i_rot.transpose(1,2), 
                                    corr_pred.reshape(-1, corr_pred.shape[1], 3).transpose(1,2)).transpose(1,2)

        # place samples back in original tensor at original indices
        out_pred[corr_idxs] = corr_pred

        if ret_corr_idxs:
            return out_pred, corr_idxs
        else:
            return out_pred

    def _corr(self, in_pose, step_sizes):
        '''
        Correct the input poses
        '''
        inp = in_pose[:,self.in_kpts].flatten(1)
        # inp = inp.reshape(inp.shape[0], -1) #.flatten(1)
        pred_errs = self.cnet(inp) / self.config.err_scale
        pred_errs = pred_errs.reshape(-1, len(self.target_kpts), 3) 
        corr_pred = in_pose.detach().clone()
        # do correction
        for dim in self.corr_dims:
            if self.pred_errs:
                diff = pred_errs[..., dim]
            else:
                in_pose[:, self.target_kpts, dim] - pred_errs[..., dim]
            corr_pred[:, self.target_kpts, dim] -= step_sizes*diff
        return corr_pred

    def _energy(self, data):
        '''
        Compute the energy score of the regression data
        '''
        E = torch.logsumexp(data.flatten(1) * 1000, dim=-1)
        return E
    
    def _zero_orientation(self, in_poses):
        '''
        Zero the orientation of the poses, according to the neck->pelvis and Lhip->Rhip vectors.
        The cross prod. of the two should face (0,0,1)
        '''
        poses = in_poses.detach().clone()
        # get neck->pelvis and Lhip->Rhip vectors
        y_vec = poses[:, 8] - poses[:, 0]
        x_vec = poses[:, 1] - poses[:, 4]
        x_vec /= torch.norm(x_vec, dim=-1, keepdim=True)
        y_vec /= torch.norm(y_vec, dim=-1, keepdim=True)

        # get cross prod of the two
        z_vec = torch.cross(x_vec, y_vec,)
        z_vec /= torch.norm(z_vec, dim=-1, keepdim=True)
        
        # Rot around z-axis by z_ang
        rot = torch.ones((len(z_vec), 3, 3)).to(self.device)
        rot[..., 0] = x_vec
        rot[..., 1] = y_vec
        rot[..., 2] = z_vec
        i_rot = torch.inverse(rot)

        rot_poses = torch.bmm(rot.transpose(1,2), poses.transpose(1,2)).transpose(1,2)
        return rot_poses, rot, i_rot

    def _create_synth_amass(self, data,):
        '''
        add noise to kpts, to emulate backbone prediction errs
        '''
        noise_mean = 0.0
        distal_noise_std = 0.05
        body_noise_std = 0.03

        noisy_preds = data[0].reshape(data.shape[1], -1, 3).clone().detach()
        body_noise = torch.empty_like(noisy_preds).normal_(mean=noise_mean, std=body_noise_std)
        body_noise[:, :, 2] *= 2
        distal_noise = torch.empty_like(noisy_preds[:,self.config.distal_kpts]).normal_(mean=noise_mean, std=distal_noise_std)
        distal_noise[:, :, 2] *= 2

        noisy_preds[:, [kpt for kpt in range(1,17) if kpt not in self.config.distal_kpts]] += body_noise[:,[kpt for kpt in range(1,17) if kpt not in self.config.distal_kpts]]
        noisy_preds[:,self.config.distal_kpts] += distal_noise

        # Update "backbone" predictions with the noisy preds
        data[0] = noisy_preds.reshape(data.shape[1], data.shape[2])
        return data

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
            
            data = self._create_synth_amass(data)

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
                # TEMP DEBUG ----------------------------------------------
                if ('RICH' in trainset_path):
                    # add img ids to data
                    ids = torch.arange(len(data[1])).reshape(-1,1)
                    ids = ids.repeat(51, 1)
                    ids = ids.reshape(1, -1, 51)
                    data = torch.cat([data, ids], axis=0)
                # ----------------------------------------------------------
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

                # only keep bad backbone pred samples if specified
                if self.config.only_hard_samples:
                    mse_err = (((data[0] - data[1])*1000)**2).mean(1)
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

    def train(self, pretrain_AMASS=False, continue_train=False):
        data_train, data_val = self._load_data_files(pretrain_AMASS)

        data_train = data_train.permute(1,0,2)
        data_val = data_val.permute(1,0,2)
        data_train = data_train.reshape(data_train.shape[0], 3, -1, 3) # batch, 2, kpts, xyz)
        data_val = data_val.reshape(data_val.shape[0], 3, -1, 3)

        train_transform = hflip_keypoints()
        gt_trainset = cnet_pose_dataset(data_train, transform=train_transform)
        # gt_trainset = cnet_pose_dataset(data_train,)
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

                # rotate poses to zero orientation
                if self.config.zero_orientation:
                    backbone_pred, rot, i_rot = self._zero_orientation(backbone_pred)

                # inp = torch.flatten(backbone_pred[:,self.in_kpts], start_dim=1)
                inp = backbone_pred[:,self.in_kpts].flatten(1)
                out = self.cnet(inp)

                # restore corr poses to original orientation
                if self.config.zero_orientation:
                    out = torch.bmm(i_rot.transpose(1,2), 
                                    out.reshape(-1, len(self.target_kpts), 3).transpose(1,2)).transpose(1,2)
                    out = out.flatten(1)
                    backbone_pred = torch.bmm(i_rot.transpose(1,2),
                                                backbone_pred.transpose(1,2)).transpose(1,2)

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

    