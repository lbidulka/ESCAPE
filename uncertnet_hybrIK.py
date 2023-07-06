"""uncertnet script for HybrIK"""
import argparse
import os
import sys
from pathlib import Path
from types import SimpleNamespace
import copy

import numpy as np
import torch

from utils import errors

# path_root = Path(__file__)#.parents[3]
# sys.path.append(str(path_root))
# sys.path.append(str(path_root)+ 'backbones/HybrIK/hybrik/')

# print("\n", os.getcwd(), "\n", sys.path, "\n")

from hybrik.models import builder
from hybrik.utils.config import update_config
from torchvision import transforms as T
from tqdm import tqdm
det_transform = T.Compose([T.ToTensor()])

import pickle as pk
from hybrik.datasets import HP3D, PW3D, H36mSMPL
from hybrik.utils.transforms import get_func_heatmap_to_coord

import uncertnet.distal_cnet as uncertnet_models
from utils import eval

from uncertnet.cnet_all import adapt_net

def get_config():
    config = SimpleNamespace()
    config.root = 'uncertnet_poserefiner/backbones/HybrIK/'
    os.chdir(config.root)

    # Main Settings
    config.use_cnet = True
    config.use_FF = False
    config.corr_steps = 1

    config.test_adapt = False
    config.test_adapt_lr = 1e-3
    config.adapt_steps = 1

    config.train_datalim = None # None

    # Tasks
    # config.tasks = ['make_trainset', 'make_testset', 'train', 'test']
    config.tasks = ['train', 'test'] # 'make_trainset' 'make_testset' 'train', 'test'
    # config.tasks = ['make_trainset', 'train', 'test']
    # config.tasks = ['make_testset', 'test']
    # config.tasks = ['make_trainset']
    # config.tasks = ['test']

    # Data
    config.trainset = 'HP3D' # 'HP3D', 'PW3D',
    config.testset = 'PW3D' # 'HP3D', 'PW3D',

    # HybrIK config
    config.hybrIK_version = 'hrw48_wo_3dpw' # 'res34_cam', 'hrw48_wo_3dpw'

    if config.hybrIK_version == 'res34_cam':
        config.cfg = 'configs/256x192_adam_lr1e-3-res34_smpl_3d_cam_2x_mix.yaml'
        config.ckpt = 'pretrained_w_cam.pth'
    if config.hybrIK_version == 'hrw48_wo_3dpw':
        config.cfg = 'configs/256x192_adam_lr1e-3-hrw48_cam_2x_wo_pw3d.yaml'    # w/o 3DPW
        config.ckpt = 'hybrik_hrnet48_wo3dpw.pth' 

    # cnet dataset
    config.cnet_ckpt_path = '../../ckpts/hybrIK/'
    config.cnet_dataset_path = '/media/ExtHDD/luke_data/adapt_3d/' #3DPW

    config.cnet_trainset_path = '{}{}/{}_cnet_hybrik_train.npy'.format(config.cnet_dataset_path, 
                                                                config.trainset,
                                                                config.hybrIK_version,)
    config.cnet_testset_path = '{}{}/{}_cnet_hybrik_test.npy'.format(config.cnet_dataset_path, 
                                                                config.testset,
                                                                config.hybrIK_version,)
    
    # CUDA
    config.device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    print_useful_configs(config)
    return config

def print_useful_configs(config):
    print('\n ----- CONFIG: -----')
    print(' -------------------')
    print('hybrIK_version: {}'.format(config.hybrIK_version))
    print('Tasks: {}'.format(config.tasks))
    print(' --- CNet: ---')
    print('Use CNet: {}'.format(config.use_cnet))
    print('Use FF: {}'.format(config.use_FF))
    print('Corr Steps: {}'.format(config.corr_steps))
    print('Test Adapt: {}'.format(config.test_adapt))
    print('Test Adapt LR: {}'.format(config.test_adapt_lr))
    print('Adapt Steps: {}'.format(config.adapt_steps)) 
    print(' --- Data: ---')
    print('Trainset: {}'.format(config.trainset))
    print('Testset: {}'.format(config.testset))
    print('Trainset path: {}'.format(config.cnet_trainset_path))
    print('Testset path: {}'.format(config.cnet_testset_path))
    print(' ----------------- \n') 
    return

config = get_config()
cfg = update_config(config.cfg)

def load_pretrained_hybrik(ckpt=config.ckpt):
    hybrik_model = builder.build_sppe(cfg.MODEL)
    
    print(f'\nLoading HybrIK model from {ckpt}...\n')
    save_dict = torch.load(ckpt, map_location='cpu')
    if type(save_dict) == dict:
        model_dict = save_dict['model']
        hybrik_model.load_state_dict(model_dict)
    else:
        hybrik_model.load_state_dict(save_dict, strict=False)

    return hybrik_model

def create_cnet_dataset(m, cfg, gt_dataset, task='train'):
    # Data/Setup
    gt_loader = torch.utils.data.DataLoader(gt_dataset, batch_size=64, shuffle=False, 
                                            num_workers=16, drop_last=False, pin_memory=True)
    m.eval()
    m = m.to(config.device)

    opt = SimpleNamespace()
    opt.device = config.device
    opt.flip_test = True

    hm_shape = cfg.MODEL.get('HEATMAP_SIZE')
    hm_shape = (hm_shape[1], hm_shape[0])

    if isinstance(gt_dataset, HP3D):
        target_key = 'target_xyz'
    elif isinstance(gt_dataset, PW3D):
        target_key = 'target_xyz_17'

    backbone_preds = []
    target_xyz_17s = []
    img_idss = []
    for i, data in enumerate(tqdm(gt_loader, dynamic_ncols=True)):
        
        (inps, labels, img_ids, bboxes) = data

        if isinstance(inps, list):
            inps = [inp.cuda(opt.gpu) for inp in inps]
        else:
            inps=inps.to(config.device)

        for k, _ in labels.items():
            try:
                labels[k] = labels[k].to(config.device)
            except AttributeError:
                assert k == 'type'

        m_output = m(inps, flip_test=opt.flip_test, bboxes=bboxes.to(config.device), img_center=labels['img_center'])
        backbone_pred = m_output.pred_xyz_jts_17

        backbone_preds.append(backbone_pred.detach().cpu().numpy())
        target_xyz_17s.append(labels[target_key].detach().cpu().numpy())
        img_idss.append(img_ids.detach().cpu().numpy())

        # if i > 250: break   # DEBUG

    # dataset_outpath = '{}{}_cnet_hybrik_{}.npy'.format(config.cnet_dataset_path, config.hybrIK_version, task)
    if task == 'train':
        dataset_outpath = config.cnet_trainset_path
    elif task == 'test':
        dataset_outpath = config.cnet_testset_path

    if isinstance(gt_dataset, HP3D):
    #     all_joint_names = 
    #   {'spine3', 'spine4', 'spine2', 'spine', 
    #    'pelvis', ...     %5       
    #    'neck', 'head', 'head_top', 
    #    'left_clavicle', 'left_shoulder', 'left_elbow', ... %11 'left_wrist', 'left_hand',  
    #    'right_clavicle', 'right_shoulder', 'right_elbow', 'right_wrist', ... %17 'right_hand', 
    #    'left_hip', 'left_knee', 'left_ankle', 'left_foot', 'left_toe', ...        %23   
    #    'right_hip' , 'right_knee', 'right_ankle', 'right_foot', 'right_toe'}; 
        EVAL_JOINTS_17 = [
            4,
            18, 19, 20,
            23, 24, 25,
            3, 5, # 'spine' == spine_extra, 'neck' == throat (not quite 'neck_extra' as desired)
            6, 7, # 
            9, 10, 11,
            14, 15, 16,
        ]
        target_xyz_17s = [np.take(t.reshape(-1, 28, 3), EVAL_JOINTS_17, axis=1) for t in target_xyz_17s]
        target_xyz_17s = [t.reshape(-1, 51) for t in target_xyz_17s]

    np.save(dataset_outpath, np.array([np.concatenate(backbone_preds, axis=0), 
                                       np.concatenate(target_xyz_17s, axis=0),
                                       np.repeat(np.concatenate(img_idss, axis=0).reshape(-1,1), backbone_pred.shape[1], axis=1)]))
    return

def eval_gt(m, cnet, cfg, gt_eval_dataset, heatmap_to_coord, test_vertice=False, test_cnet=False, use_data_file=False):
    if config.test_adapt:
        batch_size = 1
    else:
        batch_size = 128
    gt_eval_dataset_for_scoring = gt_eval_dataset
    # Data/Setup
    if use_data_file:
        test_file = config.cnet_testset_path
        test_data = torch.from_numpy(np.load(test_file)).float().permute(1,0,2)
        gt_eval_dataset = torch.utils.data.TensorDataset(test_data)
    gt_eval_loader = torch.utils.data.DataLoader(gt_eval_dataset, batch_size=batch_size, shuffle=False, 
                                                 num_workers=16, drop_last=False, pin_memory=True)
    kpt_pred = {}
    kpt_all_pred = {}
    m.eval()
    cnet.cnet.eval()

    hm_shape = cfg.MODEL.get('HEATMAP_SIZE')
    hm_shape = (hm_shape[1], hm_shape[0])
    pve_list = []

    errs = []
    errs_s = []

    opt = SimpleNamespace()
    opt.device = config.device
    opt.flip_test = True

    def set_bn_eval(module):
        ''' Batch Norm layer freezing '''
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()

    for data in tqdm(gt_eval_loader, dynamic_ncols=True):
        if use_data_file:
            labels = {}
            output = SimpleNamespace()
            output.pred_xyz_jts_17 = data[0][:,0].to(config.device)
            labels['target_xyz_17'] = data[0][:,1].to(config.device)
            img_ids = data[0][:,2,:1].to(config.device)
        else:
            (inps, labels, img_ids, bboxes) = data
            if isinstance(inps, list):
                inps = [inp.cuda(opt.gpu) for inp in inps]
            else:
                # inps = inps.cuda(opt.gpu)
                inps=inps.to(config.device)
            for k, _ in labels.items():
                try:
                    labels[k] = labels[k].to(config.device)
                except AttributeError:
                    assert k == 'type'

            with torch.no_grad():
                output = m(inps, flip_test=opt.flip_test, bboxes=bboxes.to(config.device), img_center=labels['img_center'])
        
        if test_cnet:
            backbone_pred = output.pred_xyz_jts_17
            backbone_pred = backbone_pred.reshape(-1, 17, 3)

            if config.use_FF:
                poses_2d = labels['target_xyz_17'].reshape(labels['target_xyz_17'].shape[0], -1, 3)[:,:,:2]    # TEMP: using labels for 2D
                cnet_in = (poses_2d, backbone_pred)
            elif config.test_adapt:
                for i in range(config.adapt_steps):
                    # Setup Optimizer
                    cnet.load_cnets(print_str=False)   # reset to trained cnet
                    cnet.cnet.train()
                    cnet.cnet.apply(set_bn_eval)
                    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, cnet.cnet.parameters()),
                                                  lr=config.test_adapt_lr)#, weight_decay=1e-3)
                    # Get 2d reproj loss & take grad step
                    corrected_pred = cnet(backbone_pred)
                    poses_2d = labels['target_xyz_17'].reshape(labels['target_xyz_17'].shape[0], -1, 3)[:,:,:2]    # TEMP: using labels for 2D
                    loss = errors.loss_weighted_rep_no_scale(poses_2d, corrected_pred, sum_kpts=True).sum()
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    # get corrected pred, with adjusted cnet
                    cnet_in = backbone_pred
            else:
                cnet_in = backbone_pred

            if config.use_cnet:
                with torch.no_grad():
                    for i in range(config.corr_steps):
                        corrected_pred = cnet(cnet_in)
                        cnet_in = corrected_pred
                output.pred_xyz_jts_17 = corrected_pred.reshape(labels['target_xyz_17'].shape[0], -1)
            else:
                output.pred_xyz_jts_17 = backbone_pred.reshape(labels['target_xyz_17'].shape[0], -1)

        pred_xyz_jts_17 = output.pred_xyz_jts_17.reshape(labels['target_xyz_17'].shape[0], 17, 3)
        pred_xyz_jts_17 = pred_xyz_jts_17.cpu().data.numpy()
        assert pred_xyz_jts_17.ndim in [2, 3]
        pred_xyz_jts_17 = pred_xyz_jts_17.reshape(pred_xyz_jts_17.shape[0], 17, 3)
        for i in range(pred_xyz_jts_17.shape[0]):
            kpt_pred[int(img_ids[i])] = {
                'xyz_17': pred_xyz_jts_17[i],
            }
        kpt_all_pred.update(kpt_pred)

    tot_err_17 = gt_eval_dataset_for_scoring.evaluate_xyz_17(kpt_all_pred, os.path.join('exp', 'test_3d_kpt.json'))
    return tot_err_17

def make_trainset(hybrik, cfg, gt_train_dataset_3dpw):
    with torch.no_grad():
        print('##### Creating CNET {} Trainset #####'.format(config.trainset))
        create_cnet_dataset(hybrik, cfg, gt_train_dataset_3dpw, task='train')

def make_testset(hybrik, cfg, gt_test_dataset_3dpw):
    with torch.no_grad():
        print('##### Creating CNET {} Testset #####'.format(config.testset))
        create_cnet_dataset(hybrik, cfg, gt_test_dataset_3dpw, task='test')

def test(hybrik, cnet, cfg, gt_test_dataset_3dpw):
    cnet.load_cnets()
    hybrik = hybrik.to(config.device)
    heatmap_to_coord = get_func_heatmap_to_coord(cfg) 

    print('\n##### 3DPW TESTSET ERRS #####\n')
    tot_corr_PA_MPJPE = eval_gt(hybrik, cnet, cfg, gt_test_dataset_3dpw, heatmap_to_coord, test_vertice=False, test_cnet=True, use_data_file=True)
    print('\n--- Vanilla: --- ')
    with torch.no_grad():
        gt_tot_err = eval_gt(hybrik, cnet, cfg, gt_test_dataset_3dpw, heatmap_to_coord, test_vertice=False, test_cnet=False, use_data_file=True)
    # if config.hybrIK_version == 'res34_cam':
    #     print('XYZ_14 PA-MPJPE: 45.917672 | MPJPE: 74.113751, x: 27.145215, y: 28.64, z: 51.785723')  # w/ 3DPW
    # if config.hybrIK_version == 'hrw48_wo_3dpw':
    #     print('XYZ_14 PA-MPJPE: 49.346562 | MPJPE: 88.707589, x: 29.233308, y: 30.03, z: 66.807150')  # wo/ 3DPW

def get_dataset(cfg):
    # Datasets for HybrIK
    if config.trainset == 'PW3D':
        trainset = PW3D(
            cfg=cfg,
            ann_file='3DPW_train_new_fresh.json',
            train=False,
            root='/media/ExtHDD/Mohsen_data/3DPW')
    # elif config.trainset == 'HP3D':
    #     trainset = HP3D(
    #         cfg=cfg,
    #         ann_file='annotation_mpi_inf_3dhp_train_v2.json',
    #         train=False,
    #         root='/media/ExtHDD/luke_data/3DHP')
    elif config.trainset == 'HP3D':
        trainset = HP3D(
            cfg=cfg,
            ann_file='train_v2',   # dumb adjustment...
            train=True,
            root='/media/ExtHDD/luke_data/HP3D')
        
    if config.testset == 'PW3D':
        testset = PW3D(
            cfg=cfg,
            ann_file='3DPW_test_new_fresh.json',
            train=False,
            root='/media/ExtHDD/Mohsen_data/3DPW')
    if config.testset == 'HP3D':
        raise NotImplementedError    
    
    return trainset, testset

def main_worker(cfg, hybrIK_model): 
    print(' USING HYBRIK VER: {}'.format(config.hybrIK_version))
    
    hybrik = hybrIK_model.to('cpu')
    cnet = adapt_net(config)

    hybrik_trainset, hybrik_testset = get_dataset(cfg)

    if 'make_trainset' in config.tasks:
        make_trainset(hybrik, cfg, hybrik_trainset)
    if 'make_testset' in config.tasks: 
        make_testset(hybrik, cfg, hybrik_testset)    
    if 'train' in config.tasks:
        cnet.train()
    if 'test' in config.tasks:
        test(hybrik, cnet, cfg, hybrik_testset)

if __name__ == "__main__":    
    hybrik = load_pretrained_hybrik()
    main_worker(cfg, hybrIK_model=hybrik)

