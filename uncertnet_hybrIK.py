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
    
    # CUDA
    config.device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    # Tasks
    config.tasks = ['train', 'test'] # 'make_trainset' 'make_testset' 'train', 'test'
    # config.tasks = ['make_testset', 'test']
    # config.tasks = ['test']

    # Main Settings
    config.use_FF = False
    config.corr_steps = 1

    config.test_adapt = False
    config.test_adapt_lr = 1e-3
    config.adapt_steps = 1

    # HybrIK config
    config.hybrIK_version = 'hrw48_wo_3dpw' # 'res34_cam', 'hrw48_wo_3dpw'

    if config.hybrIK_version == 'res34_cam':
        config.cfg = 'configs/256x192_adam_lr1e-3-res34_smpl_3d_cam_2x_mix.yaml'
        config.ckpt = 'pretrained_w_cam.pth'
    if config.hybrIK_version == 'hrw48_wo_3dpw':
        config.cfg = 'configs/256x192_adam_lr1e-3-hrw48_cam_2x_wo_pw3d.yaml'    # w/o 3DPW
        config.ckpt = 'hybrik_hrnet48_wo3dpw.pth'    

    
    # Data
    config.trainset = '3DPW'
    config.testset = '3DPW'
    
    # AMASS data
    # data_3d_AMASS=np.load('/media/ExtHDD/Mohsen_data/AMASS/processed_AMASS.npz')
    # config.data_amass_dir = '/media/ExtHDD/Mohsen_data/AMASS'
    

    # 3DPW data
    config.data_3DPW_dir = '/media/ExtHDD/Mohsen_data/3DPW'
    config.data_3DPW_test_annot = '{}/json/3DPW_test_new.json'.format(config.data_3DPW_dir)
    # config.data_3DPW_test_annot = '{}/3DPW_latest_test.json'.format(config.data_3DPW_dir)
    config.data_3DPW_train_annot = '{}/3DPW_latest_train.json'.format(config.data_3DPW_dir)
    config.data_3DPW_img_dir = '{}/imageFiles'.format(config.data_3DPW_dir)

    # cnet dataset
    config.cnet_ckpt_path = '../../ckpts/hybrIK/'
    config.cnet_dataset_path = '/media/ExtHDD/luke_data/adapt_3d/' #3DPW

    print_useful_configs(config)
    return config

def print_useful_configs(config):
    print('\n --- Config: ---')
    print('hybrIK_version: {}'.format(config.hybrIK_version))
    print('Tasks: {}'.format(config.tasks))
    print('Use FF: {}'.format(config.use_FF))
    print('Corr Steps: {}'.format(config.corr_steps))
    print('Test Adapt: {}'.format(config.test_adapt))
    print('Test Adapt LR: {}'.format(config.test_adapt_lr))
    print('Adapt Steps: {}'.format(config.adapt_steps))
    print(' ----------------- \n')  
    return

config = get_config()
parser = argparse.ArgumentParser(description='HybrIK Demo')

parser.add_argument('--gpu',
                    help='gpu',
                    default=0,
                    type=int)
parser.add_argument('--img-dir',
                    help='image folder',
                    default=config.data_3DPW_img_dir,
                    type=str)
parser.add_argument('--annot-dir',
                    help='image folder',
                    default=config.data_3DPW_train_annot,
                    type=str)
parser.add_argument('--out-dir',
                    help='output folder',
                    default='',
                    type=str)

parser.add_argument('--cfg',
                    help='experiment configure file name',
                    required=False,
                    type=str)
parser.add_argument('--checkpoint',
                    help='checkpoint file name',
                    required=False,
                    type=str)
parser.add_argument('--gpus',
                    help='gpus',
                    type=str)
parser.add_argument('--batch',
                    help='validation batch size',
                    type=int)
parser.add_argument('--flip-test',
                    default=False,
                    dest='flip_test',
                    help='flip test',
                    action='store_true')
parser.add_argument('--flip-shift',
                    default=False,
                    dest='flip_shift',
                    help='flip shift',
                    action='store_true')

opt = parser.parse_args()
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

def create_cnet_dataset(m, opt, cfg, gt_dataset, task='train'):
    # Data/Setup
    gt_loader = torch.utils.data.DataLoader(gt_dataset, batch_size=64, shuffle=False, num_workers=8, drop_last=False)
    m.eval()
    m = m.to(config.device)

    hm_shape = cfg.MODEL.get('HEATMAP_SIZE')
    hm_shape = (hm_shape[1], hm_shape[0])

    backbone_preds = []
    target_xyz_17s = []
    img_idss = []
    for(inps, labels, img_ids, bboxes) in tqdm(gt_loader, dynamic_ncols=True):
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
        target_xyz_17s.append(labels['target_xyz_17'].detach().cpu().numpy())
        if task == 'test':
            img_idss.append(img_ids.detach().cpu().numpy())

    dataset_outpath = '{}{}_cnet_hybrik_{}.npy'.format(config.cnet_dataset_path, config.hybrIK_version, task)
    np.save(dataset_outpath, np.array([np.concatenate(backbone_preds, axis=0), 
                                       np.concatenate(target_xyz_17s, axis=0),
                                       np.repeat(np.concatenate(img_idss, axis=0).reshape(-1,1), backbone_pred.shape[1], axis=1)]))
    return

def eval_gt(m, cnet, opt, cfg, gt_eval_dataset, heatmap_to_coord, test_vertice=False, test_cnet=False, use_data_file=False):
    if config.test_adapt:
        batch_size = 1
    else:
        batch_size = 128
    gt_eval_dataset_for_scoring = gt_eval_dataset
    # Data/Setup
    if use_data_file:
        test_file = '{}{}_cnet_hybrik_test.npy'.format(config.cnet_dataset_path, config.hybrIK_version,)
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

            with torch.no_grad():
                for i in range(config.corr_steps):
                    corrected_pred = cnet(cnet_in)
                    cnet_in = corrected_pred
            output.pred_xyz_jts_17 = corrected_pred.reshape(labels['target_xyz_17'].shape[0], -1)

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

def make_trainset(hybrik, opt, cfg, gt_train_dataset_3dpw):
    with torch.no_grad():
        print('##### Creating CNET 3DPW Trainset #####')
        create_cnet_dataset(hybrik, opt, cfg, gt_train_dataset_3dpw, task='train')

def make_testset(hybrik, opt, cfg, gt_test_dataset_3dpw):
    with torch.no_grad():
        print('##### Creating CNET 3DPW Testset #####')
        create_cnet_dataset(hybrik, opt, cfg, gt_test_dataset_3dpw, task='test')

def test(hybrik, cnet, opt, cfg, gt_test_dataset_3dpw):
    cnet.load_cnets()
    hybrik = hybrik.to(config.device)
    heatmap_to_coord = get_func_heatmap_to_coord(cfg) 

    print('\n##### 3DPW TESTSET ERRS #####\n')
    tot_corr_PA_MPJPE = eval_gt(hybrik, cnet, opt, cfg, gt_test_dataset_3dpw, heatmap_to_coord, test_vertice=False, test_cnet=True, use_data_file=True)
    print('\n--- Vanilla: --- ')
    # with torch.no_grad():
    #     gt_tot_err = eval_gt(hybrik, cnet, opt, cfg, gt_test_dataset_3dpw, heatmap_to_coord, test_vertice=False, test_cnet=False)
    if config.hybrIK_version == 'res34_cam':
        print('XYZ_14 PA-MPJPE: 45.917672 | MPJPE: 74.113751, x: 27.145215, y: 28.64, z: 51.785723')  # w/ 3DPW
    if config.hybrIK_version == 'hrw48_wo_3dpw':
        print('XYZ_14 PA-MPJPE: 49.346562 | MPJPE: 88.707589, x: 29.233308, y: 30.03, z: 66.807150')  # wo/ 3DPW

def main_worker(opt, cfg, hybrIK_model): 
    print(' USING HYBRIK VER: {}'.format(config.hybrIK_version))
    
    hybrik = hybrIK_model.to('cpu')
    config.train_datalim = None
    # config.train_datalim = 100
    cnet = adapt_net(config)

    # Datasets for HybrIK
    gt_train_dataset_3dpw = PW3D(
        cfg=cfg,
        ann_file='3DPW_train_new_fresh.json',
        train=False,
        root='/media/ExtHDD/Mohsen_data/3DPW')
    gt_test_dataset_3dpw = PW3D(
        cfg=cfg,
        # ann_file='3DPW_test_new.json',
        ann_file='3DPW_test_new_fresh.json',
        train=False,
        root='/media/ExtHDD/Mohsen_data/3DPW')

    if 'make_trainset' in config.tasks:
        make_trainset(hybrik, opt, cfg, gt_train_dataset_3dpw)
    if 'make_testset' in config.tasks: 
        make_testset(hybrik, opt, cfg, gt_test_dataset_3dpw)    
    if 'train' in config.tasks:
        cnet.train()
    if 'test' in config.tasks:
        test(hybrik, cnet, opt, cfg, gt_test_dataset_3dpw)

if __name__ == "__main__":    
    hybrik = load_pretrained_hybrik()
    main_worker(opt, cfg, hybrIK_model=hybrik)

