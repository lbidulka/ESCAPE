"""uncertnet script for HybrIK"""
import os
import sys
from pathlib import Path

# path_root = Path(__file__)#.parents[3]
# sys.path.append(str(path_root))
# sys.path.append(str(path_root)+ 'backbones/HybrIK/hybrik/')

# print("\n", os.getcwd(), "\n", sys.path, "\n")

import numpy as np
import torch
from torchvision import transforms as T
det_transform = T.Compose([T.ToTensor()])

from hybrik.models import builder
from hybrik.utils.config import update_config

from datasets.MPII import MPII
from datasets.PW3D import PW3D
from datasets.HP3D import HP3D
from cnet.multi_distal import multi_distal
from cnet.full_body import adapt_net
from core.cnet_data import create_cnet_dataset_w_HybrIK
from core.cnet_eval import eval_gt
from config import get_config


def load_pretrained_hybrik(config, hybrik_cfg,):
    ckpt = config.ckpt
    hybrik_model = builder.build_sppe(hybrik_cfg.MODEL)
    
    print(f'\nLoading HybrIK model from {ckpt}...\n')
    save_dict = torch.load(ckpt, map_location='cpu')
    if type(save_dict) == dict:
        model_dict = save_dict['model']
        hybrik_model.load_state_dict(model_dict)
    else:
        hybrik_model.load_state_dict(save_dict, strict=False)

    return hybrik_model

def make_trainsets(hybrik, gt_train_datasets_3dpw, config):
    with torch.no_grad():
        for gt_train_dataset_3dpw in gt_train_datasets_3dpw:
            print('##### Creating CNET {} Trainset #####'.format(config.trainset))
            create_cnet_dataset_w_HybrIK(hybrik, config, gt_train_dataset_3dpw, dataset=config.trainset, task='train',)

def make_testset(hybrik, gt_test_dataset_3dpw, config):
    with torch.no_grad():
        print('##### Creating CNET {} Testset #####'.format(config.testset))
        create_cnet_dataset_w_HybrIK(hybrik, config, gt_test_dataset_3dpw, dataset=config.trainset, task='test',)

def test(hybrik, cnet, R_cnet, gt_test_dataset_3dpw, config):
    cnet.load_cnets()
    if config.test_adapt: R_cnet.load_cnets()
    hybrik = hybrik.to(config.device)

    print('\n##### 3DPW TESTSET ERRS #####')
    if config.test_adapt:
        print('--- CNet w/TTT: --- ')
        TTT_corr_eval_summary = eval_gt(hybrik, cnet, R_cnet, config, gt_test_dataset_3dpw, 
                                test_cnet=True, test_adapt=True, use_data_file=True)
        print('XYZ_14 PA-MPJPE: {:2f} | MPJPE: {:2f}, x: {:2f}, y: {:.2f}, z: {:2f}'.format(TTT_corr_eval_summary['PA-MPJPE'], 
                                                                                            TTT_corr_eval_summary['MPJPE'], 
                                                                                            TTT_corr_eval_summary['x'], 
                                                                                            TTT_corr_eval_summary['y'], 
                                                                                            TTT_corr_eval_summary['z']))    
    print('--- CNet Only: --- ')
    cnet.load_cnets()
    if config.test_adapt: R_cnet.load_cnets()
    corr_eval_summary = eval_gt(hybrik, cnet, R_cnet, config, gt_test_dataset_3dpw, 
                                test_cnet=True, use_data_file=True)
    print('XYZ_14 PA-MPJPE: {:2f} | MPJPE: {:2f}, x: {:2f}, y: {:.2f}, z: {:2f}'.format(corr_eval_summary['PA-MPJPE'], 
                                                                                          corr_eval_summary['MPJPE'], 
                                                                                          corr_eval_summary['x'], 
                                                                                          corr_eval_summary['y'], 
                                                                                          corr_eval_summary['z']))
    print('--- Vanilla: --- ')
    with torch.no_grad():
        van_eval_summary = eval_gt(hybrik, cnet, R_cnet, config, gt_test_dataset_3dpw, 
                             test_cnet=False, use_data_file=True)
    print('XYZ_14 PA-MPJPE: {:2f} | MPJPE: {:2f}, x: {:2f}, y: {:.2f}, z: {:2f}'.format(van_eval_summary['PA-MPJPE'], 
                                                                                          van_eval_summary['MPJPE'], 
                                                                                          van_eval_summary['x'], 
                                                                                          van_eval_summary['y'], 
                                                                                          van_eval_summary['z']))
    print('--- Corr. - Van.: --- ')
    if config.test_adapt:
        print('CN w/TTT: XYZ_14 PA-MPJPE: {:2f} | MPJPE: {:2f}, x: {:2f}, y: {:.2f}, z: {:2f}'.format(TTT_corr_eval_summary['PA-MPJPE'] - van_eval_summary['PA-MPJPE'],
                                                                                                TTT_corr_eval_summary['MPJPE'] - van_eval_summary['MPJPE'],
                                                                                                TTT_corr_eval_summary['x'] - van_eval_summary['x'],
                                                                                                TTT_corr_eval_summary['y'] - van_eval_summary['y'],
                                                                                                TTT_corr_eval_summary['z'] - van_eval_summary['z'],))
    print('CN alone: XYZ_14 PA-MPJPE: {:2f} | MPJPE: {:2f}, x: {:2f}, y: {:.2f}, z: {:2f}'.format(corr_eval_summary['PA-MPJPE'] - van_eval_summary['PA-MPJPE'],
                                                                                              corr_eval_summary['MPJPE'] - van_eval_summary['MPJPE'],
                                                                                              corr_eval_summary['x'] - van_eval_summary['x'],
                                                                                              corr_eval_summary['y'] - van_eval_summary['y'],
                                                                                              corr_eval_summary['z'] - van_eval_summary['z'],))

def get_datasets(hybrik_cfg, config):
    trainsets = []
    for dataset in config.trainsets:
        if dataset == 'PW3D':
            trainset = PW3D(
                cfg=hybrik_cfg,
                ann_file='3DPW_train_new_fresh.json',
                train=False,
                root='/media/ExtHDD/Mohsen_data/3DPW'
            )
        elif dataset == 'MPii':
            trainset = MPII(
                cfg=hybrik_cfg,
                annot_dir='/media/ExtHDD/Mohsen_data/mpii_human_pose/mpii_cliffGT.npz',
                image_dir='/media/ExtHDD/Mohsen_data/mpii_human_pose/',
            )
        elif dataset == 'HP3D':
            trainset = HP3D(
                cfg=hybrik_cfg,
                ann_file='train_v2',   # dumb adjustment...
                train=False,
                root='/media/ExtHDD/luke_data/HP3D'
            )
        else:
            raise NotImplementedError
        trainsets.append(trainset)

    if config.testset == 'PW3D':
        testset = PW3D(
            cfg=hybrik_cfg,
            ann_file='3DPW_test_new_fresh.json',
            train=False,
            root='/media/ExtHDD/Mohsen_data/3DPW'
        )
    elif dataset == 'HP3D':
        raise NotImplementedError    # Need to extract the test img frames
    else:
        raise NotImplementedError
        
    return trainsets, testset

def main_worker(hybrik_cfg, config): 
    print('USING HYBRIK VER: {}'.format(config.hybrIK_version))
    hybrik_model = load_pretrained_hybrik(config, hybrik_cfg)
    hybrik = hybrik_model.to('cpu')

    if config.use_multi_distal:
        cnet = multi_distal(config)
        # TODO: MULTI-DISTAL R-CNET
    else:
        cnet = adapt_net(config, target_kpts=config.distal_kpts,)
        R_cnet = adapt_net(config, target_kpts=config.proximal_kpts, R=True,
                           in_kpts=[kpt for kpt in range(17) if kpt not in config.proximal_kpts])

    cnet_trainsets, cnet_testset = get_datasets(hybrik_cfg, config)

    if 'make_trainset' in config.tasks:
        make_trainsets(hybrik, cnet_trainsets, config)
    if 'make_testset' in config.tasks: 
        make_testset(hybrik, cnet_testset, config)
    if 'train_CNet' in config.tasks:
        cnet.train()
    if 'train_RCNet' in config.tasks:
        R_cnet.train()
    if 'test' in config.tasks:
        test(hybrik, cnet, R_cnet, cnet_testset, config)

if __name__ == "__main__":    
    config = get_config()
    hybrik_cfg = update_config(config.hybrik_cfg) 
    main_worker(hybrik_cfg, config)

