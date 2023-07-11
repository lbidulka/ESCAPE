"""uncertnet script for HybrIK"""
import os
import sys
from pathlib import Path
from types import SimpleNamespace
import copy

import numpy as np
import torch

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


from utils import errors
from datasets.mpii import mpii_dataset
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
    R_cnet.load_cnets()
    hybrik = hybrik.to(config.device)

    print('\n##### 3DPW TESTSET ERRS #####\n')
    tot_corr_MPJPE = eval_gt(hybrik, cnet, R_cnet, config, gt_test_dataset_3dpw, 
                                test_cnet=True, use_data_file=True)
    print('\n--- Vanilla: --- ')
    with torch.no_grad():
        gt_tot_err = eval_gt(hybrik, cnet, R_cnet, config, gt_test_dataset_3dpw, 
                             test_cnet=False, use_data_file=True)

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
            trainset = mpii_dataset(
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

def main_worker(hybrik_cfg, hybrIK_model, config): 
    print(' USING HYBRIK VER: {}'.format(config.hybrIK_version))
    hybrik = hybrIK_model.to('cpu')

    if config.use_multi_distal:
        cnet = multi_distal(config)
        # TODO: MULTI-DISTAL R-CNET
    else:
        cnet = adapt_net(config, target_kpts=config.distal_kpts,)
        R_cnet = adapt_net(config, target_kpts=config.proximal_kpts, R=True,)

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
    hybrik_model = load_pretrained_hybrik(config, hybrik_cfg)
    main_worker(hybrik_cfg, hybrik_model, config)

