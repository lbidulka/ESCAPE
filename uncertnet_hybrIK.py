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

def make_trainsets(hybrik, trainsets, config):
    with torch.no_grad():
        for i, trainset in enumerate(trainsets):
            print('##### Creating CNET {} Trainset #####'.format(config.trainsets[i]))
            create_cnet_dataset_w_HybrIK(hybrik, config, trainset, 
                                         dataset=config.trainsets[i], task='train',)

def make_testset(hybrik, testset, config):
    with torch.no_grad():
        print('##### Creating CNET {} Testset #####'.format(config.testset))
        create_cnet_dataset_w_HybrIK(hybrik, config, testset, 
                                     dataset=config.trainset, task='test',)

def test(backbone, cnet, R_cnet, testset, config):
    cnet.load_cnets()
    if config.test_adapt and (config.TTT_loss == 'consistency'): R_cnet.load_cnets()
    if backbone is not None: backbone.to(config.device)

    for test_path, test_scale, test_backbone in zip(config.cnet_testset_paths, 
                                                    config.cnet_testset_scales, 
                                                    config.cnet_testset_backbones):
        print('\n##### {} TESTSET ERRS #####'.format(test_backbone))
        if config.test_adapt:
            print('--- CNet w/TTT: --- ')
            TTT_eval_summary = eval_gt(backbone, cnet, R_cnet, config, testset, 
                                       testset_path=test_path, backbone_scale=test_scale, 
                                       test_cnet=True, test_adapt=True, use_data_file=config.TTT_from_file)
            TTT_corr_eval_summary = TTT_eval_summary['corrected']
            print('XYZ_14 PA-MPJPE: {:2f} | MPJPE: {:2f}, x: {:2f}, y: {:.2f}, z: {:2f}'.format(TTT_corr_eval_summary['PA-MPJPE'], 
                                                                                                TTT_corr_eval_summary['MPJPE'], 
                                                                                                TTT_corr_eval_summary['x'], 
                                                                                                TTT_corr_eval_summary['y'], 
                                                                                                TTT_corr_eval_summary['z']))    
        print('--- CNet Only: --- ')
        cnet.load_cnets()
        if config.test_adapt and (config.TTT_loss == 'consistency'): R_cnet.load_cnets()
        eval_summary = eval_gt(backbone, cnet, R_cnet, config, testset, 
                               testset_path=test_path, backbone_scale=test_scale, test_cnet=True, use_data_file=True)
        corr_eval_summary = eval_summary['corrected']
        van_eval_summary = eval_summary['backbone']
        print('XYZ_14 PA-MPJPE: {:2f} | MPJPE: {:2f}, x: {:2f}, y: {:.2f}, z: {:2f}'.format(corr_eval_summary['PA-MPJPE'], 
                                                                                            corr_eval_summary['MPJPE'], 
                                                                                            corr_eval_summary['x'], 
                                                                                            corr_eval_summary['y'], 
                                                                                            corr_eval_summary['z']))
        print('--- Vanilla: --- ')
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

def make_mmlab_test(hybrik, cnet, R_cnet, config):
    cnet.load_cnets()
    if config.test_adapt and (config.TTT_loss == 'consistency'): R_cnet.load_cnets()
    hybrik = hybrik.to(config.device)
    
    print('\n##### CNET SAMPLE GEN FOR mmlab 3DPW  #####')
    if config.test_adapt:
        print('--- CNet w/TTT: --- ')
        TTT_corr_eval_summary = eval_gt(hybrik, cnet, R_cnet, config, 
                                test_cnet=True, test_adapt=True, use_data_file=config.TTT_from_file)
    print('--- CNet Only: --- ')
    cnet.load_cnets()
    if config.test_adapt and (config.TTT_loss == 'consistency'): R_cnet.load_cnets()
    corr_eval_summary = eval_gt(hybrik, cnet, R_cnet, config,
                                test_cnet=True, use_data_file=True, mmlab_out=True)

def get_datasets(backbone_cfg, config):
    trainsets = []
    if config.backbone == 'spin':
        return trainsets, None
    elif any([(task in config.tasks) for task in 
            ['make_trainset', 'train_RCNet', 'train_CNet']]):
        for dataset in config.trainsets:
            if dataset == 'PW3D':
                trainset = PW3D(
                    cfg=backbone_cfg,
                    ann_file='3DPW_train_new_fresh.json',
                    train=False,
                    root='/media/ExtHDD/Mohsen_data/3DPW')
            elif dataset == 'MPii':
                trainset = MPII(
                    cfg=backbone_cfg,
                    annot_dir='/media/ExtHDD/Mohsen_data/mpii_human_pose/mpii_cliffGT.npz',
                    image_dir='/media/ExtHDD/Mohsen_data/mpii_human_pose/',)
            elif dataset == 'HP3D':
                trainset = HP3D(
                    cfg=backbone_cfg,
                    ann_file='train_v2',   # dumb adjustment...
                    train=False,
                    root='/media/ExtHDD/luke_data/HP3D')
            else:
                raise NotImplementedError
            trainsets.append(trainset)
    if 'test' in config.tasks:
        if config.testset == 'PW3D':
            testset = PW3D(
                cfg=backbone_cfg,
                ann_file='3DPW_test_new_fresh.json',
                train=False,
                root='/media/ExtHDD/Mohsen_data/3DPW')
        elif dataset == 'HP3D':
            raise NotImplementedError    # Need to extract the test img frames
        else:
            raise NotImplementedError
    else: 
        testset = None
    return trainsets, testset

def load_backbone(config):
    ''' Load the backbone 3D pose estimation model '''
    if config.backbone == 'hybrik':
        model_cfg = update_config(config.hybrik_cfg) 
        print('USING HYBRIK VER: {}'.format(config.hybrIK_version))
        model = load_pretrained_hybrik(config, model_cfg)
        model = model.to('cpu')
    elif config.backbone == 'spin':
        if (config.test_adapt and not config.TTT_from_file):
            raise NotImplementedError
        model = None
        model_cfg = None
    else:
        raise NotImplementedError
    return model, model_cfg

def setup_adapt_nets(config):
    ''' Define the adaptation networks CNet and RCNet '''
    if config.use_multi_distal:
        cnet = multi_distal(config)
        R_cnet = None # TODO: MULTI-DISTAL R-CNET
    else:
        cnet = adapt_net(config, target_kpts=config.cnet_targets,
                        in_kpts=[kpt for kpt in range(17) if kpt not in [9,10]])
        R_cnet = adapt_net(config, target_kpts=config.rcnet_targets,  #config.proximal_kpts, 
                           R=True,)
                        #    in_kpts=[kpt for kpt in range(17) if kpt not in config.rcnet_targets])
    return cnet, R_cnet

def main_worker(config): 
    backbone_model, backbone_cfg = load_backbone(config)    
    cnet, R_cnet = setup_adapt_nets(config)    
    cnet_trainsets, cnet_testset = get_datasets(backbone_cfg, config)

    if 'make_trainset' in config.tasks:
        make_trainsets(backbone_model, cnet_trainsets, config)
    if 'make_testset' in config.tasks: 
        make_testset(backbone_model, cnet_testset, config)
    if 'train_CNet' in config.tasks:
        cnet.train()
    if 'make_RCNet_trainset' in config.tasks:
        cnet.write_train_preds()
    if 'train_RCNet' in config.tasks:
        R_cnet.train()
    if 'test' in config.tasks:
        test(backbone_model, cnet, R_cnet, cnet_testset, config)
    if 'make_mmlab_test' in config.tasks:
        make_mmlab_test(backbone_model, cnet, R_cnet, config)
    print("All Done!")

if __name__ == "__main__":    
    config = get_config()
    main_worker(config)

