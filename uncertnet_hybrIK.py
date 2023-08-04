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

def print_test_summary(summary):
    '''
    args:
        summary: dict of dicts, where each dict is a summary of the testset using a different backbone
    '''
    print('\n##### TEST SUMMARY #####')
    for backbone in summary.keys():
        print('-- {}: --'.format(backbone), end=' ')
        for test in summary[backbone].keys():
            print('\n   {}:    '.format(test if test == 'vanilla' else ('   ' + test)), end=' ')
            for key in summary[backbone][test].keys():
                if test == 'vanilla': 
                    print('{}: {:.2f},'.format(key, summary[backbone][test][key]), end=' ')
                else:
                    diff = summary[backbone][test][key] - summary[backbone]['vanilla'][key]
                    print('{}: {:.2f},'.format(key, diff), end=' ')
        print('\n')

def test(backbone, cnet, R_cnet, testset, config):
    cnet.load_cnets()
    if config.test_adapt and (config.TTT_loss == 'consistency'): R_cnet.load_cnets()
    if backbone is not None: backbone.to(config.device)

    summary = {}
    for test_path, test_scale, test_backbone in zip(config.cnet_testset_paths, 
                                                    config.cnet_testset_scales, 
                                                    config.cnet_testset_backbones):
        summary[test_backbone] = {}
        TTT_eval_summary = None
        cnet.load_cnets(print_str=False)
        if config.test_adapt:
            if config.test_adapt and (config.TTT_loss == 'consistency'): R_cnet.load_cnets(print_str=False)
            TTT_eval_summary = eval_gt(cnet, R_cnet, config, backbone, testset, 
                                       testset_path=test_path, backbone_scale=test_scale, 
                                       test_cnet=True, test_adapt=True, use_data_file=config.TTT_from_file)
        cnet.load_cnets(print_str=False)
        eval_summary = eval_gt(cnet, R_cnet, config, testset, backbone, 
                               testset_path=test_path, backbone_scale=test_scale, test_cnet=True, use_data_file=True)
        summary[test_backbone]['vanilla'] = eval_summary['backbone']
        summary[test_backbone]['w/CN'] = eval_summary['corrected']
        if TTT_eval_summary: summary[test_backbone]['+TTT'] = TTT_eval_summary['corrected']
    print_test_summary(summary)

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
    if any([(task in config.tasks) for task in 
            ['make_trainsets', 'make_trainsets']]):
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
    else:
        return trainsets, None
    if 'test' in config.tasks:
        if config.testset == 'PW3D':
            testset = PW3D(
                cfg=backbone_cfg,
                ann_file='3DPW_test_new_fresh.json',
                train=False,
                root='/media/ExtHDD/Mohsen_data/3DPW')
        elif dataset == 'HP3D':
            raise NotImplementedError
        else:
            raise NotImplementedError
    else: 
        testset = None
    return trainsets, testset

def load_hybrik(config):
    ''' Load the backbone 3D pose estimation model '''
    model_cfg = update_config(config.hybrik_cfg) 
    print('USING HYBRIK VER: {}'.format(config.hybrIK_version))
    model = load_pretrained_hybrik(config, model_cfg)
    model = model.to('cpu')
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
    cnet, R_cnet = setup_adapt_nets(config)
    if ('make_trainsets' in config.tasks) or ('make_testset' in config.tasks) or (config.TTT_from_file == False):
        backbone_model, backbone_cfg = load_hybrik(config)
        cnet_trainsets, cnet_testset = get_datasets(backbone_cfg, config)
    else:
        backbone_model, backbone_cfg, cnet_trainsets, cnet_testset = None, None, None, None

    for task in config.tasks:
        if task == 'make_trainsets':
            make_trainsets(backbone_model, cnet_trainsets, config)
        if task == 'make_testset':
            make_testset(backbone_model, cnet_testset, config)
        if task == 'train_CNet':
            cnet.train()
        if task == 'make_RCNet_trainset':
            cnet.write_train_preds()
        if task == 'train_RCNet':
            R_cnet.train()
        if task == 'test':
            test(backbone_model, cnet, R_cnet, cnet_testset, config)
        if task == 'make_mmlab_test':
            make_mmlab_test(backbone_model, cnet, R_cnet, config)
    print("All Done!")

if __name__ == "__main__":    
    config = get_config()
    main_worker(config)

