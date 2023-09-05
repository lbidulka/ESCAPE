"""uncertnet script for HybrIK"""
import os
import sys
from pathlib import Path
import pickle

# path_root = Path(__file__)#.parents[3]
# sys.path.append(str(path_root))
# sys.path.append(str(path_root)+ 'backbones/HybrIK/hybrik/')

# print("\n", os.getcwd(), "\n", sys.path, "\n")

import optuna
import json

import numpy as np
import torch
from torchvision import transforms as T
det_transform = T.Compose([T.ToTensor()])

from datasets.hybrik import load_hybrik, get_datasets, make_hybrik_pred_dataset
from cnet.multi_distal import multi_distal
from cnet.full_body import adapt_net
from core.cnet_eval import eval_gt
from config import get_config

import utils.quick_plot
from utils.optuna_objectives import optuna_objective
from utils.AMASS import make_amass_kpts

def plot_TTT_loss(config, task='test'):
    '''
    Loads up the losses and plots them
    '''
    print("Plotting TTT Losses...")
    loss_paths = config.cnet_testset_paths if task == 'test' else config.cnet_trainset_paths
    # combine rcnet target names
    rcnet_targets_name = ''
    for target in config.rcnet_targets_name:
        rcnet_targets_name += target + '_'
    TTT_losses_outpath = '../../outputs/TTT_losses/'
    losses = []
    # iterate overall all backbones 
    for out_path in loss_paths:
        backbone_name = out_path.split('/')[-1].split('.')[-2]
        dataset_name = out_path.split('/')[-2]
        loss_path = TTT_losses_outpath
        if dataset_name != 'PW3D':
            loss_path += dataset_name + '_'
        loss_path += '_'.join([backbone_name, config.TTT_loss, 'losses.npy'])
        bb_losses = np.load(loss_path)
        losses.append(bb_losses)
    losses = np.concatenate(losses)

    save_dir = '../../outputs/testset/test_' if task == 'test' else '../../outputs/trainset/train_'
    title = 'Testset ' if task == 'test' else 'Trainset '
    if config.TTT_loss == 'consistency':
        save_dir += rcnet_targets_name + '_consist_losses.png'
        title += rcnet_targets_name + ' Consist. Loss vs GT 3D MSE'
        utils.quick_plot.simple_2d_plot(losses, save_dir=save_dir, title=title, 
                                        xlabel='Consistency Loss', ylabel='GT 3D MSE Loss',
                                        # x_lim=[0, 25], y_lim=[0, 7500])
                                        # x_lim=[0, 1000], y_lim=[0, 7500])
                                        x_lim=[0, 20], y_lim=[0, 200], alpha=0.15)
    if config.TTT_loss == 'reproj_2d':
        save_dir += rcnet_targets_name + '_reproj_2d_losses.png'
        title += rcnet_targets_name + ' 2D reproj. Loss vs GT 3D MSE'
        utils.quick_plot.simple_2d_plot(losses, save_dir=save_dir, title=title, 
                                        xlabel='2D reproj Loss', ylabel='GT 3D MSE Loss',
                                        x_lim=[0,1], y_lim=[0, 7500], alpha=0.1)

def plot_TTT_train_corr(cnet, R_cnet, config, print_summary=True):
    if not config.test_adapt or not (config.TTT_loss == 'consistency'):
        raise NotImplementedError
    # Get HybrIK model if required
    if config.TTT_from_file == False:
        raise NotImplementedError
    else:
        backbone, backbone_cfg, testset = None, None, None
    # Load CNet & R-CNet, then test
    if backbone is not None: backbone.to(config.device)

    subsets = []
    for trainpath in config.cnet_trainset_paths:
        data = torch.from_numpy(np.load(trainpath)).float().permute(1,0,2)
        set_len = data.shape[0]
        subset = np.random.choice(set_len, 
                                  min(config.test_eval_limit, set_len), 
                                  replace=False)
        subsets.append(subset)

    # small function to update the dict with mean of new value and old value
    def update_bb_summary(bb_summary, test_key, new_vals):
        # new_vals has keys: [PA-MPJPE, MPJPE, x, y, z]
        if test_key not in bb_summary.keys():
            bb_summary[test_key] = {}
        for k in new_vals.keys():
            if k not in bb_summary[test_key].keys():
                bb_summary[test_key][k] = new_vals[k]
            else:
                bb_summary[test_key][k] = (bb_summary[test_key][k] + new_vals[k])/2

    summary = {}
    for train_path, train_backbone, subset in zip(config.cnet_trainset_paths, 
                                  config.train_backbone_list, 
                                  subsets):
        train_scale = 2.2 if train_backbone == 'hybrik' else 1.0
        if train_backbone not in summary.keys():
            summary[train_backbone] = {}
        TTT_eval_summary = None
        cnet.load_cnets(print_str=False)
        R_cnet.load_cnets(print_str=False)
        TTT_eval_summary = eval_gt(cnet, R_cnet, config, backbone, testset, 
                                    testset_path=train_path, backbone_scale=train_scale, 
                                    test_cnet=True, test_adapt=True, subset=subset,
                                    use_data_file=config.TTT_from_file)
        
        cnet.load_cnets(print_str=False)
        eval_summary = eval_gt(cnet, R_cnet, config, testset, backbone, 
                               testset_path=train_path, backbone_scale=train_scale, test_cnet=True, 
                               subset=subset, use_data_file=True)
        
        update_bb_summary(summary[train_backbone], 'vanilla', eval_summary['backbone'])
        update_bb_summary(summary[train_backbone], 'w/CN', eval_summary['corrected'])
        update_bb_summary(summary[train_backbone], '+TTT', TTT_eval_summary['corrected'])

    if print_summary: print_test_summary(summary)

    plot_TTT_loss(config, task='train')
        

def print_test_summary(summary):
    '''
    args:
        summary: dict of dicts, where each dict is a summary of the testset using a different backbone
    '''
    print('\n##### TEST SUMMARY #####')
    print('P1: PA-MPJPE, P2: MPJPE')
    for backbone in summary.keys():
        print('-- {}: --'.format(backbone), end=' ')
        for test in summary[backbone].keys():
            print('\n   {}:    '.format(test if test == 'vanilla' else ('   ' + test)), end=' ')
            for key in summary[backbone][test].keys():
                if test == 'vanilla': 
                    print('{}: {:7.2f},'.format(key, summary[backbone][test][key]), end=' ')
                else:
                    diff = summary[backbone][test][key] - summary[backbone]['vanilla'][key]
                    print('{}: {:7.2f},'.format(key, diff), end=' ')
        print('\n')

def test(cnet, R_cnet, config, print_summary=True):
    # Get HybrIK model if required
    if config.TTT_from_file == False:
        backbone, backbone_cfg = load_hybrik(config)
        _, testset = get_datasets(backbone_cfg, config)
    else:
        backbone, backbone_cfg, testset = None, None, None
    # Load CNet & R-CNet, then test
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
    if print_summary: print_test_summary(summary)
    return summary

def setup_adapt_nets(config):
    ''' Define the adaptation networks CNet and RCNet '''
    if config.use_multi_distal:
        cnet = multi_distal(config)
        R_cnet = None # TODO: MULTI-DISTAL R-CNET
    else:
        cnet = adapt_net(config, target_kpts=config.cnet_targets,)
                        # in_kpts=[kpt for kpt in range(17) if kpt not in config.cnet_targets])
                        # in_kpts=[kpt for kpt in range(17) if kpt not in [9,10]])
        R_cnet = adapt_net(config, target_kpts=config.rcnet_targets,
                           R=True,)
                        #    in_kpts=[kpt for kpt in range(17) if kpt not in [9,10]])
                        #    in_kpts=[kpt for kpt in range(17) if kpt not in config.rcnet_targets])
    return cnet, R_cnet

def main_worker(config): 
    cnet, R_cnet = setup_adapt_nets(config)
    for task in config.tasks:
        if task == 'make_trainsets':
            make_hybrik_pred_dataset(config, 'train')
        elif task == 'make_testset':
            make_hybrik_pred_dataset(config, 'test')
        elif task == 'make_kpt_amass':
            make_amass_kpts(config)
        elif task == 'pretrain_CNet':
            cnet.train(pretrain_AMASS=True)
        elif task == 'train_CNet':
            cnet.train(continue_train=config.pretrain_AMASS)
        elif task == 'make_RCNet_trainset':
            cnet.write_train_preds()
        elif task == 'pretrain_RCNet':
            R_cnet.train(pretrain_AMASS=True)
        elif task == 'train_RCNet':
            R_cnet.train(continue_train=config.pretrain_AMASS)
        elif task == 'test':
            test(cnet, R_cnet, config)
        elif task == 'plot_TTT_loss':
            plot_TTT_loss(config)
        elif task == 'plot_TTT_train_corr':
            plot_TTT_train_corr(cnet, R_cnet, config,)
        elif task == 'optuna_CNet':
            study = optuna.create_study(directions=['minimize', 'minimize'])
            study.optimize(optuna_objective('CNet', config, cnet, R_cnet, test), n_trials=config.optuna_num_trials,)

            print(f"Number of trials on the Pareto front: {len(study.best_trials)}")
            trial_with_highest_mean = max(study.best_trials, key=lambda t: (t.values[0] + t.values[1])/2)
            print(f"Trial with highest mean(PA-MPJPE, MPJPE): ")
            print(f"\tnumber: {trial_with_highest_mean.number}")
            print(f"\tparams: {trial_with_highest_mean.params}")
            print(f"\tvalues: {trial_with_highest_mean.values}")
            
            log_json = {
                "number": trial_with_highest_mean.number,
                "params": trial_with_highest_mean.params,
                "values": trial_with_highest_mean.values,
            }
            json.dump(log_json, open(config.optuna_log_path + 'CNet_best_mean_params.json', 'w'))
            with open(config.optuna_log_path + 'CNet_study.pkl', 'wb') as file: 
                pickle.dump(study, file)

        elif task == 'optuna_TTT':
            study = optuna.create_study(directions=['minimize', 'minimize'])
            study.optimize(optuna_objective('TTT', config, cnet, R_cnet, test), n_trials=config.optuna_num_trials,)

            print(f"Number of trials on the Pareto front: {len(study.best_trials)}")
            trial_with_highest_mean = max(study.best_trials, key=lambda t: (t.values[0] + t.values[1])/2)
            print(f"Trial with highest mean(PA-MPJPE, MPJPE): ")
            print(f"\tnumber: {trial_with_highest_mean.number}")
            print(f"\tparams: {trial_with_highest_mean.params}")
            print(f"\tvalues: {trial_with_highest_mean.values}")
            
            log_json = {
                "number": trial_with_highest_mean.number,
                "params": trial_with_highest_mean.params,
                "values": trial_with_highest_mean.values,
            }
            json.dump(log_json, open(config.optuna_log_path + 'TTT_best_mean_params.json', 'w'))
            with open(config.optuna_log_path + 'TTT_study.pkl', 'wb') as file: 
                pickle.dump(study, file)

        else:
            raise NotImplementedError
    

if __name__ == "__main__":    
    config = get_config()
    main_worker(config)

