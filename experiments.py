"""uncertnet script for HybrIK"""
import os
import sys
from pathlib import Path

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

def plot_TTT_loss(config):
    '''
    Loads up the losses from the testset and plots them
    '''
    print("Plotting TTT Losses...")
    # combine rcnet target names
    rcnet_targets_name = ''
    for target in config.rcnet_targets_name:
        rcnet_targets_name += target + '_'
    TTT_losses_outpath = '../../outputs/TTT_losses/'
    losses = []
    # iterate overall all backbones 
    for test_path in config.cnet_testset_paths:
        backbone_name = test_path.split('/')[-1].split('.')[-2]
        loss_path = TTT_losses_outpath + '_'.join([backbone_name, config.TTT_loss, 'losses.npy'])
        bb_losses = np.load(loss_path)
        losses.append(bb_losses)
    losses = np.concatenate(losses)

    save_dir = '../../outputs/testset/'
    if config.TTT_loss == 'consistency':
        save_dir += rcnet_targets_name + '_consist_losses.png'
        title = rcnet_targets_name + ' Consist. Loss vs GT 3D MSE'
        utils.quick_plot.simple_2d_plot(losses, save_dir=save_dir, title=title, 
                                        xlabel='Consistency Loss', ylabel='GT 3D MSE Loss',
                                        # x_lim=[0, 25], y_lim=[0, 7500])
                                        # x_lim=[0, 1000], y_lim=[0, 7500])
                                        x_lim=[0, 50], y_lim=[0, 250])
    if config.TTT_loss == 'reproj_2d':
        save_dir += rcnet_targets_name + '_reproj_2d_losses.png'
        title = rcnet_targets_name + ' 2D reproj. Loss vs GT 3D MSE'
        utils.quick_plot.simple_2d_plot(losses, save_dir=save_dir, title=title, 
                                        xlabel='2D reproj Loss', ylabel='GT 3D MSE Loss',
                                        x_lim=[0,1], y_lim=[0, 7500])

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
        cnet = adapt_net(config, target_kpts=config.cnet_targets,
                        in_kpts=[kpt for kpt in range(17) if kpt not in [9,10]])
        R_cnet = adapt_net(config, target_kpts=config.rcnet_targets,
                           R=True,
                           in_kpts=[kpt for kpt in range(17) if kpt not in [9,10]])
                        #    in_kpts=[kpt for kpt in range(17) if kpt not in config.rcnet_targets])
    return cnet, R_cnet

def main_worker(config): 
    cnet, R_cnet = setup_adapt_nets(config)
    for task in config.tasks:
        if task == 'make_trainsets':
            make_hybrik_pred_dataset(config, 'train')
        elif task == 'make_testset':
            make_hybrik_pred_dataset(config, 'test')
        elif task == 'train_CNet':
            cnet.train()
        elif task == 'make_RCNet_trainset':
            cnet.write_train_preds()
        elif task == 'train_RCNet':
            R_cnet.train()
        elif task == 'test':
            test(cnet, R_cnet, config)
        elif task == 'plot_TTT_loss':
            plot_TTT_loss(config)
        elif task == 'TTT_optuna':
            study = optuna.create_study(directions=['minimize', 'minimize'])
            study.optimize(optuna_objective(config, cnet, R_cnet, test), n_trials=config.optuna_num_trials,)

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

        else:
            raise NotImplementedError
    

if __name__ == "__main__":    
    config = get_config()
    main_worker(config)

