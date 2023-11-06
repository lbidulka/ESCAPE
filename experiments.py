"""uncertnet script for HybrIK"""
import os
import sys
from pathlib import Path
import pickle

# path_root = Path(__file__)#.parents[3]
# sys.path.append(str(path_root))
# sys.path.append(str(path_root)+ 'backbones/HybrIK/hybrik/')

# print("\n", os.getcwd(), "\n", sys.path, "\n")

# import optuna
import json

import numpy as np
import torch
from torchvision import transforms as T
det_transform = T.Compose([T.ToTensor()])

from datasets.hybrik import load_hybrik, get_datasets, make_hybrik_pred_dataset
from cnet.multi_distal import multi_distal
from cnet.full_body import adapt_net
from cnet.full_body_feats import adapt_net as adapt_net_feats
from cnet.cotrain import CoTrainer
from core.cnet_eval import eval_gt
from config import get_config

from utils.optuna_objectives import optuna_objective
from utils.AMASS import make_amass_kpts
from utils.output_reporting import plot_TTT_loss, test_trainsets, plot_energies, print_test_summary


def test(cnet, R_cnet, config, print_summary=True, agora_out=False):
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

    allsets_summary = {}
    for testset_name, info in config.testset_info.items():
        summary = {}
        for test_path, test_scale, test_backbone in zip(info['paths'], info['scales'], info['backbones']):
            summary[test_backbone] = {}
            TTT_eval_summary = None
            cnet.load_cnets(print_str=False)
            if config.test_adapt:
                if config.test_adapt and (config.TTT_loss == 'consistency'): R_cnet.load_cnets(print_str=False)
                TTT_eval_summary = eval_gt(cnet, R_cnet, config, backbone, testset, 
                                        testset_path=test_path, backbone_scale=test_scale, 
                                        test_adapt=True, use_data_file=config.TTT_from_file,
                                        subset=config.test_eval_subsets[testset_name],
                                        agora_out=agora_out)
            cnet.load_cnets(print_str=False)
            eval_summary = eval_gt(cnet, R_cnet, config, testset, backbone, 
                                testset_path=test_path, backbone_scale=test_scale, use_data_file=True,
                                subset=config.test_eval_subsets[testset_name],
                                agora_out=agora_out,)
            summary[test_backbone]['vanilla'] = eval_summary['backbone']
            summary[test_backbone]['w/CN'] = eval_summary['corrected']
            summary[test_backbone]['at V'] = eval_summary['attempted_backbone']
            summary[test_backbone]['at C'] = eval_summary['attempted_corr']
            if TTT_eval_summary: 
                summary[test_backbone]['+TTT'] = TTT_eval_summary['corrected']
                summary[test_backbone]['aTTT'] = TTT_eval_summary['attempted_corr']
        allsets_summary[testset_name] = summary
    if print_summary: 
        print_test_summary(config, allsets_summary)
    return summary

def setup_adapt_nets(config):
    ''' Define the adaptation networks CNet and RCNet '''
    if config.use_multi_distal:
        cnet = multi_distal(config)
        R_cnet = None # TODO: MULTI-DISTAL R-CNET
    else:
        if config.use_features:
            cnet = adapt_net_feats(config, target_kpts=config.cnet_targets, 
                             in_kpts=config.EVAL_JOINTS)
            R_cnet = adapt_net_feats(config, target_kpts=config.rcnet_targets, R=True, 
                               in_kpts=config.EVAL_JOINTS)
        else:
            cnet = adapt_net(config, target_kpts=config.cnet_targets,
                            in_kpts=config.EVAL_JOINTS)
            R_cnet = adapt_net(config, target_kpts=config.rcnet_targets,
                            R=True,
                            in_kpts=config.EVAL_JOINTS)
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
        elif task == 'cotrain':
            cotrainer = CoTrainer(cnet, R_cnet)
            cotrainer.train()

        elif task == 'test':
            test(cnet, R_cnet, config)
        elif task == 'plot_TTT_loss':
            plot_TTT_loss(config)
        elif task == 'plot_TTT_train_corr':
            test_trainsets(cnet, R_cnet, config,)
            plot_TTT_loss(config, task='train')
        elif task == 'plot_test_energies':
            plot_energies(config, task='test')
        elif task == 'plot_train_energies':
            test_trainsets(cnet, R_cnet, config,)
            plot_energies(config, task='train')
        elif task == 'export_agora':
            test(cnet, R_cnet, config, agora_out=True)

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
        
        elif task == 'get_inference_time':
            from utils.inference_timing import get_inference_time
            get_inference_time(config, cnet, R_cnet)
        else:
            raise NotImplementedError
    

if __name__ == "__main__":    
    config = get_config()
    main_worker(config)

