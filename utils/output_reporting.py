import torch
import numpy as np

import utils.quick_plot
from core.cnet_eval import eval_gt

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
                if key in ['PA-MPJPE', 'MPJPE', 'x',]:
                    print(' || ', end=' ')
                if test == 'vanilla' or test == 'at V' or key == '(num_samples)':
                    print('{}: {:7.2f},'.format(key, summary[backbone][test][key]), end=' ')
                else:
                    if test == 'at C':
                        ref_test = 'at V'
                    else:
                        ref_test = 'vanilla'
                    diff = summary[backbone][test][key] - summary[backbone][ref_test][key]
                    print('{}: {:7.2f},'.format(key, diff), end=' ')
        print('\n')

def plot_energies(config):
    ''' 
    loads up the losses (including energies) and plots them
    '''
    print("Plotting Energies + Losses...")
    loss_paths = config.cnet_testset_paths
    energies_outpath = '../../outputs/energies/'
    # iterate overall all backbones
    energies_losses = []
    for out_path in loss_paths:
        backbone_name = out_path.split('/')[-1].split('.')[-2]
        dataset_name = out_path.split('/')[-2]
        loss_path = energies_outpath
        if dataset_name != 'PW3D':
            loss_path += dataset_name + '_'
        loss_path += '_'.join([backbone_name, 'energies.npy'])
        bb_losses = np.load(loss_path)
        energies_losses.append(bb_losses)
    energies_losses = np.concatenate(energies_losses)

    save_dir = '../../outputs/testset/test_'
    title = 'Testset '
    save_dir += 'bb_energies_losses.png'
    title += 'Backbone Pred Energies vs GT 3D MSE'
    utils.quick_plot.simple_2d_plot(energies_losses, save_dir=save_dir, title=title,
                                    xlabel='BB Sample Energies', ylabel='GT 3D MSE Loss',
                                    x_lim=[250, 900], y_lim=[0, 20000], alpha=0.1)


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
                                        x_lim=[0, 75], y_lim=[0, 200], alpha=0.15)
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