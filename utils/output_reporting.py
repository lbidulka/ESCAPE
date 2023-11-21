import torch
import numpy as np
import matplotlib.pyplot as plt

import utils.quick_plot
from core.cnet_eval import eval_gt

def print_test_summary(config, all_summary):
    '''
    args:
        summary: dict of dicts, where each dict is a summary of the testset using a different backbone
    '''
    print('\n##### TEST SUMMARY #####')
    for dataset in all_summary.keys():
        summary = all_summary[dataset]
        print('{}:'.format(dataset))
        for backbone in summary.keys():
            print('-- {}: --'.format(backbone,), end=' ')
            for test in summary[backbone].keys():
                if test in ['at V', 'at C'] and not config.use_cnet_energy:
                    pass
                else:
                    print('\n   {}:    '.format(test if test == 'vanilla' else ('   ' + test)), end=' ')
                    for key in summary[backbone][test].keys():
                        if key in ['PA-MPJPE', 'MPJPE', 'x', 3]:
                            print(' || ', end=' ')
                        if test == 'vanilla' or test == 'at V' or key == '(num_samples)':
                            print('{}: {:7.2f},'.format(key, summary[backbone][test][key]), end=' ')
                        else:
                            if test == 'at C' or test == 'aTTT':
                                ref_test = 'at V'
                            else:
                                ref_test = 'vanilla'
                            diff = summary[backbone][test][key] - summary[backbone][ref_test][key]
                            print('{}: {:7.2f},'.format(key, diff), end=' ')
            print('\n')

def plot_E_sep(config, task, dataset, cnet=False):
    ''' 
    loads up E scores + GT MSE and plots histogram of errs with & w/o E thresholding
    '''
    file_suffix = 'energies_cnet.npy' if cnet else 'energies.npy'

    if task == 'test':
        datasets = config.testsets
        loss_paths = []
        for set in datasets:
            for path in config.testset_info[set]['paths']:
                loss_paths.append(path)
    else:
        datasets = config.trainsets
        loss_paths = config.cnet_trainset_paths
            
    loss_paths = [path for path in loss_paths if dataset in path]
    if len(loss_paths) == 0:
        print(f"dataset {dataset} not found in loss_paths")
        return
    
    energies_outpath = '../../outputs/energies/'
    # iterate overall all backbones
    energies_losses = []
    for out_path in loss_paths:
        backbone_name = out_path.split('/')[-1].split('.')[-2]
        dataset_name = out_path.split('/')[-2]
        loss_path = energies_outpath
        loss_path += dataset_name + '_'
        loss_path += '_'.join([backbone_name, file_suffix])
        bb_losses = np.load(loss_path)
        energies_losses.append(bb_losses)
    energies_losses = np.concatenate(energies_losses)

    save_dir = '../../outputs/energies/plots/' + f'{dataset}/{dataset}_{backbone_name}_{task}_'
    if cnet:
        save_dir += 'E_histogram_cnet.png'
        title = f'{dataset} # {config.test_backbones[0]} Cnet Preds vs. GT Target MSE'
    else:
        save_dir += 'E_histogram.png'
        title = f'{dataset} # {config.test_backbones[0]} Poses vs. GT 3D MSE'

    # Get thresh for top 10% of MSE err samples
    num_tail = int(0.10*energies_losses[:,1].shape[0])
    tail_idxs = np.argpartition(energies_losses[:,1], -num_tail)[-num_tail:]
    check_thresh = int(energies_losses[tail_idxs,1].min())   # 7000

    E_thresh = config.E_thresh_cnet if cnet else config.energy_thresh # config.energy_thresh, 800
    # E_thresh = 600
    
    MSE_errs_full = energies_losses[:,1]
    MJPEs_full = energies_losses[:,2]
    Es_full = energies_losses[:,0]
    E_t_MSE_errs_full = MSE_errs_full[Es_full < E_thresh]  # normal
    E_t_MPJPEs_full = MJPEs_full[Es_full < E_thresh] 
    # E_t_MSE_errs_full = MSE_errs_full[Es_full > E_thresh]
    num_top10 = np.sum(MSE_errs_full > check_thresh)
    num_Et_top10 = np.sum(E_t_MSE_errs_full > check_thresh)

    frac_full = round(E_t_MSE_errs_full.shape[0] / MSE_errs_full.shape[0], 2)
    frac_top10p = round(num_Et_top10 / num_top10, 2)

    # avg error of all samples before and after E_thresh
    avg_MSE_full = np.mean(MSE_errs_full) / 1000
    avg_MSE_Et = np.mean(E_t_MSE_errs_full) / 1000
    avg_MPJPE_full = np.mean(MJPEs_full)
    avg_MPJPE_Et = np.mean(E_t_MPJPEs_full)

    print("\n|| {}: {} ||".format(dataset, backbone_name))    
    print('frac full: {}, frac top10p: {}'.format(frac_full, frac_top10p))
    print('MPJPE full: {:.1f}, MPJPE Et: {:.1f}'.format(avg_MPJPE_full, avg_MPJPE_Et))
    
    # Plotting
    print(f"Plotting Energies + Losses to {save_dir}")
    xlim = 50_000 if cnet else 30_000
    energies_losses_bound = energies_losses[energies_losses[:,1] < xlim] # remove some outliers for plotting
    MSE_errs = energies_losses_bound[:,1]
    Es = energies_losses_bound[:,0]
    E_t_MSE_errs = MSE_errs[Es < E_thresh]    # normal
    # E_t_MSE_errs = MSE_errs[Es > E_thresh]
    num_bins = 50
    ylims = {
        1: {    # CNET
            'train': {'MPii': [0, 750], 'HP3D': [0, 15_000]},
            'test': {'HP3D': [0, 250], 'PW3D': [0, 3_500]}
            },
        0: {    # BB
            'train': {'MPii': [0, 5_000], 'HP3D': [0, 25_000]},
            'test': {'HP3D': [0, 500], 'PW3D': [0, 6_000]}
            },
    }
    fig, ax = plt.subplots()
    ax.hist(MSE_errs, label='all MSE', bins=num_bins)
    ax.hist(E_t_MSE_errs, label='E_thresh MSE', bins=num_bins)
    ax.set(title=title, xlabel='GT 3D MSE Loss', ylabel='Count', 
           xlim=[0, 30_000], ylim=ylims[cnet][task][dataset])
    ax.legend()
    if save_dir is not None:
        plt.savefig(save_dir) 


def plot_energies(config, task):
    ''' 
    loads up the E scores + GT MSE and plots them
    '''
    print("Plotting Energies + Losses...")
    
    if task == 'test':
        loss_paths = config.cnet_testset_paths
        save_dir = '../../outputs/testset/test_'
        title = 'Testset '
    else:
        loss_paths = config.cnet_trainset_paths
        save_dir = '../../outputs/trainset/train_'
        title = 'Trainset '
    
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

    
    save_dir += 'bb_energies_losses.png'
    title += 'Backbone Pred Energies vs GT 3D MSE'
    utils.quick_plot.simple_2d_plot(energies_losses, save_dir=save_dir, title=title,
                                    xlabel='BB Sample Energies', ylabel='GT 3D MSE Loss',
                                    x_lim=[250, 900], y_lim=[0, 20000], alpha=0.1) 

def plot_TTT_loss(config, task='test'):
    '''
    Loads up the losses and plots them
    '''
    print("Plotting TTT Losses...", end=' ')
    datasets = config.testsets if task == 'test' else config.trainsets
    for dataset in datasets:
        if task == 'test':
            loss_paths = config.testset_info[dataset]['paths']
        else:
            loss_paths = config.cnet_trainset_paths
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
            # if dataset_name != 'PW3D':
            loss_path += dataset_name + '_'
            loss_path += '_'.join([backbone_name, config.TTT_loss, 'losses.npy'])
            bb_losses = np.load(loss_path)
            losses.append(bb_losses)
        losses = np.concatenate(losses)

        # want consistency loss vs GT cnet target MSE
        losses = losses[:, [0, 2]]
        losses[:, 1] /= 1000    # scale

        bb_name = config.test_backbones[0].upper() if len(config.test_backbones) == 1 else 'multi'
        save_dir = f'../../outputs/{task}set/{dataset}_{task}_{bb_name}'
        title = 'Testset ' if task == 'test' else 'Trainset '
        if config.TTT_loss == 'consistency':
            save_dir += '_consist_losses.png'
            title += rcnet_targets_name + ' Consist. Loss vs GT CNet MSE' #' Consist. Loss vs GT 3D MSE'

            title = f'{dataset} {config.test_backbones[0].upper()} Consistency Loss vs. GT CNet Prediction Err.'

            fig, ax = utils.quick_plot.simple_2d_plot(losses, save_dir=save_dir, title=title, 
                                            xlabel='Consistency Loss', ylabel='GT 3D MSE', 
                                            # data_label='TTA Loss',
                                            # x_lim=[0, 12.5], y_lim=[0, 125], 
                                            x_lim=[0, 12.5], y_lim=[0, 30], 
                                            alpha=0.05,
                                            # data_label='test sample loss',
                                            )
            # split data into 10 bins according to x values 0,1,2...,10
            losses_sorted = np.sort(losses, axis=0)
            
            # avg the y values in each bin
            bin_size = 1
            num_bins = 50
            
            # get the x values for each bin
            bin_edges = np.arange(0, num_bins*bin_size + 1, bin_size)
            bin_vals = np.arange(0.5, num_bins*bin_size, bin_size)

            # get the y values for each bin
            bin_ys = []
            for i, edge in enumerate(bin_edges[:-1]):
                ys = losses_sorted[np.logical_and(losses_sorted[:,0] > edge, losses_sorted[:,0] < bin_edges[i+1])][:,1]
                bin_ys.append(np.mean(ys))

            # replace nans with 0's and trim to same length as bin_vals
            lim = 11
            bin_ys = np.nan_to_num(bin_ys)
            bin_ys = bin_ys[:len(bin_vals)][:lim]
            bin_vals = bin_vals[:lim]

            # plot the avg y values
            ax.plot(bin_vals, bin_ys, 'o--', markersize=5, alpha=1, color='red', label='binned avg. GT loss')
            ax.legend()
            plt.savefig(save_dir, bbox_inches='tight') 
            plt.close()

        if config.TTT_loss == 'reproj_2d':
            save_dir += '_reproj_2d_losses.png'
            title += rcnet_targets_name + ' 2D reproj. Loss vs GT 3D MSE'
            utils.quick_plot.simple_2d_plot(losses, save_dir=save_dir, title=title, 
                                            xlabel='2D reproj Loss', ylabel='GT 3D MSE Loss',
                                            x_lim=[0,1], y_lim=[0, 7500], alpha=0.1)
        print("saved to {}".format(save_dir))

def test_trainsets(cnet, R_cnet, config, print_summary=True):
    if not (config.TTT_loss == 'consistency'):
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

    allsets_summary = {}
    for dataset_name in config.trainsets:
        summary = {}
        for train_path, train_backbone, subset in zip(config.cnet_trainset_paths, 
                                    config.train_backbone_list, 
                                    subsets):
            if dataset_name in train_path:
                train_scale = 2.2 if train_backbone == 'hybrik' else 1.0
                if train_backbone not in summary.keys():
                    summary[train_backbone] = {}
                TTT_eval_summary = None
                cnet.load_cnets(print_str=False)
                R_cnet.load_cnets(print_str=False)
                if config.test_adapt:
                    TTT_eval_summary = eval_gt(cnet, R_cnet, config, backbone, testset, 
                                                testset_path=train_path, backbone_scale=train_scale, 
                                                test_adapt=config.test_adapt, subset=subset,
                                                use_data_file=config.TTT_from_file)
                cnet.load_cnets(print_str=False)
                eval_summary = eval_gt(cnet, R_cnet, config, testset, backbone, 
                                    testset_path=train_path, backbone_scale=train_scale, 
                                    subset=subset, use_data_file=True)
                
                summary[train_backbone]['vanilla'] = eval_summary['backbone']
                summary[train_backbone]['w/CN'] = eval_summary['corrected']
                summary[train_backbone]['at V'] = eval_summary['attempted_backbone']
                summary[train_backbone]['at C'] = eval_summary['attempted_corr']
                if TTT_eval_summary: 
                    summary[train_backbone]['+TTT'] = TTT_eval_summary['corrected']
                    summary[train_backbone]['aTTT'] = TTT_eval_summary['attempted_corr']
        allsets_summary[dataset_name] = summary
    if print_summary: print_test_summary(config, allsets_summary)
    return summary