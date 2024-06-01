
# path_root = Path(__file__)#.parents[3]
# sys.path.append(str(path_root))
# sys.path.append(str(path_root)+ 'backbones/HybrIK/hybrik/')

# print("\n", os.getcwd(), "\n", sys.path, "\n")

import numpy as np

from datasets.hybrik import make_hybrik_pred_dataset
from cnet.full_body import adapt_net
from core.cnet_eval import eval_gt
from config import get_config

import utils.output_reporting as output_reporting
from utils.output_reporting import plot_TTT_loss, test_trainsets, plot_energies, print_test_summary
from utils.mmlab_varying_conditions import vcd_samples_idxs


def test(cnet, R_cnet, config, print_summary=True):
    # Load CNet & R-CNet, then test
    cnet.load_cnets()
    if config.test_adapt and (config.TTT_loss == 'consistency'): R_cnet.load_cnets()

    allsets_summary = {}
    for testset_name, info in config.testset_info.items():
        summary = {}
        for test_path, test_scale, test_backbone in zip(info['paths'], info['scales'], info['backbones']):
            summary[test_backbone] = {}
            TTT_eval_summary = None
            cnet.load_cnets(print_str=False)
            if config.test_adapt:
                if config.test_adapt and (config.TTT_loss == 'consistency'): R_cnet.load_cnets(print_str=False)
                TTT_eval_summary = eval_gt(cnet, R_cnet, config, 
                                        testset_path=test_path, 
                                        backbone_scale=test_scale, 
                                        test_adapt=True, subset=config.test_eval_subsets[testset_name],)
            cnet.load_cnets(print_str=False)
            eval_summary = eval_gt(cnet, R_cnet, config, 
                                testset_path=test_path, 
                                backbone_scale=test_scale, 
                                subset=config.test_eval_subsets[testset_name],)
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
    cnet = adapt_net(config, target_kpts=config.cnet_targets,
                    in_kpts=config.EVAL_JOINTS)
    R_cnet = adapt_net(config, target_kpts=config.rcnet_targets,
                    R=True,
                    in_kpts=config.EVAL_JOINTS)
    return cnet, R_cnet

def main_worker(config): 
    cnet, R_cnet = setup_adapt_nets(config)
    for task in config.tasks:
        if task == 'gen_hybrik_trainsets':
            make_hybrik_pred_dataset(config, 'train')
        elif task == 'gen_hybrik_testset':
            make_hybrik_pred_dataset(config, 'test')
        
        elif task == 'train_CNet':
            cnet.train()
        elif task == 'make_RCNet_trainset':
            cnet.write_train_preds()
        elif task == 'train_RCNet':
            R_cnet.train()

        elif task == 'test':
            test(cnet, R_cnet, config)
        elif task == 'test_trainsets':
            test_trainsets(cnet, R_cnet, config,)

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

        elif task == 'plot_E_sep':
            from utils.output_reporting import plot_E_sep
            for testset in config.testsets:
                plot_E_sep(config, task='test', dataset=testset)
            for trainset in config.trainsets:
                plot_E_sep(config, task='train', dataset=trainset)
        elif task == 'plot_E_sep_cnet':
            from utils.output_reporting import plot_E_sep
            for testset in config.testsets:
                plot_E_sep(config, task='test', dataset=testset, cnet=True)
            for trainset in config.trainsets:
                plot_E_sep(config, task='train', dataset=trainset, cnet=True)

        elif task == 'get_inference_time':
            from utils.inference_timing import get_inference_time
            get_inference_time(config, cnet, R_cnet)

        elif task == 'eval_varying_conditions':
            # Make sure we adapt on all samples
            config.energy_thresh = 800000
            # adjust testset path to get the varying conditions subset samples
            for testset in config.testsets:
                config.testset_info[testset]['paths'] = [config.testset_info[testset]['paths'][0].replace('test', 'test_varying_conditions')]
                config.test_eval_subsets[testset] = [i for i in range(len(config.effective_adapt_idxs_pw3d))]
            test(cnet, R_cnet, config)
        elif task == 'eval_raw_conditions':
            # load up the testset and set the subset to be the config.effective_adapt_idxs_pw3d
            if (len(config.testsets) > 1) or ((len(config.testsets) == 1) and (config.testsets[0] != 'PW3D')):
                raise NotImplementedError
            # Make sure we adapt on all samples
            config.energy_thresh = 800000
            # Adjust subset to be the desired indices
            test_data_ = np.load(config.testset_info['PW3D']['paths'][0])
            img_idxs = test_data_[2,:,0]
            config.test_eval_subsets['PW3D'] = np.array([np.where(img_idxs == idx)[0] for idx in config.effective_adapt_idxs_pw3d]).reshape(-1)
            test(cnet, R_cnet, config)
        elif task == 'eval_varying_conditions_miniset':
            # adjust testset path to get the varying conditions miniset
            for testset in config.testsets:
                vcd_title = f'test_varying_conditions_miniset_{config.vcd_variation_type}'
                config.testset_info[testset]['paths'] = [config.testset_info[testset]['paths'][0].replace('test', vcd_title)]
                config.test_eval_subsets[testset] = [i for i in range(len(config.effective_adapt_idxs_pw3d))]
            # set the varying conditions subset samples
            test_data_ = np.load(config.testset_info['PW3D']['paths'][0])
            img_idxs = test_data_[2,:,0]
            config.test_eval_subsets['PW3D'] = np.array([np.where(img_idxs == idx)[0] for idx in vcd_samples_idxs]).reshape(-1)
            test(cnet, R_cnet, config)
        elif task == 'eval_raw_conditions_miniset':
            # set the varying conditions subset samples
            test_data_ = np.load(config.testset_info['PW3D']['paths'][0])
            img_idxs = test_data_[2,:,0]
            config.test_eval_subsets['PW3D'] = np.array([np.where(img_idxs == idx)[0] for idx in vcd_samples_idxs]).reshape(-1)
            test(cnet, R_cnet, config)
        elif task == 'vcd_plot_E':
            output_reporting.vcd_plot_E_dists(config)
        else:
            raise NotImplementedError
    

if __name__ == "__main__":    
    config = get_config()
    main_worker(config)

