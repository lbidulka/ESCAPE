
# path_root = Path(__file__)#.parents[3]
# sys.path.append(str(path_root))
# sys.path.append(str(path_root)+ 'backbones/HybrIK/hybrik/')

# print("\n", os.getcwd(), "\n", sys.path, "\n")

from torchvision import transforms as T
det_transform = T.Compose([T.ToTensor()])

from datasets.hybrik import make_hybrik_pred_dataset
from cnet.multi_distal import multi_distal
from cnet.full_body import adapt_net
from core.cnet_eval import eval_gt
from config import get_config

from utils.output_reporting import plot_TTT_loss, test_trainsets, plot_energies, print_test_summary


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
    if config.use_multi_distal:
        cnet = multi_distal(config)
        R_cnet = None # TODO: MULTI-DISTAL R-CNET
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
        else:
            raise NotImplementedError
    

if __name__ == "__main__":    
    config = get_config()
    main_worker(config)

