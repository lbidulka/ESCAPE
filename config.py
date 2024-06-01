import os
from types import SimpleNamespace
import torch
import numpy as np

'''
H36M kpts:
'pelvis_extra',                                 # 0
'left_hip_extra', 'left_knee', 'left_ankle',    # 3
'right_hip_extra', 'right_knee', 'right_ankle', # 6
'spine_extra', 'neck_extra',                    # 8
'head_extra', 'headtop',                        # 10
'left_shoulder', 'left_elbow', 'left_wrist',    # 13
'right_shoulder', 'right_elbow', 'right_wrist', # 16
'''

RCNET_TARGET_NAMES = {
    'Shoulders': [11, 14],
    'Elbows': [12, 15],
    'Hips': [1, 4],
    'Knees': [2, 5],
}


def get_config():
    config = SimpleNamespace()
    config.root = 'uncertnet_poserefiner/backbones/HybrIK/'
    os.chdir(config.root)
    
    # Rand Seed
    config.seed = np.random.randint(0, 1000)
    np.random.seed(config.seed) # For test set random slice

    ######################################################################################
    ######################################################################################

    # Main paths
    config.cnet_ckpt_path = '../../ckpts/'                       # path to save/load correction net ckpts
    config.cnet_dataset_path = '/data/lbidulka/adapt_3d/'        # path to backbone predictions for datasets
    config.pose_datasets_path = '/data/lbidulka/pose_datasets/'  # path to raw datasets (images)
    
    # All Possible Tasks/experiments
    # ------
    # gen_hybrik_trainsets, gen_hybrik_testset, train_CNet, make_RCNet_trainset, train_RCNet, 
    # test, test_trainsets, 
    # plot_TTT_loss, plot_TTT_train_corr, plot_test_energies, plot_train_energies, 
    # plot_E_sep, plot_E_sep_cnet, 
    # get_inference_time
    # -----

    # What experiments to perform
    config.tasks = ['test']
    # config.tasks = ['train_CNet', 'test']
    # config.tasks = ['train_CNet', 'make_RCNet_trainset', 'train_RCNet', 'test']
    # config.tasks = ['make_RCNet_trainset', 'train_RCNet', 'test']
    # config.tasks = ['eval_raw_conditions']    # eval_varying_conditions, eval_raw_conditions
    config.tasks = ['eval_raw_conditions_miniset']    # eval_varying_conditions_miniset, eval_raw_conditions_miniset
    config.tasks = ['vcd_plot_E']

    # What backbone to use
    config.backone = 'pare' # 'hybrik', 'spin', 'pare', 'bal_mse', 'cliff',

    # What data to use
    config.trainsets = ['MPii', 'HP3D', 'h36m'] # 'MPii', 'HP3D', 'h36m'
    config.testsets = ['PW3D', ]#'HP3D',]

    # Test time adaptation and Energy-based sample selection
    config.test_adapt = True              # test with test-time adaptation?
    config.TTT_e_thresh = True            # use Energy function to select samples for adaptation? (adapt if below thresh)
    config.energy_thresh = 800            # dont correct samples with energy above this

    # CNET ---
    config.cnet_align_root = True          # make CNet align root to pelvis of inputs?
    config.zero_orientation = False        # zero out the orientation of CNet inputs?
    config.cnet_unit_inscale = False       # scale CNet/RCNet input poses to L2(pose) = 1?

    config.corr_steps = 1                  # How many correction iterations at inference?
    config.corr_step_size = 1              # base correction step size
     
    # Test time adaptation
    config.TTT_loss = 'consistency'       # 'reproj_2d' 'consistency'
    if config.TTT_loss == 'reproj_2d':
        config.test_adapt_lr = 5e-4
        config.adapt_steps = 5
        config.TTT_errscale = 1e2
    if config.TTT_loss == 'consistency':        
        config.test_adapt_lr = 5e-4
        config.adapt_steps = 2
        config.TTT_errscale = 1e2

    # Other
    config.include_target_kpt_errs = True  # report individual target kpt erors?
    config.use_cnet = True                  # use CNet at test time?

    # For varying condition ablations on PW3D
    config.vcd_variation_type = 'darkening_0.1'     # darkening_0.01, darkening_0.1, darkening_0.3, darkening_0.5
    config.num_avging_runs = 5 #100
    config.effective_adapt_idxs_pw3d = [29000, 32746, 2620, 3163, 2966,  
                 22481,  12120, 16181, 4606, 20253,
                 7031, 5804, 19750, 30116, 28069, 3352,
                 13552, 21284, 33710, 26749, 10585
                 ]
    
    # [5392, 5842,  30053, 29994,  5,  34268, 21783, 13231, 24375, 19679, 11239]
    # [5385, 31051, 2162, 4153, 8149]
    # [5385, 21007, 31360, 19037, 14454, 
    #             19006, 31051, 20327, 15184, 6817, 
    #             24949, 29978, 2162, 15166, 24974, 
    #             29541, 2018, 32225, 31357, 7822, 
    #             4153, 8149]
    # [4153, 7, 
    #             11196, 34278, 24948, 
    #             21003, 5860, 13399, 35270,
    #             4853, 8149,]
    

    #   [6650, 15171, 15175, 31353, 29539, 4153, 7, 24810, 6786,
    #    11196, 30375, 34278, 24948, 29974,
    #    28266, 21003, 5860, 29551, 24046, 13399, 35270, 14342,
    #    16975,  4853, 15136, 24964,  4878,  8149, 
    #    24997, 21775, 15186,  4152,  2004, 13295,  5388,
    #    32059, 33431, 15209,  2140, 30527, 25660,
    #    13249,  8607, 15193,  6807, 13472, 13371,
    #    24490,  6793, 25591]

    ######################################################################################
    ######################################################################################

    config = config_data(config)

    config.continue_train_CNet = False
    config.continue_train_RCNet = False
    
    # Network inputs    
    config.proximal_kpts = [1, 4, 11, 14,] # LHip, RHip, LShoulder, RShoulder
    config.distal_kpts = [3, 6, 13, 16,]  # LAnkle, RAnkle, LWrist, RWrist
    config.cnet_targets = config.distal_kpts
    # get all the entries in the dict, make a combined list
    config.rcnet_targets_name = ['Hips', 'Shoulders']
    config.rcnet_targets = []
    for name in config.rcnet_targets_name:
        config.rcnet_targets += RCNET_TARGET_NAMES[name]

    # CUDA
    config.device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    return config

def config_data(config):
    '''
    Setup data paths and other data related config
    '''
    # Data
    config.train_backbones = [config.backone]  
    config.trainsets_str = '_'.join(config.trainsets)
    config.val_sets = []
    config.test_backbones = config.train_backbones 

    config.test_eval_limit = 120_000 #120_000   # limit test samples for debug
    config.test_eval_subsets = {}
    for testset in config.testsets:
        if testset in ['PW3D', 'HP3D']:
            config.EVAL_JOINTS = [6, 5, 4, 1, 2, 3, 16, 15, 14, 11, 12, 13, 8, 10]
            config.EVAL_JOINTS.sort()
            testlens = {
                'HP3D': 2_875,
                'PW3D': 35_456, #35_515,
            }
            test_eval_subset = np.random.choice(testlens[testset], 
                                                    min(config.test_eval_limit, testlens[testset]), 
                                                    replace=False)
            config.test_eval_subsets[testset] = test_eval_subset
        else:
            raise NotImplementedError

    config.backbone_scales = {
        'spin': 1.0,
        'hybrik': 2.2,
        'pare': 1.0,
        'cliff': 1.0,
        'bal_mse': 1.0,
    }
    config.mmlab_backbones = ['spin', 'pare', 'cliff', 'bal_mse']

    # trainsets
    config.backbone_trainset_lims = {
        'hybrik': {'MPii': None, 'HP3D': None, 'h36m': None,}, # 50_000, None},
        'spin': {'MPii': None, 'HP3D': None, 'h36m': None,}, # 50_000,},
        'cliff': {'MPii': None,'HP3D': None, 'h36m': None,}, # 50_000,},
        'pare': {'MPii': None, 'HP3D': None, 'h36m': None,}, # 50_000,},
        'bal_mse': {'MPii': None,'HP3D': None, 'h36m': None,}, # 50_000,},
    }
    config.backbone_trainset_ids = {
        'hybrik': {'MPii': None, 'HP3D': None},
    }

    config.hybrIK_version = 'hrw48_wo_3dpw' # 'res34_cam', 'hrw48_wo_3dpw'
    if config.hybrIK_version == 'res34_cam':
        config.hybrik_cfg = 'configs/256x192_adam_lr1e-3-res34_smpl_3d_cam_2x_mix.yaml'
        config.ckpt = 'pretrained_w_cam.pth'
    if config.hybrIK_version == 'hrw48_wo_3dpw':
        config.hybrik_cfg = 'configs/256x192_adam_lr1e-3-hrw48_cam_2x_wo_pw3d.yaml'
        config.ckpt = 'hybrik_hrnet48_wo3dpw.pth' 

    config.trainset_info = {}
    config.cnet_trainset_paths = []
    config.cnet_trainset_scales = []
    config.train_datalims = []
    config.train_backbone_list = []
    for train_backbone in config.train_backbones:
        for trainset in config.trainsets:
            path = None
            trainlim = None
            if train_backbone == 'hybrik':
                path = '{}{}/{}_cnet_hybrik_train.npy'.format(config.cnet_dataset_path, trainset, config.hybrIK_version,)
                trainlim = config.backbone_trainset_lims[train_backbone][trainset]
            elif (train_backbone in config.mmlab_backbones) and (trainset in config.backbone_trainset_lims[train_backbone].keys()):
                path = '{}{}/mmlab_{}_cnet_train.npy'.format(config.cnet_dataset_path, trainset, train_backbone,)
                trainlim = config.backbone_trainset_lims[train_backbone][trainset]
            else:
                print('WARNING: No trainset for {} and {}'.format(train_backbone, trainset))
            if path: 
                config.cnet_trainset_paths.append(path)
                config.cnet_trainset_scales.append(config.backbone_scales[train_backbone])
                config.train_datalims.append(trainlim)
                config.train_backbone_list.append(train_backbone)

    # testsets
    config.testset_info = {}
    for testset in config.testsets:
        cnet_testset_paths = []
        cnet_testset_scales = []
        cnet_testset_backbones = []
        for test_backbone in config.test_backbones:
            path = None
            if test_backbone == 'hybrik':
                path = '{}{}/{}_cnet_hybrik_test.npy'.format(config.cnet_dataset_path, 
                                                            testset,
                                                            config.hybrIK_version,)
            elif test_backbone == 'cliff':
                if testset in ['PW3D', 'HP3D']:
                    path = '{}{}/mmlab_{}_test.npy'.format(config.cnet_dataset_path, 
                                                    testset,
                                                    test_backbone,)
                else:
                    raise NotImplementedError
            elif test_backbone in config.mmlab_backbones:
                path = '{}{}/mmlab_{}_test.npy'.format(config.cnet_dataset_path, 
                                                    testset,
                                                    test_backbone,)
            else:
                raise NotImplementedError
            if path:
                cnet_testset_paths.append(path)
                cnet_testset_scales.append(config.backbone_scales[test_backbone])
                cnet_testset_backbones.append(test_backbone)
        config.testset_info[testset] = {'paths': cnet_testset_paths,
                                        'scales': cnet_testset_scales,
                                        'backbones': cnet_testset_backbones,
                                        }

    return config