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

RCNET_TARGET_NAMES_OTHER = {
    'Shoulders': [11, 14],
    'Elbows': [12, 15],
    'Hips': [1, 4],
    'Knees': [2, 5],
}

RCNET_TARGET_NAMES_AGORA = {
    'Shoulders': [8, 11],
    'Elbows': [9, 12],
    'Hips': [0, 3],
    'Knees': [1, 4],
}


def get_config():
    config = SimpleNamespace()
    config.root = 'uncertnet_poserefiner/backbones/HybrIK/'
    os.chdir(config.root)
    
    # Rand Seed
    config.seed = np.random.randint(0, 1000)
    np.random.seed(config.seed) # For test set random slice

    # Tasks
    # config.tasks = ['make_testset']
    # config.tasks = ['make_trainsets', 'make_testset', 
    #                 'train_CNet', 'make_RCNet_trainset', 'train_RCNet',
    #                 'test', 'plot_TTT_loss'] 
    # config.tasks = ['train_CNet', 'make_RCNet_trainset', 
    #                 'train_RCNet', 'test', 'plot_TTT_loss']
    # config.tasks = ['make_kpt_amass']
    # config.tasks = ['test', 'train_CNet', 'test']
    # config.tasks = ['pretrain_CNet']
    # config.tasks = ['pretrain_RCNet']
    
    # config.tasks = ['make_RCNet_trainset', 'train_RCNet', 'test', 'plot_TTT_loss']
    config.tasks = ['train_CNet', 'test']

    # config.tasks = ['test']
    # config.tasks = ['test', 'plot_test_energies']
    # config.tasks = ['plot_test_energies']
    # config.tasks = ['plot_train_energies']

    # config.tasks = ['make_RCNet_trainset', 'train_RCNet']#, 'test',]# 'plot_TTT_loss', 'plot_TTT_train_corr']
    # config.tasks = ['make_RCNet_trainset', 'train_RCNet', 'test', 'plot_TTT_loss',] # 'plot_TTT_train_corr']
    config.tasks = ['test', 'plot_TTT_loss',]

    # config.tasks = ['optuna_CNet', 'optuna_TTT']

    # config.tasks = ['cotrain', 'test',]# 'plot_TTT_loss']

    # config.tasks = ['train_CNet', 'export_agora']

    # config.tasks = ['test', 'plot_TTT_loss']
    # config.tasks = ['export_agora']
    # config.tasks = ['plot_TTT_loss']
    config.tasks = ['get_inference_time']

    # Main Settings
    config.optuna_num_trials = 1
    config.print_config = False
    config.err_binned_results = True
    config.include_target_kpt_errs = True
    config.use_cnet = True
    config.use_features = False  # use feature maps as input to CNet?

    # ENERGY ---
    config.use_cnet_energy = False     # use energy function to select OOD samples?
    config.energy_lower_thresh = True    # don't correct samples too high energy?
    config.energy_thresh = 800    #450    # dont correct samples with energy above this
    config.energy_scaled_corr = False  # scale the correction step size by energy?

    config.reverse_thresh = False      # reverse the energy thresholding? (correct if above thresh)
    # ---

    config.pred_errs = True  # True: predict distal joint errors, False: predict 3d-joints directly
    
    config.cnet_dont_corr_dims = [] #[0,1] #[0,2]  # which dims to not correct with CNet at all
    config.cnet_dont_Escale_dims = [] #[0,1] #[0,2]  # which dims to not scale steps with E
    # config.split_corr_dim_trick = False
    config.split_corr_dims = []  # [0,2] # which dims to not correct with TTT tuned CNet

    config.corr_steps = 1   # How many correction iterations at inference?
    config.corr_step_size = 1 # base correction step size

    # Fancy Training Options
    config.PA_mse_loss = True           # use PA-MSE loss?
    config.zero_orientation = False     # zero out the orientation of CNet inputs?
    config.cotrain = True if 'cotrain' in config.tasks else False
    config.pretrain_AMASS = False        # use pretrain networks on AMASS?
    config.AMASS_scale = 0.4            # scale AMASS data by this factor when getting kpts
    config.loss_pose_scaling = False
    config.only_hard_samples = False     # only train on samples with high error?
    config.hard_sample_thresh = 3000     # threshold for hard samples
    config.sample_weighting = False
    config.continue_train_CNet = False
    config.continue_train_RCNet = False
     
    # TTT
    config.test_adapt = True
    config.TTT_e_thresh = True      # only apply TTT to samples with samples below energy thresh?
    config.TTT_loss = 'consistency' # 'reproj_2d' 'consistency'
    config.TTT_from_file = True
    if config.TTT_loss == 'reproj_2d':
        config.test_adapt_lr = 1e-3
        config.adapt_steps = 5
        config.TTT_errscale = 1e2
    if config.TTT_loss == 'consistency':
        if config.pretrain_AMASS:
            config.test_adapt_lr = 2e-4 # 5e-4
            config.adapt_steps = 2 
        else:
            # config.test_adapt_lr = 5e-4 # 5e-4
            # config.adapt_steps = 3
            
            # CLIFF
            config.test_adapt_lr = 5e-4 # 5e-4
            config.adapt_steps = 2 if config.TTT_e_thresh else 1

            # BEDLAM-CLIFF
            # config.test_adapt_lr = 1e-4 # 5e-4
            # config.adapt_steps = 1
        config.TTT_errscale = 1e2

    # DATA
    # config.train_backbones = ['hybrik', 'spin', 'pare', 'cliff', 'bal_mse'] # 'spin', 'hybrik', 'cliff', 'pare', 'bal_mse'
    # config.train_backbones = ['cliff',]
    config.train_backbones = ['hybrik',]
    # config.train_backbones = ['bedlam-cliff', 'cliff']

    config.trainsets = ['MPii', 'HP3D',] # 'MPii', 'HP3D', 
    # config.trainsets = ['MPii', 'HP3D', 'RICH'] # 'MPii', 'HP3D', 
    # config.trainsets = ['AGORA', 'MPii', 'HP3D',]
    # config.trainsets = ['AGORA', 'BEDLAM',]# 'MPii', 'HP3D',] 
    # config.trainsets.sort()
    config.trainsets_str = '_'.join(config.trainsets)

    # config.val_sets = {'MPii': 'cliff', 
    #                    'HP3D': 'cliff',}
    config.val_sets = []
    
    # config.test_backbones = ['hybrik', 'spin', 'pare', 'cliff', 'bal_mse']
    # config.test_backbones = ['cliff',]
    config.test_backbones = ['hybrik',]
    # config.test_backbones = ['bedlam-cliff',]

    # config.testset = 'PW3D'
    # config.testset = 'RICH' 
    # config.testset = 'AGORA'
    # config.testset = 'HP3D'

    config.testsets = ['PW3D', 'HP3D', ]
    config.testsets = ['PW3D',]
    # config.testsets = ['PW3D', 'RICH']

    if 'export_agora' in config.tasks:
        config.testset = 'AGORA'

    # bedlam-cliff is in 14 kpt format, so it will mess up eval of other backbones if tried together
    if 'PW3D' in config.testsets and 'bedlam-cliff' in config.test_backbones:
        assert len(config.test_backbones) == 1


    config.test_eval_limit = 120_000 # 50_000    For debugging cnet testing (3DPW has 35515 test samples)
    config.test_eval_subsets = {}
    for testset in config.testsets:
        if testset in ['PW3D', 'RICH', 'AGORA', 'HP3D']:
            if testset == 'AGORA' and 'bedlam-cliff' not in config.test_backbones: # or (config.testset == 'PW3D' and 'bedlam-cliff' in config.test_backbones):
                config.EVAL_JOINTS = [i for i in range(14)]
            else:
                config.EVAL_JOINTS = [6, 5, 4, 1, 2, 3, 16, 15, 14, 11, 12, 13, 8, 10]
            config.EVAL_JOINTS.sort()
            testlens = {
                'HP3D': 2_875,
                'PW3D': 35_456, #35_515,
                'RICH': 21_248,
                'AGORA': 63_552, #5286,    # AGORA: manually set
            }
            # DEBUG ----
            if config.use_features:
                # set the testset to only choose from the samples with extracted feature maps

                # first, crawl the feature map directory to get the list of samples with feature maps
                feature_map_dir = f'/data/lbidulka/adapt_3d/{testset}/feature_maps/' + 'mmlab_cliff_test'
                feature_map_files = os.listdir(feature_map_dir)
                feature_map_ids = [int(f.split('.')[0]) for f in feature_map_files]
                                
                # next, 
                test_eval_subset = np.random.choice(feature_map_ids, 
                                                    min(config.test_eval_limit, len(feature_map_ids)), 
                                                    replace=False)
            # ----------
            else:
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
        'bedlam-cliff': 1.0,
    }
    config.mmlab_backbones = ['spin', 'pare', 'cliff', 'bal_mse']
    config.bedlam_backbones = ['bedlam-cliff',]

    # Main base baths
    config.cnet_ckpt_path = '../../ckpts/' #hybrIK/w_{}/'.format(config.trainsets_str)
    config.cnet_dataset_path = '/data/lbidulka/adapt_3d/'
    config.pose_datasets_path = '/data/lbidulka/pose_datasets/'
    config.optuna_log_path = '../../outputs/optuna/'

    config.amass_path = config.pose_datasets_path + 'AMASS/processed_AMASS.npz'
    config.amass_kpts_path = config.pose_datasets_path + 'AMASS/processed_AMASS_kpts.npz'

    # trainsets
    config.backbone_trainset_lims = {
        'hybrik': {'MPii': None, 'HP3D': None}, # 50_000, None},
        'spin': {'MPii': None, 'HP3D': None}, # 50_000,},
        'cliff': {'MPii': None,'HP3D': None, 'RICH': None,}, # 50_000,},
        'bedlam-cliff': {'AGORA': None, 'BEDLAM': None,},
        'pare': {'MPii': None, 'HP3D': None}, # 50_000,},
        'bal_mse': {'MPii': None,'HP3D': None}, # 50_000,},
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
            elif (train_backbone in config.bedlam_backbones) and (trainset in config.backbone_trainset_lims[train_backbone].keys()):
                path = '{}{}/bedlam_{}_cnet_train.npy'.format(config.cnet_dataset_path, trainset, train_backbone,)
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
                elif testset in ['RICH', 'AGORA']:
                    path = '{}{}/bedlam_{}_test.npy'.format(config.cnet_dataset_path, 
                                                    testset,
                                                    test_backbone,)
                else:
                    raise NotImplementedError
            elif test_backbone in config.mmlab_backbones:
                path = '{}{}/mmlab_{}_test.npy'.format(config.cnet_dataset_path, 
                                                    testset,
                                                    test_backbone,)
                if not config.TTT_from_file:
                    raise NotImplementedError
            elif test_backbone in config.bedlam_backbones:
                path = '{}{}/bedlam_{}_test.npy'.format(config.cnet_dataset_path, 
                                                    testset,
                                                    test_backbone,)
                if not config.TTT_from_file:
                    raise NotImplementedError
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
    
    # Network inputs
    config.use_multi_distal = False  # Indiv. nets for each limb + distal pred
    config.limbs = ['LA', 'RA', 'LL', 'RL'] # 'LL', 'RL', 'LA', 'RA'    limbs for multi_distal net
    
    if 'AGORA' in config.testsets  and 'bedlam-cliff' not in config.test_backbones:
        config.proximal_kpts = [0, 3, 8, 11]    # adjusted due to AGORA data being in 14-joint format already
        config.distal_kpts = [2, 5, 10, 13]
        RCNET_TARGET_NAMES = RCNET_TARGET_NAMES_AGORA
    else:
        config.proximal_kpts = [1, 4, 11, 14,] # LHip, RHip, LShoulder, RShoulder
        config.distal_kpts = [3, 6, 13, 16,]  # LAnkle, RAnkle, LWrist, RWrist
        RCNET_TARGET_NAMES = RCNET_TARGET_NAMES_OTHER
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
    if config.print_config: print_useful_configs(config)
    return config

def print_useful_configs(config):
    print('\n ----- CONFIG: -----')
    print(' -------------------')
    print('hybrIK_version: {}'.format(config.hybrIK_version))
    print('Tasks: {}'.format(config.tasks))
    print(' --- CNet: ---')
    print('Use CNet: {}'.format(config.use_cnet))
    print('Corr Steps: {}'.format(config.corr_steps))
    print('Corr Step Size: {}'.format(config.corr_step_size))
    print('Test Adapt: {}'.format(config.test_adapt))
    print('Test Adapt LR: {}'.format(config.test_adapt_lr))
    print('Adapt Steps: {}'.format(config.adapt_steps)) 
    print('TTT Loss: {}'.format(config.TTT_loss))
    print('Split Corr Dim Trick: {}'.format(config.split_corr_dim_trick))
    print(' --- Data: ---')
    print('Trainsets: {}'.format(config.trainsets))
    print('Testsets: {}'.format(config.testsets))
    # print('Trainset paths: {}'.format(config.cnet_trainset_paths))
    # print('Testset paths: {}'.format(config.cnet_testset_paths))
    print(' ----------------- \n') 
    return