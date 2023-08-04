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

def get_config():
    config = SimpleNamespace()
    config.root = 'uncertnet_poserefiner/backbones/HybrIK/'
    os.chdir(config.root)
    
    # Rand Seed
    config.seed = np.random.randint(0, 1000)
    np.random.seed(config.seed) # For test set random slice

    # Main Settings
    config.print_config = True
    config.use_cnet = True
    config.pred_errs = True  # True: predict distal joint errors, False: predict 3d-joints directly
    
    config.proximal_kpts = [1, 4, 11, 14,] # LHip, RHip, LShoulder, RShoulder
    config.distal_kpts = [3, 6, 13, 16,]  # LAnkle, RAnkle, LWrist, RWrist
    config.cnet_targets = config.distal_kpts
    config.rcnet_targets = [2, 5]   # LKnee, RKnee

    config.use_multi_distal = False  # Indiv. nets for each limb + distal pred
    config.limbs = ['LA', 'RA', 'LL', 'RL'] # 'LL', 'RL', 'LA', 'RA'    limbs for multi_distal net
    
    config.split_corr_dim_trick = False  # correct z with trained CNet, correct x/y with tuned CNet

    config.corr_steps = 1   # How many correction iterations at inference?
    config.corr_step_size = 0.5 # for err pred, what fraction of CNet corr to do
    config.test_adapt = False
    config.TTT_loss = 'consistency' # 'reproj_2d' 'consistency'
    config.TTT_from_file = True
    config.test_adapt_lr = 1e-4
    config.adapt_steps = 1
    config.TTT_errscale = 1e2

    # Tasks
    # config.tasks = ['make_trainsets', 'make_testset', 'train_CNet', 'make_RCNet_trainset', 
    #                 train_RCNet, 'test', make_mmlab_test] 
    # config.tasks = ['make_trainsets']
    config.tasks = ['train_CNet', 'make_RCNet_trainset', 
                    'train_RCNet', 'test']
    # config.tasks = ['train_CNet', 'test']
    # config.tasks = ['make_RCNet_trainset', 'train_RCNet']
    # config.tasks = ['train_RCNet', 'test']
    # config.tasks = ['test']
    # config.tasks = ['make_mmlab_test']

    # Data
    config.trainsets = ['MPii', 'HP3D'] # 'MPii', 'HP3D', 'PW3D',
    config.trainsets.sort()
    config.trainsets_str = '_'.join(config.trainsets)
    config.testset = 'PW3D' # 'HP3D', 'PW3D',

    config.test_eval_limit = 50_000 # 50_000    For debugging cnet testing (3DPW has 35515 test samples)
    if config.testset == 'PW3D':
        config.EVAL_JOINTS = [6, 5, 4, 1, 2, 3, 16, 15, 14, 11, 12, 13, 8, 10]
        config.EVAL_JOINTS.sort()
        PW3D_testlen = 35_515
        config.test_eval_subset = np.random.choice(PW3D_testlen, min(config.test_eval_limit, PW3D_testlen), replace=False)
    else:
        raise NotImplementedError

    config.train_backbones = ['cliff', 'pare', 'spin', 'hybrik'] # 'spin', 'hybrik', 'cliff', 'pare'
    # config.test_backbones = ['hybrik', 'spin'] # 'spin', 'hybrik', 'pare', 'cliff'
    # config.test_backbones = ['hybrik', 'spin'] 
    # config.test_backbones = ['spin'] 
    config.test_backbones = ['hybrik', 'spin', 'pare', 'cliff']
    config.hybrIK_version = 'hrw48_wo_3dpw' # 'res34_cam', 'hrw48_wo_3dpw'

    config.backbone_scales = {
        'spin': 1.0, #0.85,
        'hybrik': 2.2,
        'pare': 1.0,
        'cliff': 1.0,
    }
    config.mmlab_backbones = ['spin', 'pare', 'cliff']

    if config.hybrIK_version == 'res34_cam':
        config.hybrik_cfg = 'configs/256x192_adam_lr1e-3-res34_smpl_3d_cam_2x_mix.yaml'
        config.ckpt = 'pretrained_w_cam.pth'
    if config.hybrIK_version == 'hrw48_wo_3dpw':
        config.hybrik_cfg = 'configs/256x192_adam_lr1e-3-hrw48_cam_2x_wo_pw3d.yaml'    # w/o 3DPW
        config.ckpt = 'hybrik_hrnet48_wo3dpw.pth' 

    # cnet dataset
    config.cnet_ckpt_path = '../../ckpts/' #hybrIK/w_{}/'.format(config.trainsets_str)
    config.cnet_dataset_path = '/data/lbidulka/adapt_3d/' #3DPW

    # trainsets
    config.cnet_trainset_paths = []
    config.cnet_trainset_scales = []
    config.train_datalims = []
    for train_backbone in config.train_backbones:
        for trainset in config.trainsets:
            path = None
            trainlim = None
            if train_backbone == 'hybrik':
                if trainset in ['HP3D', 'MPii']:
                    path = '{}{}/{}_cnet_hybrik_train.npy'.format(config.cnet_dataset_path, trainset, config.hybrIK_version,)
                if trainset == 'HP3D':
                    trainlim = 50_000 #50_000   # hyperparam
            elif train_backbone == 'spin':
                if trainset in ['HP3D', 'MPii']:
                    path = '{}{}/mmlab_{}_train.npy'.format(config.cnet_dataset_path, trainset, train_backbone,)
                if trainset == 'HP3D':
                    trainlim = 50_000 #50_000   # hyperparam
            elif train_backbone == 'cliff':
                if trainset in ['HP3D', 'MPii']:
                    path = '{}{}/mmlab_{}_train.npy'.format(config.cnet_dataset_path, trainset, train_backbone,)
                if trainset == 'HP3D':
                    trainlim = 50_000 #50_000   # hyperparam
            elif train_backbone == 'pare':
                if trainset in ['HP3D', 'MPii']:
                    path = '{}{}/mmlab_{}_train.npy'.format(config.cnet_dataset_path, trainset, train_backbone,)
                if trainset == 'HP3D':
                    trainlim = 50_000 #50_000   # hyperparam
            else:
                raise NotImplementedError
            if path: 
                config.cnet_trainset_paths.append(path)
                config.cnet_trainset_scales.append(config.backbone_scales[train_backbone])
                config.train_datalims.append(trainlim)

    # testsets
    config.cnet_testset_paths = []
    config.cnet_testset_scales = []
    config.cnet_testset_backbones = []
    for test_backbone in config.test_backbones:
        path = None
        if test_backbone == 'hybrik':
            path = '{}{}/{}_cnet_hybrik_test.npy'.format(config.cnet_dataset_path, 
                                                                        config.testset,
                                                                        config.hybrIK_version,)
        elif test_backbone in config.mmlab_backbones:
            path = '{}{}/mmlab_{}_test.npy'.format(config.cnet_dataset_path, 
                                                config.testset,
                                                test_backbone,)
            if not config.TTT_from_file:
                raise NotImplementedError
        else:
            raise NotImplementedError
        if path:
            config.cnet_testset_paths.append(path)
            config.cnet_testset_scales.append(config.backbone_scales[test_backbone])
            config.cnet_testset_backbones.append(test_backbone)
    
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
    print('Testset: {}'.format(config.testset))
    print('Trainset paths: {}'.format(config.cnet_trainset_paths))
    print('Testset paths: {}'.format(config.cnet_testset_paths))
    print(' ----------------- \n') 
    return