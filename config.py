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
    config.corr_step_size = 1 # for err pred, what fraction of CNet corr to do
    config.test_adapt = True
    config.TTT_loss = 'consistency' # 'reproj_2d' 'consistency'
    config.TTT_from_file = True
    config.test_adapt_lr = 1e-3
    config.adapt_steps = 3
    config.TTT_errscale = 1e3

    # Tasks
    # config.tasks = ['make_trainset', 'make_testset', 'train_CNet', 'make_RCNet_trainset', 
    #                 train_RCNet, 'test', make_mmlab_test] 
    config.tasks = ['make_trainset']
    # config.tasks = ['train_CNet', 'make_RCNet_trainset', 
    #                 'train_RCNet', 'test']
    config.tasks = ['train_CNet', 'test']
    # config.tasks = ['make_RCNet_trainset']
    # config.tasks = ['train_RCNet', 'test']
    config.tasks = ['test']
    # config.tasks = ['make_mmlab_test']

    # Data
    config.trainsets = ['HP3D', 'MPii'] # 'MPii', 'HP3D', 'PW3D',
    config.trainsets.sort()
    config.trainsets_str = '_'.join(config.trainsets)
    config.testset = 'PW3D' # 'HP3D', 'PW3D',

    config.train_datalims = [50_000, None] # None      For debugging cnet training
    config.test_eval_limit = 1_000 # 50_000    For debugging cnet testing (3DPW has 35515 test samples)
    if config.testset == 'PW3D':
        config.EVAL_JOINTS = [6, 5, 4, 1, 2, 3, 16, 15, 14, 11, 12, 13, 8, 10]
        config.EVAL_JOINTS.sort()
        PW3D_testlen = 35_515
        config.test_eval_subset = np.random.choice(PW3D_testlen, min(config.test_eval_limit, PW3D_testlen), replace=False)
    else:
        raise NotImplementedError

    config.backbone = 'spin' # 'spin' 'hybrik'
    config.hybrIK_version = 'hrw48_wo_3dpw' # 'res34_cam', 'hrw48_wo_3dpw'
    if config.backbone == 'hybrik':
        config.backbone_scale = 2.2
    elif config.backbone == 'spin':
        config.backbone_scale = 0.85    # empirically found by matching metrics with reported vals
    else:
        raise NotImplementedError
    if config.hybrIK_version == 'res34_cam':
        config.hybrik_cfg = 'configs/256x192_adam_lr1e-3-res34_smpl_3d_cam_2x_mix.yaml'
        config.ckpt = 'pretrained_w_cam.pth'
    if config.hybrIK_version == 'hrw48_wo_3dpw':
        config.hybrik_cfg = 'configs/256x192_adam_lr1e-3-hrw48_cam_2x_wo_pw3d.yaml'    # w/o 3DPW
        config.ckpt = 'hybrik_hrnet48_wo3dpw.pth' 

    # cnet dataset
    config.cnet_ckpt_path = '../../ckpts/hybrIK/w_{}/'.format(config.trainsets_str)
    config.cnet_dataset_path = '/data/lbidulka/adapt_3d/' #3DPW

    config.mmlab_testset_path = '{}{}/mmlab_{}_test'.format(config.cnet_dataset_path, 
                                                                config.testset,
                                                                config.backbone,)
    if config.backbone == 'hybrik':
        config.cnet_trainset_paths = ['{}{}/{}_cnet_hybrik_train.npy'.format(config.cnet_dataset_path,
                                                                trainset,
                                                                config.hybrIK_version,) 
                                                                for trainset in config.trainsets]
        config.cnet_testset_path = '{}{}/{}_cnet_hybrik_test.npy'.format(config.cnet_dataset_path, 
                                                                    config.testset,
                                                                    config.hybrIK_version,)
    elif config.backbone == 'spin':
        config.cnet_trainset_paths = ['']
        config.cnet_testset_path = config.mmlab_testset_path + '.npy'
        if not config.TTT_from_file:
            raise NotImplementedError
    
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
    print('Testset path: {}'.format(config.cnet_testset_path))
    print(' ----------------- \n') 
    return