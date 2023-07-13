import os
from types import SimpleNamespace
import torch
import numpy as np

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

    config.use_multi_distal = False  # Indiv. nets for each limb + distal pred
    config.limbs = ['LA', 'RA', 'LL', 'RL'] # 'LL', 'RL', 'LA', 'RA'    limbs for multi_distal net
    # config.limbs = ['LL', 'RL']
    # config.limbs = ['LA', 'RA']
    # config.limbs = ['LL', 'RL', 'LA', 'RA']
    
    config.corr_steps = 1   # How many correction iterations at inference?
    config.corr_step_size = 0.5 # for err pred, what fraction of CNet corr to do
    config.test_adapt = True 
    config.TTT_loss = 'reproj_2d' # 'reproj_2d' 'consistency'
    config.test_adapt_lr = 1e-2 
    config.adapt_steps = 5

    # Tasks
    # config.tasks = ['make_trainset', 'make_testset', 'train', 'test'] # 'make_trainset' 'make_testset' 'train_CNet' 'train_RCNet' 'test'
    # config.tasks = ['make_trainset', 'train', 'test']
    # config.tasks = ['make_testset', 'test']
    # config.tasks = ['make_trainset']
    config.tasks = ['train_CNet', 'train_RCNet', 'test']
    # config.tasks = ['train_CNet', 'test']
    config.tasks = ['train_RCNet', 'test']
    config.tasks = ['test']
    # config.tasks = ['train']

    # Data
    config.trainsets = ['HP3D', 'MPii'] # 'MPii', 'HP3D', 'PW3D',
    config.trainsets.sort()
    config.trainsets_str = '_'.join(config.trainsets)
    config.testset = 'PW3D' # 'HP3D', 'PW3D',

    config.train_datalims = [50_000, None] # None      For debugging cnet training
    config.test_eval_limit = 1000 # 50_000    For debugging cnet testing (3DPW has 35515 test samples)
    if config.testset == 'PW3D':
        PW3D_testlen = 35_515
        config.test_eval_subset = np.random.choice(PW3D_testlen, min(config.test_eval_limit, PW3D_testlen), replace=False)
    else:
        raise NotImplementedError

    # HybrIK config
    config.hybrIK_version = 'hrw48_wo_3dpw' # 'res34_cam', 'hrw48_wo_3dpw'

    if config.hybrIK_version == 'res34_cam':
        config.hybrik_cfg = 'configs/256x192_adam_lr1e-3-res34_smpl_3d_cam_2x_mix.yaml'
        config.ckpt = 'pretrained_w_cam.pth'
    if config.hybrIK_version == 'hrw48_wo_3dpw':
        config.hybrik_cfg = 'configs/256x192_adam_lr1e-3-hrw48_cam_2x_wo_pw3d.yaml'    # w/o 3DPW
        config.ckpt = 'hybrik_hrnet48_wo3dpw.pth' 

    # cnet dataset
    config.cnet_ckpt_path = '../../ckpts/hybrIK/w_{}/'.format(config.trainsets_str)
    config.cnet_dataset_path = '/media/ExtHDD/luke_data/adapt_3d/' #3DPW

    config.cnet_trainset_paths = ['{}{}/{}_cnet_hybrik_train.npy'.format(config.cnet_dataset_path,
                                                                trainset,
                                                                config.hybrIK_version,) 
                                                                for trainset in config.trainsets]
    config.cnet_testset_path = '{}{}/{}_cnet_hybrik_test.npy'.format(config.cnet_dataset_path, 
                                                                config.testset,
                                                                config.hybrIK_version,)
    
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
    print(' --- Data: ---')
    print('Trainsets: {}'.format(config.trainsets))
    print('Testset: {}'.format(config.testset))
    print('Trainset paths: {}'.format(config.cnet_trainset_paths))
    print('Testset path: {}'.format(config.cnet_testset_path))
    print(' ----------------- \n') 
    return