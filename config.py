import os
from types import SimpleNamespace
import torch

def get_config():
    config = SimpleNamespace()
    config.root = 'uncertnet_poserefiner/backbones/HybrIK/'
    os.chdir(config.root)

    # Main Settings
    config.use_cnet = True
    config.pred_errs = True  # True: predict distal joint errors, False: predict 3d-joints directly

    config.use_multi_distal = False  # Indiv. nets for each limb + distal pred
    config.limbs = ['LA', 'RA', 'LL', 'RL'] # 'LL', 'RL', 'LA', 'RA'    limbs for multi_distal net
    # config.limbs = ['LL', 'RL']
    # config.limbs = ['LA', 'RA']
    # config.limbs = ['LL', 'RL', 'LA', 'RA']
    
    config.corr_steps = 1   # How many correction iterations at inference?
    config.test_adapt = False 
    config.test_adapt_lr = 1e-3
    config.adapt_steps = 1

    config.train_datalim = None # None      For debugging cnet training

    # Tasks
    # config.tasks = ['make_trainset', 'make_testset', 'train', 'test']
    config.tasks = ['train', 'test'] # 'make_trainset' 'make_testset' 'train', 'test'
    # config.tasks = ['make_trainset', 'train', 'test']
    # config.tasks = ['make_testset', 'test']
    # config.tasks = ['make_trainset']
    config.tasks = ['test']
    # config.tasks = ['train']

    # Data
    config.trainset = 'MPii' # 'MPii', 'HP3D', 'PW3D',
    config.testset = 'PW3D' # 'HP3D', 'PW3D',

    # HybrIK config
    config.hybrIK_version = 'hrw48_wo_3dpw' # 'res34_cam', 'hrw48_wo_3dpw'

    if config.hybrIK_version == 'res34_cam':
        config.hybrik_cfg = 'configs/256x192_adam_lr1e-3-res34_smpl_3d_cam_2x_mix.yaml'
        config.ckpt = 'pretrained_w_cam.pth'
    if config.hybrIK_version == 'hrw48_wo_3dpw':
        config.hybrik_cfg = 'configs/256x192_adam_lr1e-3-hrw48_cam_2x_wo_pw3d.yaml'    # w/o 3DPW
        config.ckpt = 'hybrik_hrnet48_wo3dpw.pth' 

    # cnet dataset
    config.cnet_ckpt_path = '../../ckpts/hybrIK/w_{}/'.format(config.trainset)
    config.cnet_dataset_path = '/media/ExtHDD/luke_data/adapt_3d/' #3DPW

    config.cnet_trainset_path = '{}{}/{}_cnet_hybrik_train.npy'.format(config.cnet_dataset_path, 
                                                                config.trainset,
                                                                config.hybrIK_version,)
    config.cnet_testset_path = '{}{}/{}_cnet_hybrik_test.npy'.format(config.cnet_dataset_path, 
                                                                config.testset,
                                                                config.hybrIK_version,)
    
    # CUDA
    config.device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    print_useful_configs(config)
    return config

def print_useful_configs(config):
    print('\n ----- CONFIG: -----')
    print(' -------------------')
    print('hybrIK_version: {}'.format(config.hybrIK_version))
    print('Tasks: {}'.format(config.tasks))
    print(' --- CNet: ---')
    print('Use CNet: {}'.format(config.use_cnet))
    print('Corr Steps: {}'.format(config.corr_steps))
    print('Test Adapt: {}'.format(config.test_adapt))
    print('Test Adapt LR: {}'.format(config.test_adapt_lr))
    print('Adapt Steps: {}'.format(config.adapt_steps)) 
    print(' --- Data: ---')
    print('Trainset: {}'.format(config.trainset))
    print('Testset: {}'.format(config.testset))
    print('Trainset path: {}'.format(config.cnet_trainset_path))
    print('Testset path: {}'.format(config.cnet_testset_path))
    print(' ----------------- \n') 
    return