import torch

from hybrik.models import builder
from hybrik.utils.config import update_config

from core.cnet_data import create_cnet_dataset_w_HybrIK

from .MPII import MPII
from .PW3D import PW3D
from .HP3D import HP3D


def load_pretrained_hybrik(config, hybrik_cfg,):
    ckpt = config.ckpt
    hybrik_model = builder.build_sppe(hybrik_cfg.MODEL)
    
    print(f'\nLoading HybrIK model from {ckpt}...\n')
    save_dict = torch.load(ckpt, map_location='cpu')
    if type(save_dict) == dict:
        model_dict = save_dict['model']
        hybrik_model.load_state_dict(model_dict)
    else:
        hybrik_model.load_state_dict(save_dict, strict=False)

    return hybrik_model

def load_hybrik(config):
    ''' Load the backbone 3D pose estimation model '''
    model_cfg = update_config(config.hybrik_cfg) 
    print('USING HYBRIK VER: {}'.format(config.hybrIK_version))
    model = load_pretrained_hybrik(config, model_cfg)
    model = model.to('cpu')
    return model, model_cfg

def get_datasets(backbone_cfg, config):
    trainsets = []
    testsets = []
    if any([(task in config.tasks) for task in 
            ['make_trainsets', 'make_trainsets']]):
        for dataset in config.trainsets:
            if dataset == 'PW3D':
                trainset = PW3D(
                    cfg=backbone_cfg,
                    ann_file='3DPW_train_new_fresh.json',
                    train=False,
                    root='/media/ExtHDD/Mohsen_data/3DPW')
            elif dataset == 'MPii':
                trainset = MPII(
                    cfg=backbone_cfg,
                    annot_dir='/media/ExtHDD/Mohsen_data/mpii_human_pose/mpii_cliffGT.npz',
                    image_dir='/media/ExtHDD/Mohsen_data/mpii_human_pose/',)
            elif dataset == 'HP3D':
                trainset = HP3D(
                    cfg=backbone_cfg,
                    ann_file='train_v2',   # dumb adjustment...
                    train=False,
                    root='/media/ExtHDD/luke_data/HP3D')
            else:
                raise NotImplementedError
            trainsets.append(trainset)
    elif 'make_testset' in config.tasks:
        for dataset in config.testsets:
            if dataset == 'PW3D':
                testset = PW3D(
                    cfg=backbone_cfg,
                    ann_file='3DPW_test_new_fresh.json',
                    train=False,
                    root='/media/ExtHDD/Mohsen_data/3DPW')
            elif dataset == 'HP3D':
                testset = HP3D(
                    cfg=backbone_cfg,
                    ann_file='test',
                    train=False,
                    root='/data/lbidulka/pose_datasets' + '/mpi_inf_3dhp',
                )
            else:
                raise NotImplementedError
            testsets.append(testset)
    else: 
        testsets = [None]
    return trainsets, testsets

def make_hybrik_pred_dataset(config, task):
    '''
    Creates a dataset of HybrIK predictions

    args:
        hybrik: HybrIK model
        datasets: list of datasets to make predictions on
        config: config object
        task: str, 'train' or 'test'
    '''
    hybrik, backbone_cfg = load_hybrik(config)
    trainsets, testsets = get_datasets(backbone_cfg, config)

    if task == 'train':
        dataset_names = config.trainsets
        datasets = trainsets
    elif task == 'test':
        dataset_names = config.testsets
        datasets = testsets
    with torch.no_grad():
        for i, dataset in enumerate(datasets):
            print(f'##### Creating Cnet {dataset} HybrIK {task}set #####')
            create_cnet_dataset_w_HybrIK(hybrik, config, dataset, 
                                         dataset=dataset_names[i], task=task,)