import argparse
import os
import os.path as osp

import mmcv
import torch
from mmcv import DictAction
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (
    get_dist_info,
    init_dist,
    load_checkpoint,
    wrap_fp16_model,
)

from mmhuman3d.apis import multi_gpu_test, single_gpu_test
from mmhuman3d.data.datasets import build_dataloader, build_dataset
from mmhuman3d.models.architectures.builder import build_architecture

import numpy as np
from types import SimpleNamespace


def get_config(args, ):
    config = SimpleNamespace()
    config.root = 'uncertnet_poserefiner/backbones/mmhuman3d/'
    os.chdir(config.root)
    config.args = args

    # Tasks
    # config.tasks = ['gen_preds', 'gen_cnet_data', 'eval_corrected']
    config.tasks = ['gen_preds', 'gen_cnet_data',]
    # config.tasks = ['gen_preds']
    config.tasks = ['gen_cnet_data']
    # config.tasks = ['eval_corrected']
    config.save_preds = True    # Save the generated preds?

    # Backbone generation settings
    config.backbone = 'spin'  # 'hybrik' 'spin'
    config.dataset = 'PW3D' # 'PW3D' 'coco' 'mpii' 'hp3d'
    config.subset_len = 50_000 # pwd3d: 35k, mpii: 14810, coco2017: 40055
    set_model_and_data_config(config)

    # Paths
    config.cnet_data_path = '/data/lbidulka/adapt_3d/' #'/media/ExtHDD/luke_data/adapt_3d/'
    config.cnet_dataset_path = '{}{}/mmlab_{}_{}'.format(config.cnet_data_path, 
                                                             config.dataset, 
                                                             config.backbone, 
                                                             'test' if (config.dataset == 'PW3D') else 'train') 
    return config

def set_model_and_data_config(config):
    # HybrIK
    if config.backbone == 'hybrik':
        config.args.config = 'configs/hybrik/resnet34_hybrik_eval_train.py'
        config.args.checkpoint = 'data/pretrained/pretrain_hybrik.pth'
        config.args.work_dir = 'work_dirs/hybrik'
        if config.dataset == 'PW3D':
            config.eval_data = 'test'
        if config.dataset == 'coco':    # train2017
            config.eval_data = 'coco_eval'
        if config.dataset == 'hp3d':
            config.eval_data = 'hp3d_eval'
    # Spin
    elif config.backbone == 'spin':
        config.args.config = 'configs/spin/resnet50_spin_eval_train.py'
        config.args.checkpoint = 'data/pretrained/pretrain_spin.pth'
        config.args.work_dir = 'work_dirs/spin'
        if config.dataset == 'PW3D':
            config.eval_data = 'test'
        if config.dataset == 'mpii':
            config.eval_data = 'mpii_eval'
    # eval_data = 'mpii_eval'        
    return config

def parse_args():
    parser = argparse.ArgumentParser(description='mmhuman3d test model')
    parser.add_argument('--config', help='test config file path',
                        default='configs/hybrik/resnet34_hybrik_eval_train.py')
    parser.add_argument(
        '--work-dir', help='the dir to save evaluation results', 
        default='work_dirs/hybrik')
    parser.add_argument('--checkpoint', help='checkpoint file',
                        default='data/pretrained/pretrain_hybrik.pth')
    parser.add_argument('--out', help='output result file')
    parser.add_argument(
        '--metrics',
        type=str,
        nargs='+',
        default=['pa-mpjpe', 'mpjpe'],
        help='evaluation metric, which depends on the dataset,'
        ' e.g., "pa-mpjpe" for H36M')
    parser.add_argument(
        '--gpu_collect',
        action='store_true',
        help='whether to use gpu to collect results')
    parser.add_argument('--tmpdir', help='tmp dir for writing some results')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file.')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        default={},
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument(
        '--device',
        choices=['cpu', 'cuda'],
        default='cuda',
        help='device used for testing')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args

def gen_preds(config, cfg, dataset, data_loader):
    args = config.args

    # build the model and load checkpoint
    model = build_architecture(cfg.model)
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    load_checkpoint(model, args.checkpoint, map_location='cpu')

    # Get preds
    if args.device == 'cpu':
        model = model.cpu()
    else:
        model = MMDataParallel(model, device_ids=[0])
    outputs = single_gpu_test(model, data_loader)

    rank, _ = get_dist_info()
    eval_cfg = cfg.get('evaluation', args.eval_options)
    eval_cfg.update(dict(metric=args.metrics))
    if rank == 0:
        mmcv.mkdir_or_exist(osp.abspath(args.work_dir))
        results = dataset.dataset.get_results_and_human_gts(outputs, args.work_dir, config.save_preds)
        # results = dataset.dataset.evaluate(outputs, args.work_dir, **eval_cfg)
    
    return results

def setup_cfg_and_data(config, eval_data):
    args = config.args
    cfg = mmcv.Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.data.test.test_mode = True

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # build the dataloader
    print("Building dataset...")
    dataset = build_dataset(cfg.data[eval_data])
    dataset = torch.utils.data.Subset(dataset, torch.randperm(len(dataset))[:config.subset_len])
    # the extra round_up data will be removed during gpu/cpu collect
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=cfg.data.samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False,
        round_up=False)

    return dataset, data_loader, cfg,

def main():
    args = parse_args()
    config = get_config(args)

    dataset, data_loader, cfg, = setup_cfg_and_data(config, config.eval_data)
    
    # Do tasks
    if 'gen_preds' in config.tasks:
        results = gen_preds(config, cfg, dataset, data_loader)
    else:
        results = mmcv.load(os.path.join(args.work_dir, 'result_keypoints_gts.json'))
    results = {k: np.array(v) for k, v in results.items()}

    if 'gen_cnet_data' in config.tasks:
        print("Saving HybrIK pred dataset to {}".format(config.cnet_dataset_path + '.npy'))

        # change to CNet format
        scale = 1000
        backbone_preds = (results['preds'] / scale).reshape(results['preds'].shape[0], -1)
        target_xyz_17s = (results['gts'] / scale).reshape(results['gts'].shape[0], -1)
        img_idss = np.repeat(results['ids'].reshape(-1,1), backbone_preds.shape[1], axis=1)
        
        out_path = config.cnet_dataset_path + '.npy'
        np.save(out_path, np.array([backbone_preds, target_xyz_17s, img_idss]))

    if 'eval_corrected' in config.tasks:
        # corr_pred_path = config.cnet_dataset_path + '.npy'
        corr_pred_path = config.cnet_dataset_path + '_corrected.npy'
        print("Evaluating corrected preds from {}".format(corr_pred_path))
        corr_preds, corr_gts, corr_ids = np.load(corr_pred_path)
        
        # change back to MMLab format
        corr_results = {}
        corr_results['preds'] = (corr_preds).reshape(-1, 17, 3)
        corr_results['gts'] = (corr_gts).reshape(-1, 17, 3)
        corr_results['ids'] = corr_ids[:,0].reshape(-1,1)
        # corr_results['preds'] = results['preds'] / 1000
        corr_results['poses'] = results['poses']
        corr_results['betas'] = results['betas']

        # sort them using the ids
        sort_idx = np.argsort(corr_results['ids'].reshape(-1))
        for k, v in corr_results.items():
            corr_results[k] = v[sort_idx]

        # do evaluation
        eval_cfg = cfg.get('evaluation', args.eval_options)
        eval_cfg.update(dict(metric=args.metrics))
        results = dataset.dataset.re_evaluate(corr_results=corr_results, **eval_cfg)
        
        print(results)


if __name__ == '__main__':
    main()

