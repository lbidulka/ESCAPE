import argparse
import os
import os.path as osp
from tqdm import tqdm

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
    config.proj_root = '/home/luke/lbidulka/uncertnet_poserefiner/'
    config.root = 'uncertnet_poserefiner/backbones/mmhuman3d/'
    os.chdir(config.root)
    config.args = args

    # Tasks
    # config.tasks = ['gen_preds', 'eval', 'gen_cnet_data',]
    # config.tasks = ['gen_preds', 'eval', 'gen_cnet_data',]
    config.tasks = ['gen_preds', 'gen_cnet_data',]
    # config.tasks = ['gen_preds', 'gen_cnet_feats',]
    # config.tasks = ['gen_preds']
    # config.tasks = ['gen_cnet_data']
    config.tasks = ['get_inference_time']
    config.save_preds = False    # Save the generated preds?

    # Backbone generation settings
    config.backbone = 'spin'  # 'spin' 'pare' 'cliff' 'bal_mse' 'hybrik'
    config.dataset = 'PW3D' # 'PW3D' 'MPii' 'HP3D_train' 'HP3D_test' 'coco' 
    config.subset_len = 200_000 # hp3d: 110k, pwd3d: 35k, mpii: 14810, coco2017: 40055
    set_model_and_data_config(config)

    # Paths
    config.cnet_data_path = '/data/lbidulka/adapt_3d/'
    if config.dataset == 'HP3D_test':
        config.cnet_dataset_path = '{}{}/mmlab_{}_{}'.format(config.cnet_data_path, 'HP3D', config.backbone, 'test') 
    elif config.dataset == 'HP3D_train':
        config.cnet_dataset_path = '{}{}/mmlab_{}_cnet_{}'.format(config.cnet_data_path, 'HP3D', config.backbone, 'train') 
    else:
        config.cnet_dataset_path = '{}{}/mmlab_{}_cnet_{}'.format(config.cnet_data_path, config.dataset, config.backbone, 
                                                             'test' if (config.dataset == 'PW3D') else 'train') 
    return config

def set_model_and_data_config(config):
    if ('eval' in config.tasks) and (config.dataset != 'PW3D'):
        raise NotImplementedError
    # HybrIK
    if config.backbone == 'hybrik':
        config.args.config = config.proj_root + 'configs/mmhuman3d/hybrik_resnet34.py'
        config.args.checkpoint = 'data/pretrained/pretrain_hybrik.pth'
        config.args.work_dir = 'work_dirs/hybrik'
        if config.dataset == 'PW3D':
            config.eval_data = 'test'
        elif config.dataset == 'coco':    # train2017
            config.eval_data = 'coco_eval'
        elif config.dataset == 'HP3D_train':
            config.eval_data = 'hp3d_eval'
        elif config.dataset == 'HP3D_test':
            config.eval_data = 'hp3d_test_eval'
        else:
            raise NotImplementedError
    # Spin
    elif config.backbone == 'spin':
        config.args.config = config.proj_root + 'configs/mmhuman3d/spin_resnet50.py'
        config.args.checkpoint = 'data/pretrained/pretrain_spin.pth'
        config.args.work_dir = 'work_dirs/spin'
        if config.dataset == 'PW3D':
            config.eval_data = 'test'
        elif config.dataset == 'MPii':
            config.eval_data = 'mpii_eval'
        elif config.dataset == 'HP3D_train':
            config.eval_data = 'hp3d_eval'
        elif config.dataset == 'HP3D_test':
            config.eval_data = 'hp3d_test_eval'
        else:
            raise NotImplementedError
    # Cliff
    elif config.backbone == 'cliff':
        config.args.config = config.proj_root + 'configs/mmhuman3d/cliff_resnet50_pw3d_cache.py'
        config.args.checkpoint = 'data/pretrained/pretrain_cliff.pth'
        config.args.work_dir = 'work_dirs/cliff'
        if config.dataset == 'PW3D':
            config.eval_data = 'test'
        elif config.dataset == 'HP3D_train':
            config.eval_data = 'hp3d_eval'
        elif config.dataset == 'HP3D_test':
            config.eval_data = 'hp3d_test_eval'
        elif config.dataset == 'MPii':
            config.eval_data = 'mpii_eval'
        else: 
            raise NotImplementedError
    # Pare
    elif config.backbone == 'pare':
        config.args.config = config.proj_root + 'configs/mmhuman3d/pare_hrnet_w32_conv_mix_cache.py'
        config.args.checkpoint = 'data/pretrained/pretrain_pare_wMosh.pth'
        config.args.work_dir = 'work_dirs/pare'
        if config.dataset == 'PW3D':
            config.eval_data = 'test'
        elif config.dataset == 'HP3D_train':
            config.eval_data = 'hp3d_eval'
        elif config.dataset == 'HP3D_test':
            config.eval_data = 'hp3d_test_eval'
        elif config.dataset == 'MPii':
            config.eval_data = 'mpii_eval'
        else: 
            raise NotImplementedError
    # Balanced MSE
    elif config.backbone == 'bal_mse':
        config.args.config = config.proj_root + 'configs/mmhuman3d/bal_mse_resnet50_spin_ihmr_ft_bmc.py'
        config.args.checkpoint = 'data/pretrained/pretrain_bal_mse.pth'
        config.args.work_dir = 'work_dirs/bal_mse'
        if config.dataset == 'PW3D':
            config.eval_data = 'test'
        elif config.dataset == 'HP3D_train':
            config.eval_data = 'hp3d_eval'
        elif config.dataset == 'HP3D_test':
            config.eval_data = 'hp3d_test_eval'
        elif config.dataset == 'MPii':
            config.eval_data = 'mpii_eval'
        else: 
            raise NotImplementedError
    else:
        raise NotImplementedError
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
        choices=['cpu', 'cuda:0', 'cuda:1'],
        default='cuda:0',
        help='device used for testing')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args

def get_inference_time(config, eval_data):
    args = config.args
    cfg = mmcv.Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.data.test.test_mode = True

    # build the dataloader
    print("Building dataset...")
    dataset = build_dataset(cfg.data[eval_data])
    dataset = torch.utils.data.Subset(dataset, torch.randperm(len(dataset))[:config.subset_len])
    # the extra round_up data will be removed during gpu/cpu collect
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=2,
        workers_per_gpu=1,
        dist=False,
        shuffle=False,
        round_up=False)

    # build the model and load checkpoint
    model = build_architecture(cfg.model)
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    load_checkpoint(model, args.checkpoint, map_location='cpu')

    if args.device == 'cuda:0': device_id = 0
    elif args.device == 'cuda:1': device_id = 1
    model = MMDataParallel(model, device_ids=[device_id])

    model.eval()

    sample = next(iter(data_loader))
    # sample['img'].to(device_id)

    # INIT LOGGERS
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    repetitions = 10_000
    timings = np.zeros((repetitions,1))
    # GPU-WARM-UP
    for _ in range(10):
        _ = model(return_loss=False, **sample)
    print("Measuring...")
    # MEASURE PERFORMANCE
    with torch.no_grad():
        for rep in range(repetitions):
            starter.record()
            _ = model(return_loss=False, **sample)
            ender.record()
            # WAIT FOR GPU SYNC
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[rep] = curr_time
    # REPORT    
    mean_syn = np.sum(timings) / repetitions
    std_syn = np.std(timings)
    print(f'\nINFERENCE TIME OF {config.backbone}, {repetitions} REPS:')
    print(f'mean: {round(mean_syn, 4)} ms,  std: {round(std_syn,4)} ms')


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
        if args.device == 'cuda:0': device_id = 0
        elif args.device == 'cuda:1': device_id = 1
        model = MMDataParallel(model, device_ids=[device_id])
    outputs = single_gpu_test(model, data_loader)

    mmcv.mkdir_or_exist(osp.abspath(args.work_dir))
    results = dataset.dataset.get_results_and_human_gts(outputs, args.work_dir, config.save_preds)
    
    return results, outputs

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
    print("\n ----- Backbone: {}, Dataset: {} ----- \n".format(config.backbone, config.dataset))
    dataset, data_loader, cfg, = setup_cfg_and_data(config, config.eval_data)
    # Do tasks
    for task in config.tasks:
        if task == 'gen_preds':
            results, outputs = gen_preds(config, cfg, dataset, data_loader)
            results = {k: np.array(v) for k, v in results.items()}
        else:
            results = mmcv.load(os.path.join(args.work_dir, 'result_keypoints_gts.json'))
            results = {k: np.array(v) for k, v in results.items()}

        if task == 'eval': 
            # Only works for PW3D
            eval_cfg = cfg.get('evaluation', args.eval_options)
            eval_cfg.update(dict(metric=args.metrics))
            eval_results = dataset.dataset.evaluate(outputs, args.work_dir, **eval_cfg)
            print(eval_results)
        elif task == 'gen_cnet_data':
            out_path = config.cnet_dataset_path + '.npy'
            print("Saving {} pred dataset to {}".format(dataset, out_path))
            # change to CNet format
            scale = 1000
            backbone_preds = (results['preds'] / scale).reshape(results['preds'].shape[0], -1)
            target_xyz_17s = (results['gts'] / scale).reshape(results['gts'].shape[0], -1)
            img_idss = np.repeat(results['ids'].reshape(-1,1), backbone_preds.shape[1], axis=1)
            
            np.save(out_path, np.array([backbone_preds, target_xyz_17s, img_idss,]))
        elif task == 'gen_cnet_feats':
            out_path = config.cnet_dataset_path + '_feats'
            print("Saving {} pred features to {}".format(dataset, out_path))
            
            features = []
            for batch in outputs:
                features.append(batch['features'])
            features = np.concatenate(features)
            # change to CNet format
            # scale = 1000
            # backbone_preds = (results['preds'] / scale).reshape(results['preds'].shape[0], -1)
            # target_xyz_17s = (results['gts'] / scale).reshape(results['gts'].shape[0], -1)
            # img_idss = np.repeat(results['ids'].reshape(-1,1), backbone_preds.shape[1], axis=1)
            
            # save each feature vector to a file
            for i, feat in enumerate(tqdm(features)):
                # insert '/feature_maps/' right before 'mmlab'
                save_path = config.cnet_dataset_path
                save_path = save_path[:save_path.find('mmlab')] + 'feature_maps/' + save_path[save_path.find('mmlab'):]
                save_path = save_path.replace('_cnet_', '_')

                np.save(save_path + f'/{results["ids"][i]}.npy', feat)
            # np.save(out_path, np.array([features, img_idss,]))
        elif task == 'get_inference_time':
            get_inference_time(config, config.eval_data)
        elif task != 'gen_preds':   # not the cleanest way to do this
            raise NotImplementedError

if __name__ == '__main__':
    main()

