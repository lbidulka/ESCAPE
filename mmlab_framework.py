import argparse
import os
import os.path as osp
from tqdm import tqdm

import cv2
from matplotlib import cm

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
import mmhuman3d.utils.camera_utils as mm_cam_utils
import mmhuman3d.core.conventions.cameras as mm_cam_conv
import mmhuman3d.core.cameras as mm_cams

import numpy as np
from types import SimpleNamespace

import utils.mmlab_varying_conditions as mmlab_vc
from core.metrics import get_PA_MPJPE

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
    # config.tasks = ['gen_cnet_data']
    # config.tasks = ['get_inference_time']
    # config.tasks = ['get_qualitative_results']
    config.tasks = ['get_varying_conditions_miniset']   # get_varying_conditions_miniset, get_varying_conditions, visualize_varying_conditions
    config.save_preds = True    # Save the generated preds?

    config.variation_type = 'darkening'    # occlusion, darkening
    config.variation_darkening_alpha = 0.1  # what fraction of raw img to keep, mixed with black
    config.variation_plot_raw = True      # if True, output the raw img plots, else output the varying condition (aug) plots
    config.variation_top_img_idxs = { 
        'PW3D': 
                # Worst Cnet
                [29000, 32746, 2620, 3163, 2966,  
                 22481,  12120, 16181, 4606, 20253,
                 7031, 5804, 19750, 30116, 28069, 3352,
                 13552, 21284, 33710, 26749, 10585
                 ]
                
                # Varying
                # [5392, 5842,  30053, 29994,  5,  34268, 21783, 13231, 24375, 19679, 11239]
                # [5385, 31051, 2162, 4153, 8149]
                # [5385, 21007, 31360, 19037, 14454, 
                # 19006, 31051, 20327, 15184, 6817, 
                # 24949, 29978, 2162, 15166, 24974, 
                # 29541, 2018, 32225, 31357, 7822, 
                # 4153, 8149]
                # [4153, 8149]
                #7, 
                # 11196, 34278, 24948, 
                # 21003, 5860, 13399, 35270,
                # 4853, 8149,]

                
                # 29551, 24046, , , 14342,
                # 16975,  , 15136, 24964,  4878,  , 
                # 24997, 21775, 15186,  2004, 13295,  5388,
                # 32059, 33431, 15209,  30527, 25660,
                # 13249,  8607, 15193,  6807, 13472, 13371,
                # 24490,  6793, 25591]
    }

    # Backbone generation settings
    config.backbone = 'pare'  # 'spin' 'pare' 'cliff' 'bal_mse' 'hybrik'
    config.dataset = 'PW3D' # 'PW3D' 'MPii' 'HP3D_train' 'HP3D_test' 'coco', h36m
    config.subset_len = 325_000 # hp3d: 110k, pwd3d: 35k, mpii: 14810, coco2017: 40055, h36m: 313k ????
    set_model_and_data_config(config)

    # Paths
    config.cnet_data_path = '/data/lbidulka/adapt_3d/'
    if config.dataset == 'HP3D_test':
        config.cnet_dataset_path = '{}{}/mmlab_{}_{}'.format(config.cnet_data_path, 'HP3D', config.backbone, 'test') 
    elif config.dataset == 'HP3D_train':
        config.cnet_dataset_path = '{}{}/mmlab_{}_cnet_{}'.format(config.cnet_data_path, 'HP3D', config.backbone, 'train') 
    elif config.dataset == 'PW3D':
        config.cnet_dataset_path = '{}{}/mmlab_{}_{}'.format(config.cnet_data_path, 'PW3D', config.backbone, 'test') 
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
        elif config.dataset == 'h36m':
            config.eval_data = 'h36m_eval'
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
        elif config.dataset == 'h36m':
            config.eval_data = 'h36m_eval'
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
        elif config.dataset == 'h36m':
            config.eval_data = 'h36m_eval'
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
        elif config.dataset == 'h36m':
            config.eval_data = 'h36m_eval'
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
        for rep in tqdm(range(repetitions)):
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


def gen_preds(config, cfg, dataset, data_loader, 
              zero_hips=False, save_results=None,):
    args = config.args
    if save_results is None: save_results = config.save_preds

    # build the model and load checkpoint
    model = build_architecture(cfg.model)
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    load_checkpoint(model, args.checkpoint, map_location=args.device)

    # Get preds
    if args.device == 'cpu':
        model = model.cpu()
    else:
        if args.device == 'cuda:0': device_id = 0
        elif args.device == 'cuda:1': device_id = 1
        model = MMDataParallel(model, device_ids=[device_id])
    outputs = single_gpu_test(model, data_loader)

    mmcv.mkdir_or_exist(osp.abspath(args.work_dir))
    results = dataset.dataset.get_results_and_human_gts(outputs, args.work_dir, save_results, align_root=zero_hips)
    
    return results, outputs

def setup_cfg_and_data(config, eval_data, data_ids=None):
    args = config.args
    cfg = mmcv.Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.data[eval_data].test_mode = True

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # build the dataloader
    print("Building dataset...")
    dataset = build_dataset(cfg.data[eval_data])
    if data_ids is None:
        dataset = torch.utils.data.Subset(dataset, torch.randperm(len(dataset))[:config.subset_len])
    else:
        dataset = torch.utils.data.Subset(dataset, torch.tensor(data_ids))
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
        elif task in ['get_qualitative_results', 'visualize_varying_conditions']:
            # scale and translate to fit bbx
            def proj_and_scale(gts, preds, corrs, boxes, cams):
                ''' '''

                # if cams is None:
                # project to 2d
                proj_gts = gts[..., :2] 
                proj_preds = preds[..., :2]
                proj_corrs = corrs[..., :2]

                # get scales & translations using gts
                scales = np.zeros((len(proj_gts), 2))
                translations = np.zeros((len(proj_gts), 2))
                for i, pose in enumerate(proj_gts):
                    bbx = boxes[i]
                    bbx_centre = bbx[:2] + bbx[2:-1] / 2
                    pose_w = np.abs(pose[:,0].max() - pose[:,0].min())
                    pose_h = np.abs(pose[:,1].max() - pose[:,1].min())
                    bbx_w = bbx[2]
                    bbx_h = bbx[3]
                    scales[i,0] = (bbx_w / pose_w) * 1 #0.95     # TEMP: ADJUST FOR DIFFERENT KPT THAN USED FOR BBX
                    scales[i,1] = (bbx_h / pose_h) * 1 #0.95    # TEMP: ADJUST FOR DIFFERENT KPT THAN USED FOR BBX
                    # translations[i] = img_centre + (pose * np.array([scales[i,0], scales[i,1]]))[0]  # gt rescaled hip
                    pose_centre = np.array([pose[:,0].mean(), pose[:,1].mean()])
                    translations[i] = bbx[:2] + bbx[2:-1] / 2 - (pose_centre * scales[i])
                scales = scales.reshape(-1,1,2)
                translations = translations.reshape(-1,1,2)
                # rescale and translate
                t_gts = proj_gts * scales + translations 
                t_preds = proj_preds * scales + translations
                t_corrs = proj_corrs * scales + translations
                # else:
                    # # project to 2d using camera
                    # resolutions = torch.tensor([(cam['H'], cam['W']) for cam in cams])
                    # Ks = torch.tensor([cam['in_mat'] for cam in cams])
                    # Rs = torch.tensor([cam['rotation_mat'] for cam in cams])
                    # Ts = torch.tensor([cam['translation'] for cam in cams])
                    # t_gts = mm_cam_utils.project_points(gts, resolution=resolutions, K=Ks, R=Rs, T=Ts)[...,:2]
                    # t_preds = mm_cam_utils.project_points(preds, resolution=resolutions, K=Ks, R=Rs, T=Ts)[...,:2]
                    # t_corrs = mm_cam_utils.project_points(corrs, resolution=resolutions, K=Ks, R=Rs, T=Ts)[...,:2]

                return t_gts.astype(int), t_preds.astype(int), t_corrs.astype(int)

            def img_draw_skeleton(img, kpts, limbs, kpt_color, limb_color):
                for j, joint in enumerate(kpts[i]):
                    img = cv2.circle(img, tuple(joint), 5, kpt_color, -1)    # colors[j]
                for limb in limbs.values():
                    for k in range(len(limb) - 1):
                        img = cv2.line(img, tuple(kpts[i][limb[k]]), tuple(kpts[i][limb[k+1]]), limb_color, 3)
                return img
            # def plt_draw_skeleton()
            
            # define skeleton limb connections, for drawing lines
            limbs = {
                'lleg': [0, 1, 2, 3],
                'rleg': [0, 4, 5, 6],
                'body': [0, 7, 8, 9, 10],
                'larm': [8, 11, 12, 13],
                'rarm': [8, 14, 15, 16],
            }

            def get_mpjpe(preds, gts, 
                          eval_joints=[6, 5, 4, 1, 2, 
                                       3, 16, 15, 14, 11, 
                                       12, 13, 8, 10]):
                ''' '''
                return np.sqrt(np.sum((preds[eval_joints]*1000 - gts[eval_joints]*1000)**2, axis=1).mean())
            
            def get_energy(pose):
                '''
                '''
                return torch.logsumexp(torch.tensor(pose.reshape(-1) * 1000), dim=-1).item()


            if task == 'get_qualitative_results':
                # load the top corrections from CNet framework and find the matching img_ids in the dataset
                qualitative_outpath = '../../outputs/qualitative'
                top_corr_path = '{}/{}_mmlab_{}_test_top_corr.npy'.format(qualitative_outpath, config.dataset, config.backbone)
                if config.dataset == 'HP3D_test':
                    top_corr_path = top_corr_path.replace('HP3D_test', 'HP3D')
                if config.backbone == 'hybrik':
                    top_corr_path = top_corr_path.replace('mmlab_', 'hrw48_wo_3dpw_cnet_')
                top_corr_data = np.load(top_corr_path)
                top_img_idxs = top_corr_data[0][:,0,0].astype(int)
                top_gts = top_corr_data[1]
                top_preds = top_corr_data[2]
                top_corrs = top_corr_data[3]
                
                top_corrs_noTTA = None

                top_img_idxs = [11853] # 24731, 25866, 23742,    32104, 26844, 4345, 31166, 32119
            elif task == 'visualize_varying_conditions':
                # load the top corrections from CNet framework and find the matching img_ids in the dataset
                qualitative_outpath = '../../outputs/qualitative/varying_conditions'
                top_corr_varying_path = '{}/{}_mmlab_{}_test_top_corr_varying_conditions.npy'.format(qualitative_outpath, config.dataset, config.backbone)
                top_corr_path = '{}/{}_mmlab_{}_test_top_corr.npy'.format(qualitative_outpath, config.dataset, config.backbone)
                if config.dataset != 'PW3D':
                    raise NotImplementedError
                top_corr_data = np.load(top_corr_path)
                top_corr_varying_data = np.load(top_corr_varying_path)

                top_img_idxs = top_corr_data[0][:,0,0].astype(int)

                # HARDCODE
                eval_joints_pw3d = [6, 5, 4, 1, 2, 3, 16, 15, 14, 11, 12, 13, 8, 10]
                top_img_idxs = config.variation_top_img_idxs[config.dataset]
                plot_raw = config.variation_plot_raw    # if True, output the raw img plots, else output the varying condition (aug) plots
                
                if plot_raw:
                    top_gts = top_corr_data[1]
                    top_preds = top_corr_data[2]
                    top_corrs_noTTA = top_corr_data[3]
                    top_corrs = top_corr_data[4]
                else:
                    top_gts = top_corr_varying_data[1]
                    top_preds = top_corr_varying_data[2]
                    top_corrs_noTTA = top_corr_varying_data[3]
                    top_corrs = top_corr_varying_data[4]
                
                for i, idx in enumerate(top_img_idxs):
                    # compute MPJPE, PA-MPJPE and Energy for the original and varying conditions. then print out
                    gt = top_gts[i]
                    
                    bb_mpjpe_raw = get_mpjpe(top_corr_data[2][i], gt)
                    bb_pa_mpjpe_raw = get_PA_MPJPE(top_corr_data[2][i:i+1,eval_joints_pw3d], top_gts[i:i+1,eval_joints_pw3d])[0].mean() * 1000
                    corr_noTTA_mpjpe_raw = get_mpjpe(top_corr_data[3][i], gt)
                    corr_noTTA_pa_mpjpe_raw = get_PA_MPJPE(top_corr_data[3][i:i+1,eval_joints_pw3d], top_gts[i:i+1,eval_joints_pw3d])[0].mean() * 1000
                    corr_mpjpe_raw = get_mpjpe(top_corr_data[4][i], gt)
                    corr_pa_mpjpe_raw = get_PA_MPJPE(top_corr_data[4][i:i+1,eval_joints_pw3d], top_gts[i:i+1,eval_joints_pw3d])[0].mean() * 1000
                    bb_E_raw = get_energy(top_corr_data[2][i])
                    corr_noTTA_E_raw = get_energy(top_corr_data[3][i])
                    corr_E_raw = get_energy(top_corr_data[4][i])

                    bb_mpjpe_aug = get_mpjpe(top_corr_varying_data[2][i], gt)
                    bb_pa_mpjpe_aug = get_PA_MPJPE(top_corr_varying_data[2][i:i+1,eval_joints_pw3d], top_gts[i:i+1,eval_joints_pw3d])[0].mean() * 1000
                    corr_noTTA_mpjpe_aug = get_mpjpe(top_corr_varying_data[3][i], gt)
                    corr_noTTA_pa_mpjpe_aug = get_PA_MPJPE(top_corr_varying_data[3][i:i+1,eval_joints_pw3d], top_gts[i:i+1,eval_joints_pw3d])[0].mean() * 1000
                    corr_mpjpe_aug = get_mpjpe(top_corr_varying_data[4][i], gt)
                    corr_pa_mpjpe_aug = get_PA_MPJPE(top_corr_varying_data[4][i:i+1,eval_joints_pw3d], top_gts[i:i+1,eval_joints_pw3d])[0].mean() * 1000
                    bb_E_aug = get_energy(top_corr_varying_data[2][i])
                    corr_noTTA_E_aug = get_energy(top_corr_varying_data[3][i])
                    corr_E_aug = get_energy(top_corr_varying_data[4][i])

                    print(f'\nImg {i} (id={idx}):')
                    print(f'Raw Img ------')
                    print(f'  (PA-MPJPE)\n   BB : {bb_pa_mpjpe_raw:.2f}\n   CNet: {corr_noTTA_pa_mpjpe_raw:.2f}\n   CNet+TTA: {corr_pa_mpjpe_raw:.2f}')
                    print(f'  (MPJPE)\n   BB : {bb_mpjpe_raw:.2f}\n   CNet: {corr_noTTA_mpjpe_raw:.2f}\n   CNet+TTA: {corr_mpjpe_raw:.2f}')
                    print(f'  (E)\n   BB : {bb_E_raw:.2f}\n   CNet: {corr_noTTA_E_raw:.2f}\n   CNet+TTA: {corr_E_raw:.2f}')
                    print(f'Aug Img ------')
                    print(f'  (PA-MPJPE)\n   BB : {bb_pa_mpjpe_aug:.2f}\n   CNet: {corr_noTTA_pa_mpjpe_aug:.2f}\n   CNet+TTA: {corr_pa_mpjpe_aug:.2f}')
                    print(f'  (MPJPE)\n   BB : {bb_mpjpe_aug:.2f}\n   CNet: {corr_noTTA_mpjpe_aug:.2f}\n   CNet+TTA: {corr_mpjpe_aug:.2f}')
                    print(f'  (E)\n   BB : {bb_E_aug:.2f}\n   CNet: {corr_noTTA_E_aug:.2f}\n   CNet+TTA: {corr_E_aug:.2f}')

            gt_smpl_params, gts, img_paths, bbxs, cams = dataset.dataset.get_gts_paths_bbxs_cams(top_img_idxs)
            
            # Check if I need to get the hip offsets again, because I didn't save them in the first place :/
            if top_gts[:,0].sum() == 0 and config.dataset != 'HP3D_test':
                top_dataset, top_data_loader, _ = setup_cfg_and_data(config, config.eval_data, data_ids=top_img_idxs)
                top_results, outputs = gen_preds(config, cfg, top_dataset, top_data_loader, zero_hips=False, save_results=False)
                top_results = {k: np.array(v) for k, v in top_results.items()}

                if top_preds[:,0].sum() == 0:
                    hips = top_results['preds'][:,:1]
                    hips /= 1000
                top_preds += hips #/ 1000
                top_corrs += hips #/ 1000

            # load the images and overlay poses on them
            imgs = []
            for i, img_path in enumerate(img_paths):
                varying_condition_outpath = f'../../outputs/varying_conditions/{config.backbone}/{config.dataset}'
                if (task == 'visualize_varying_conditions') and not (plot_raw):
                    if (config.variation_type == 'occlusion'):
                        img_path = f'{varying_condition_outpath}/occ_img_{top_img_idxs[i]}.png'
                    else:
                        img_path = f'{varying_condition_outpath}/dark_img_{top_img_idxs[i]}.png'
                    
                imgs.append(cv2.imread(img_path))
            
            if config.dataset != 'HP3D_test':
                gt_proj, pred_proj, corr_proj = proj_and_scale(gts, top_preds, top_corrs, bbxs, cams,)
            
            # overlay the gt points on the images
            for i, img in enumerate(imgs):
                gt_colour = (0,0.75,0)     # (0,1,0)
                corr_colour = (1.0,0,0)   # (1,0,0)
                pred_colour = (0,0,0.75) # (1,0.8,0)
                gt_colour_cv = (int(gt_colour[2]*255), int(gt_colour[1]*255), int(gt_colour[0]*255))
                corr_colour_cv = (int(corr_colour[2]*255), int(corr_colour[1]*255), int(corr_colour[0]*255))
                pred_colour_cv = (int(pred_colour[2]*255), int(pred_colour[1]*255), int(pred_colour[0]*255))
                
                # Raw Input img
                img_outpath = '{}/samples/{}/{}_{}_{}_input.png'.format(qualitative_outpath, config.backbone, config.dataset, config.backbone, i)
                if plot_raw is not None:
                    if not plot_raw:
                        img_outpath = img_outpath.replace('samples', 'samples_aug')
                cv2.imwrite(img_outpath, img)

                # 2D skeleton overlay
                if config.dataset != 'HP3D_test':
                    # overlay skeletons on img
                    img = img_draw_skeleton(img, gt_proj, limbs, gt_colour_cv, gt_colour_cv)
                    img = img_draw_skeleton(img, corr_proj, limbs, corr_colour_cv, corr_colour_cv)
                    img = img_draw_skeleton(img, pred_proj, limbs, pred_colour_cv, pred_colour_cv)
                    # save the image
                    img_outpath = img_outpath.replace('input', 'over')
                    cv2.imwrite(img_outpath, img)
                
                # 3D plot of gt, pred, and corr
                for j, azim in enumerate([-60, 60, 120]):
                    import matplotlib.pyplot as plt
                    fig = plt.figure()
                    ax = fig.add_subplot(111, projection='3d')
                    ax.set(xlim=(0.5,-0.5), ylim=(-0.5,0.5), zlim=(-0.7,0.3),
                        xticklabels=[], yticklabels=[], zticklabels=[],
                        )
                    ax.view_init(25, azim)

                    # plot the skeletons by drawing lines between the appropriate joints
                    plot_gt = gts[i] - gts[i][0]
                    plot_corr = top_corrs[i] - gts[i][0]
                    plot_pred = top_preds[i] - gts[i][0]
                    if (config.dataset != 'HP3D_test') and (cams[i] is not None):
                        plot_gt = plot_gt @ np.array(cams[i]['rotation_mat'])
                        plot_corr = plot_corr @ np.array(cams[i]['rotation_mat'])
                        plot_pred = plot_pred @ np.array(cams[i]['rotation_mat'])
                    for limb in limbs.values():
                        for k in range(len(limb) - 1):
                            first = limb == limbs['lleg'] and k == 0
                            ax.plot3D(plot_gt[limb[k:k+2],0], plot_gt[limb[k:k+2],2], plot_gt[limb[k:k+2],1], color=gt_colour, label='gt' if first else '')
                            ax.plot3D(plot_corr[limb[k:k+2],0], plot_corr[limb[k:k+2],2], plot_corr[limb[k:k+2],1], color=corr_colour, label='corr_backbone_pred' if first else '')
                            ax.plot3D(plot_pred[limb[k:k+2],0], plot_pred[limb[k:k+2],2], plot_pred[limb[k:k+2],1], color=pred_colour, label='backbone_pred' if first else '')
                    ax.legend(loc='lower left')
                    # save fig
                    fig_outpath = '{}/samples/{}/{}_{}_{}_3d_{}.png'.format(qualitative_outpath, config.backbone, config.dataset, config.backbone, i, j)                    

                    if plot_raw is not None:
                        if not plot_raw:
                            fig_outpath = fig_outpath.replace('samples', 'samples_aug')

                    plt.savefig(fig_outpath, bbox_inches='tight', dpi=300)
                    plt.close()

        elif task == 'get_varying_conditions':
            top_img_idxs = config.variation_top_img_idxs[config.dataset]

            gt_smpl_params, gts, img_paths, bbxs, cams = dataset.dataset.get_gts_paths_bbxs_cams(top_img_idxs)

            # save the imgs
            varying_condition_outpath = f'../../outputs/varying_conditions/{config.backbone}/{config.dataset}'
            if config.dataset == 'PW3D':
                varying_condition_dataset_outpath = 'ESCAPE_varying_conditions_imageFiles'
                dataset_path = './data/datasets/pw3d/' #config.cnet_data_path + config.dataset.lower()
            else:
                raise NotImplementedError
            for i, img_path in enumerate(img_paths):
                img = cv2.imread(img_path)

                # DEBUG: try out occlusions
                img_occ = mmlab_vc.add_occlusion(img, top_img_idxs[i], 
                                                 type='feet', dataset=config.dataset.lower())
                cv2.imwrite(f'{varying_condition_outpath}/occ_img_{top_img_idxs[i]}.png', img_occ)
                cv2.imwrite(f'{dataset_path}/{varying_condition_dataset_outpath}/occ_img_{top_img_idxs[i]}.png', img_occ)

                # Raw img
                cv2.imwrite(f'{varying_condition_outpath}/raw_img_{top_img_idxs[i]}.png', img)
                # Darkened img
                dark_alpha = 0.15
                img = cv2.addWeighted(img, dark_alpha, np.zeros_like(img), 0, 0)
                cv2.imwrite(f'{varying_condition_outpath}/dark_img_{top_img_idxs[i]}.png', img)
                cv2.imwrite(f'{dataset_path}/{varying_condition_dataset_outpath}/dark_img_{top_img_idxs[i]}.png', img)


            top_dataset, top_data_loader, _ = setup_cfg_and_data(config, config.eval_data, data_ids=top_img_idxs)
            # top_results, outputs = gen_preds(config, cfg, top_dataset, top_data_loader, zero_hips=False, save_results=False)

            # set dataset paths to the new varying conditions paths
            for i, img_idx in enumerate(top_img_idxs):
                if (config.variation_type == 'occlusion'):
                    top_dataset.dataset.human_data['image_path'][img_idx] = f'{varying_condition_dataset_outpath}/occ_img_{img_idx}.png'
                else:
                    top_dataset.dataset.human_data['image_path'][img_idx] = f'{varying_condition_dataset_outpath}/dark_img_{img_idx}.png'
            # Get results for the varying conditions, and save to a new "dataset" for cnet
            top_results, outputs = gen_preds(config, cfg, top_dataset, top_data_loader, zero_hips=False, save_results=False)

            # save the varying conditions preds to a new dataset
            out_path = config.cnet_dataset_path + '_varying_conditions.npy'
            print("Saving varying conditions pred dataset to {}".format(out_path))
            # change to CNet format
            scale = 1000
            backbone_preds = (top_results['preds'] / scale).reshape(top_results['preds'].shape[0], -1)
            target_xyz_17s = (top_results['gts'] / scale).reshape(top_results['gts'].shape[0], -1)
            img_idss = np.repeat(np.array(top_results['ids']).reshape(-1,1), backbone_preds.shape[1], axis=1)

            np.save(out_path, np.array([backbone_preds, target_xyz_17s, img_idss,]))

            foo = 5

        elif task == 'get_varying_conditions_miniset':
            '''
            '''
            top_img_idxs = mmlab_vc.vcd_samples_idxs
            gt_smpl_params, gts, img_paths, bbxs, cams = dataset.dataset.get_gts_paths_bbxs_cams(top_img_idxs)
            # save the imgs
            varying_condition_outpath = f'../../outputs/varying_conditions/{config.backbone}/{config.dataset}'
            if config.dataset == 'PW3D':
                varying_condition_dataset_outpath = 'ESCAPE_varying_conditions_miniset_imageFiles'
                dataset_path = './data/datasets/pw3d/' #config.cnet_data_path + config.dataset.lower()
            else:
                raise NotImplementedError
            for i, img_path in enumerate(img_paths):
                img = cv2.imread(img_path)

                # Occlusion
                img_occ = mmlab_vc.add_occlusion(img, top_img_idxs[i], 
                                                 type='feet', dataset=config.dataset.lower())
                # cv2.imwrite(f'{varying_condition_outpath}/occ_img_{top_img_idxs[i]}.png', img_occ)
                cv2.imwrite(f'{dataset_path}/{varying_condition_dataset_outpath}/occ_img_{top_img_idxs[i]}.png', img_occ)

                # Raw img
                # cv2.imwrite(f'{varying_condition_outpath}/raw_img_{top_img_idxs[i]}.png', img)
                # Darkened img
                img = cv2.addWeighted(img, config.variation_darkening_alpha, np.zeros_like(img), 0, 0)
                # cv2.imwrite(f'{varying_condition_outpath}/dark_img_{top_img_idxs[i]}.png', img)
                cv2.imwrite(f'{dataset_path}/{varying_condition_dataset_outpath}/dark_img_{top_img_idxs[i]}.png', img)
            
            # set dataset paths to the new varying conditions paths
            top_dataset, top_data_loader, _ = setup_cfg_and_data(config, config.eval_data, data_ids=top_img_idxs)
            for i, img_idx in enumerate(top_img_idxs):
                if (config.variation_type == 'occlusion'):
                    raise NotImplementedError
                else:
                    top_dataset.dataset.human_data['image_path'][img_idx] = f'{varying_condition_dataset_outpath}/dark_img_{img_idx}.png'
            # Get results for the varying conditions, and save to a new "dataset" for cnet
            top_results, outputs = gen_preds(config, cfg, top_dataset, top_data_loader, zero_hips=False, save_results=False)

            # save the varying conditions preds to a new dataset
            if config.variation_type == 'darkening':
                out_path = f'{config.cnet_dataset_path}_varying_conditions_miniset_darkening_{config.variation_darkening_alpha}.npy'
            else:
                raise NotImplementedError

            print("Saving varying conditions pred dataset to {}".format(out_path))
            # change to CNet format
            scale = 1000
            backbone_preds = (top_results['preds'] / scale).reshape(top_results['preds'].shape[0], -1)
            target_xyz_17s = (top_results['gts'] / scale).reshape(top_results['gts'].shape[0], -1)
            img_idss = np.repeat(np.array(top_results['ids']).reshape(-1,1), backbone_preds.shape[1], axis=1)

            np.save(out_path, np.array([backbone_preds, target_xyz_17s, img_idss,]))


        elif task != 'gen_preds':   # not the cleanest way to do this
            raise NotImplementedError

if __name__ == '__main__':
    main()

