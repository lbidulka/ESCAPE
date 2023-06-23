"""uncertnet script for HybrIK"""
import argparse
import os
import sys
from pathlib import Path
from types import SimpleNamespace
import copy

import numpy as np
import torch

# path_root = Path(__file__)#.parents[3]
# sys.path.append(str(path_root))
# sys.path.append(str(path_root)+ 'backbones/HybrIK/hybrik/')

# print("\n", os.getcwd(), "\n", sys.path, "\n")

from hybrik.models import builder
from hybrik.utils.config import update_config
from torchvision import transforms as T
from tqdm import tqdm
det_transform = T.Compose([T.ToTensor()])

import pickle as pk
from hybrik.datasets import HP3D, PW3D, H36mSMPL
from hybrik.utils.transforms import get_func_heatmap_to_coord

import uncertnet.distal_cnet as uncertnet_models
from utils import eval

from uncertnet.cnet_all import adapt_net

def get_config():
    config = SimpleNamespace()
    config.root = 'uncertnet_poserefiner/backbones/HybrIK/'
    os.chdir(config.root)
    config.device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    #

    # Tasks
    config.tasks = ['train', 'test'] # 'make_trainset' 'train', 'test'
    # config.tasks = ['test']

    # Main Settings
    config.use_FF = True

    # HybrIK config
    config.hybrIK_version = 'hrw48_wo_3dpw' # 'res34_cam', 'hrw48_wo_3dpw'

    if config.hybrIK_version == 'res34_cam':
        config.cfg = 'configs/256x192_adam_lr1e-3-res34_smpl_3d_cam_2x_mix.yaml'
        config.ckpt = 'pretrained_w_cam.pth'
    if config.hybrIK_version == 'hrw48_wo_3dpw':
        config.cfg = 'configs/256x192_adam_lr1e-3-hrw48_cam_2x_wo_pw3d.yaml'    # w/o 3DPW
        config.ckpt = 'hybrik_hrnet48_wo3dpw.pth'    

    # 3DPW data
    config.data_3DPW_dir = '/media/ExtHDD01/Mohsen_data/3DPW'
    config.data_3DPW_test_annot = '{}/json/3DPW_test_new.json'.format(config.data_3DPW_dir)
    # config.data_3DPW_test_annot = '{}/3DPW_latest_test.json'.format(config.data_3DPW_dir)
    config.data_3DPW_train_annot = '{}/3DPW_latest_train.json'.format(config.data_3DPW_dir)
    config.data_3DPW_img_dir = '{}/imageFiles'.format(config.data_3DPW_dir)

    # cnet dataset
    config.cnet_ckpt_path = '../../ckpts/hybrIK/'
    config.cnet_dataset_path = '/media/ExtHDD01/luke_data/3DPW/' #'../../data/hybrIK/'

    print_useful_configs(config)
    return config

def print_useful_configs(config):
    print('\n -- Config: --')
    print('Tasks: {}'.format(config.tasks))
    print('Use FF: {}'.format(config.use_FF))
    print('hybrIK_version: {}'.format(config.hybrIK_version))    
    return

config = get_config()
parser = argparse.ArgumentParser(description='HybrIK Demo')

parser.add_argument('--gpu',
                    help='gpu',
                    default=0,
                    type=int)
parser.add_argument('--img-dir',
                    help='image folder',
                    default=config.data_3DPW_img_dir,
                    type=str)
parser.add_argument('--annot-dir',
                    help='image folder',
                    default=config.data_3DPW_train_annot,
                    type=str)
parser.add_argument('--out-dir',
                    help='output folder',
                    default='',
                    type=str)

parser.add_argument('--cfg',
                    help='experiment configure file name',
                    required=False,
                    type=str)
parser.add_argument('--checkpoint',
                    help='checkpoint file name',
                    required=False,
                    type=str)
parser.add_argument('--gpus',
                    help='gpus',
                    type=str)
parser.add_argument('--batch',
                    help='validation batch size',
                    type=int)
parser.add_argument('--flip-test',
                    default=False,
                    dest='flip_test',
                    help='flip test',
                    action='store_true')
parser.add_argument('--flip-shift',
                    default=False,
                    dest='flip_shift',
                    help='flip shift',
                    action='store_true')

opt = parser.parse_args()
cfg = update_config(config.cfg)

def create_cnet_dataset(m, opt, cfg, gt_dataset, task='train'):
    # Data/Setup
    gt_loader = torch.utils.data.DataLoader(gt_dataset, batch_size=128, shuffle=False, num_workers=8, drop_last=False)
    m.eval()
    m = m.to(config.device)

    hm_shape = cfg.MODEL.get('HEATMAP_SIZE')
    hm_shape = (hm_shape[1], hm_shape[0])

    backbone_preds = []
    target_xyz_17s = []
    for(inps, labels, img_ids, bboxes) in tqdm(gt_loader, dynamic_ncols=True):
        if isinstance(inps, list):
            inps = [inp.cuda(opt.gpu) for inp in inps]
        else:
            inps=inps.to(config.device)

        for k, _ in labels.items():
            try:
                labels[k] = labels[k].to(config.device)
            except AttributeError:
                assert k == 'type'

        m_output = m(inps, flip_test=opt.flip_test, bboxes=bboxes.to(config.device), img_center=labels['img_center'])
        backbone_pred = m_output.pred_xyz_jts_17

        backbone_preds.append(backbone_pred.detach().cpu().numpy())
        target_xyz_17s.append(labels['target_xyz_17'].detach().cpu().numpy())

    dataset_outpath = '{}{}_cnet_hybrik_{}.npy'.format(config.cnet_dataset_path, config.hybrIK_version, task)
    np.save(dataset_outpath, np.array([np.concatenate(backbone_preds, axis=0), np.concatenate(target_xyz_17s, axis=0)]))
    return

def eval_gt(m, cnet, opt, cfg, gt_eval_dataset, heatmap_to_coord, test_vertice=False, test_cnet=False, use_data_file=False):
    batch_size = 128
    # Data/Setup
    if use_data_file:
        gt_eval_dataset = torch.utils.data.TensorDataset(torch.from_numpy(np.load(config.cnet_dataset_path + 'test.npy')).float())
    gt_eval_loader = torch.utils.data.DataLoader(gt_eval_dataset, batch_size=batch_size, shuffle=False, num_workers=16, drop_last=False)
    kpt_pred = {}
    kpt_all_pred = {}
    m.eval()
    cnet.eval()

    hm_shape = cfg.MODEL.get('HEATMAP_SIZE')
    hm_shape = (hm_shape[1], hm_shape[0])
    pve_list = []

    errs = []
    errs_s = []

    for inps, labels, img_ids, bboxes in tqdm(gt_eval_loader, dynamic_ncols=True):

        if isinstance(inps, list):
            inps = [inp.cuda(opt.gpu) for inp in inps]
        else:
            # inps = inps.cuda(opt.gpu)
            inps=inps.to(config.device)

        for k, _ in labels.items():
            try:
                labels[k] = labels[k].to(config.device)
            except AttributeError:
                assert k == 'type'
        
        # HybrIK pred, then correct with cnet
        output = m(inps, flip_test=opt.flip_test, bboxes=bboxes.to(config.device), img_center=labels['img_center'])
        
        if test_cnet:
            backbone_pred = output.pred_xyz_jts_17
            backbone_pred = backbone_pred.reshape(inps.shape[0], -1, 3)
            poses_2d = labels['target_xyz_17'].reshape(inps.shape[0], -1, 3)[:,:,:2]    # TEMP: using labels for 2D
            
            if config.use_FF:
                cnet_in = (poses_2d, backbone_pred)
            else:
                cnet_in = backbone_pred

            corrected_pred = cnet(cnet_in)

            output.pred_xyz_jts_17 = corrected_pred.reshape(inps.shape[0], -1)

        # evaluate
        if test_vertice:
            gt_betas = labels['target_beta']
            gt_thetas = labels['target_theta']
            gt_output = m.forward_gt_theta(gt_thetas, gt_betas)

        pred_uvd_jts = output.pred_uvd_jts
        pred_xyz_jts_24 = output.pred_xyz_jts_24.reshape(inps.shape[0], -1, 3)[:, :24, :]
        pred_xyz_jts_24_struct = output.pred_xyz_jts_24_struct.reshape(inps.shape[0], 24, 3)
        pred_xyz_jts_17 = output.pred_xyz_jts_17.reshape(inps.shape[0], 17, 3)
        pred_mesh = output.pred_vertices.reshape(inps.shape[0], -1, 3)
        if test_vertice:
            gt_mesh = gt_output.vertices.reshape(inps.shape[0], -1, 3)
            gt_xyz_jts_17 = gt_output.joints_from_verts.reshape(inps.shape[0], 17, 3) / 2


        pred_xyz_jts_24 = pred_xyz_jts_24.cpu().data.numpy()
        pred_xyz_jts_24_struct = pred_xyz_jts_24_struct.cpu().data.numpy()
        pred_xyz_jts_17 = pred_xyz_jts_17.cpu().data.numpy()
        pred_uvd_jts = pred_uvd_jts.cpu().data
        pred_mesh = pred_mesh.cpu().data.numpy()
        if test_vertice:
            gt_mesh = gt_mesh.cpu().data.numpy()
            gt_xyz_jts_17 = gt_xyz_jts_17.cpu().data.numpy()

        assert pred_xyz_jts_17.ndim in [2, 3]
        pred_xyz_jts_17 = pred_xyz_jts_17.reshape(pred_xyz_jts_17.shape[0], 17, 3)
        pred_uvd_jts = pred_uvd_jts.reshape(pred_uvd_jts.shape[0], -1, 3)
        pred_xyz_jts_24 = pred_xyz_jts_24.reshape(pred_xyz_jts_24.shape[0], 24, 3)
        pred_scores = output.maxvals.cpu().data[:, :29]

        if test_vertice:
            pve = np.sqrt(np.sum((pred_mesh - gt_mesh) ** 2, 2))
            pve_list.append(np.mean(pve) * 1000)

        for i in range(pred_xyz_jts_17.shape[0]):
            bbox = bboxes[i].tolist()
            pose_coords, pose_scores = heatmap_to_coord(
                pred_uvd_jts[i], pred_scores[i], hm_shape, bbox, mean_bbox_scale=None)
            kpt_pred[int(img_ids[i])] = {
                'xyz_17': pred_xyz_jts_17[i],
                'vertices': pred_mesh[i],
                'uvd_jts': pose_coords[0],
                'xyz_24': pred_xyz_jts_24_struct[i]
            }
        kpt_all_pred.update(kpt_pred)

    tot_err_17 = gt_eval_dataset.evaluate_xyz_17(kpt_all_pred, os.path.join('exp', 'test_3d_kpt.json'))
    if test_vertice:
        print(f'PVE: {np.mean(pve_list)}')
    return tot_err_17

def main_worker(opt, cfg, model=None):
    torch.backends.cudnn.benchmark = True
    if model is None:
        m = builder.build_sppe(cfg.MODEL)

        print(f'Loading HybrIK model from {opt.checkpoint}...')
        save_dict = torch.load(opt.checkpoint)
        if type(save_dict) == dict:
            model_dict = save_dict['model']
            m.load_state_dict(model_dict, strict=False)
        else:
            m.load_state_dict(save_dict, strict=False)
    else:
        m = model
    heatmap_to_coord = get_func_heatmap_to_coord(cfg)   

    # Datasets
    gt_train_dataset_3dpw = PW3D(
        cfg=cfg,
        ann_file='3DPW_train_new_fresh.json',
        train=False,
        root='/media/ExtHDD01/Mohsen_data/3DPW')
    
    gt_test_dataset_3dpw = PW3D(
        cfg=cfg,
        # ann_file='3DPW_test_new.json',
        ann_file='3DPW_test_new_fresh.json',
        train=False,
        root='/media/ExtHDD01/Mohsen_data/3DPW')

    print(' USING HYBRIK VER: {}'.format(config.hybrIK_version))

    with torch.no_grad():
        if 'make_trainset' in config.tasks:
            print('##### Creating CNET 3DPW Trainset #####')
            create_cnet_dataset(m, opt, cfg, gt_train_dataset_3dpw, task='train')
        # # create_cnet_dataset(m, opt, cfg, gt_test_dataset_3dpw, task='test')
    m = m.to('cpu')

    config.train_datalim = None
    # config.train_datalim = 5_000

    # cnet = uncertnet_models.multi_distal_cnet(config)
    # cnet = uncertnet_models.all_limb_cnet(config) 
    cnet = adapt_net(config)

    if 'train' in config.tasks:
        cnet.train()
    cnet.load_cnets()

    m = m.to(config.device)

    # --- TEST SET EVAL ---
    if 'test' in config.tasks:
        print('\n##### 3DPW TESTSET ERRS #####\n')
        with torch.no_grad():
            tot_corr_PA_MPJPE = eval_gt(m, cnet, opt, cfg, gt_test_dataset_3dpw, heatmap_to_coord, test_vertice=False, test_cnet=True)

        print('\n--- Vanilla: --- ')
        # with torch.no_grad():
        #     gt_tot_err = eval_gt(m, cnet, opt, cfg, gt_test_dataset_3dpw, heatmap_to_coord, test_vertice=False, test_cnet=False)
        if config.hybrIK_version == 'res34_cam':
            print('XYZ_14 PA-MPJPE: 45.917672 | MPJPE: 74.113751, x: 27.145215, y: 28.64, z: 51.785723')  # w/ 3DPW
        if config.hybrIK_version == 'hrw48_wo_3dpw':
            print('XYZ_14 PA-MPJPE: 49.346562 | MPJPE: 88.707589, x: 29.233308, y: 30.03, z: 66.807150')  # wo/ 3DPW

def load_pretrained_hybrik(ckpt=config.ckpt):
    hybrik_model = builder.build_sppe(cfg.MODEL)
    
    print(f'\nLoading model from {ckpt}...\n')
    save_dict = torch.load(ckpt, map_location='cpu')
    if type(save_dict) == dict:
        model_dict = save_dict['model']
        hybrik_model.load_state_dict(model_dict)
    else:
        hybrik_model.load_state_dict(save_dict, strict=False)

    return hybrik_model

if __name__ == "__main__":    
    model = load_pretrained_hybrik()
    main_worker(opt, cfg, model=model)

