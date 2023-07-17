import torch
import numpy as np
from types import SimpleNamespace
from tqdm import tqdm
import torch.nn as nn
import os

import mmpose.apis
import mmpose.utils

import utils.errors as errors
import utils.pose_processing as pose_processing

def setup_2d_model(config):
    '''
    '''
    config.mmpose_root_dir = '/home/luke/lbidulka/uncertnet_poserefiner/backbones/mmpose/'
    config.mmpose_config_file = config.mmpose_root_dir + 'ckpts/' + 'td-hm_hrnet-w48_8xb32-210e_coco-256x192.py'
    config.mmpose_ckpt_file = config.mmpose_root_dir + 'ckpts/' + 'td-hm_hrnet-w48_8xb32-210e_coco-256x192-0e67c616_20220913.pth'
    mmpose.utils.register_all_modules()
    model = mmpose.apis.init_model(config.mmpose_config_file, config.mmpose_ckpt_file, device='cpu')  # or device='cuda:0'
    return model

def get_2d_preds(model_2d, inps, config):
    mmpose_results = mmpose.apis.inference_topdown(model=model_2d, img=inps[0].cpu().numpy())   # fwd pass

    # convert kpt format from COCO to H36M
    # keypoints = mmpose_results[0].pred_instances.keypoints[0]
    # mmpose_results[0].pred_instances.keypoints[0] = utils.mmpose.convert_keypoint_definition(
    #     keypoints, 'TopDownCocoDataset', 'Body3DH36MDataset')
    h36m_kpts = pose_processing.convert_kpts_coco_h36m(mmpose_results[0].pred_instances.keypoints[0])

    # Reformat & zero hip
    poses_2d = h36m_kpts.reshape(-1, 17, 2)
    poses_2d -= poses_2d[:,0]  # zero hip
    poses_2d = torch.from_numpy(poses_2d).to(config.device)

    return poses_2d


def cnet_TTT_loss(config, backbone_pred, Rcnet_pred, corrected_pred, poses_2d, err_scale=1000):
    '''
    CNet loss for TTT, based on config

    consistency: computes L2(proximal kpts predicted by Rcnet from cnet preds, orignal backbone preds)
    '''

    if config.TTT_loss == 'reproj_2d':
        loss = errors.loss_weighted_rep_no_scale(poses_2d, corrected_pred, sum_kpts=True).sum()
    elif config.TTT_loss == 'consistency':
        loss = torch.square((backbone_pred[:,config.proximal_kpts,:] - Rcnet_pred[:,config.proximal_kpts,:])*err_scale).mean()
    else:
        raise NotImplementedError
    return loss

def unpack_test_data(data, m, model_2d, use_data_file, config, flip_test=True):
    '''
    Unpack CNet test sample 
    '''
    if use_data_file:
        labels = {}
        output = SimpleNamespace()
        output.pred_xyz_jts_17 = data[0][:,0].to(config.device)
        labels['target_xyz_17'] = data[0][:,1].to(config.device)
        img_ids = data[0][:,2,:1].to(config.device)
        poses_2d = labels['target_xyz_17'].reshape(labels['target_xyz_17'].shape[0], -1, 3)[:,:,:2]    # TEMP: using labels for 2D
    else:
        (inps, labels, img_ids, bboxes) = data
        if isinstance(inps, list):
            inps = [inp.to(config.device) for inp in inps]
        else:
            inps=inps.to(config.device)
        for k, _ in labels.items():
            try:
                labels[k] = labels[k].to(config.device)
            except AttributeError:
                assert k == 'type'
        with torch.no_grad():
            output = m(inps, flip_test=flip_test, bboxes=bboxes.to(config.device), img_center=labels['img_center'])

            if config.test_adapt:
                poses_2d = get_2d_preds(model_2d, inps, config)
            else:
                poses_2d = None

    return output, labels, img_ids, poses_2d

def eval_gt(m, cnet, R_cnet, config, gt_eval_dataset, 
            test_cnet=False, test_adapt=False, use_data_file=False):
    if test_adapt:
        batch_size = 1 # 1
    else:
        batch_size = 128
    gt_eval_dataset_for_scoring = gt_eval_dataset
    # Data/Setup
    if use_data_file:
        test_file = config.cnet_testset_path
        test_data = torch.from_numpy(np.load(test_file)).float().permute(1,0,2)
        test_data = test_data[config.test_eval_subset, :]
        gt_eval_dataset = torch.utils.data.TensorDataset(test_data)
    gt_eval_loader = torch.utils.data.DataLoader(gt_eval_dataset, batch_size=batch_size, shuffle=False, 
                                                 num_workers=16, drop_last=False, pin_memory=True)
    kpt_pred = {}
    kpt_all_pred = {}
    m.eval()
    cnet.eval()
    R_cnet.eval()
    if test_adapt and (config.TTT_loss == 'reproj_2d') and (use_data_file == False):
        model_2d = setup_2d_model(config)
    else: model_2d = None

    def set_bn_eval(module):
        ''' Batch Norm layer freezing '''
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()

    for it, data in enumerate(tqdm(gt_eval_loader, dynamic_ncols=True)):
        output, labels, img_ids, poses_2d = unpack_test_data(data, m, model_2d, use_data_file, config)
        if test_cnet:
            backbone_pred = output.pred_xyz_jts_17
            backbone_pred = backbone_pred.reshape(-1, 17, 3)

            if config.use_cnet:
                corrected_pred_init = None
                if test_adapt:
                    # THIS IS SUPER INNEFICIENT, AND WASTES MY TIME :(
                    cnet.load_cnets(print_str=False)
                    if config.split_corr_dim_trick:
                        cnet.cnet.eval()
                        corrected_pred_init = cnet(backbone_pred.detach().clone())
                    cnet.cnet.train()
                    # Setup Optimizer
                    cnet.cnet.apply(set_bn_eval)
                    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, cnet.cnet.parameters()),
                                                    lr=config.test_adapt_lr)
                    cnet_in = backbone_pred.detach().clone()
                    for i in range(config.adapt_steps):
                        corrected_pred = cnet(cnet_in)
                        loss = cnet_TTT_loss(config, backbone_pred, R_cnet(corrected_pred), corrected_pred, poses_2d)
                        loss.backward()
                        optimizer.step()
                        optimizer.zero_grad()
                with torch.no_grad():
                    # get corrected pred, with adjusted cnet
                    cnet_in = backbone_pred
                    corrected_pred = cnet(cnet_in)
                    if test_adapt and config.split_corr_dim_trick:
                        # correct z with initial CNet, leaving x/y corrected with tuned CNet
                        corrected_pred[:, :, 2] = corrected_pred_init[:, :, 2]
                output.pred_xyz_jts_17 = corrected_pred.reshape(labels['target_xyz_17'].shape[0], -1)
            else:
                output.pred_xyz_jts_17 = backbone_pred.reshape(labels['target_xyz_17'].shape[0], -1)

        pred_xyz_jts_17 = output.pred_xyz_jts_17.reshape(labels['target_xyz_17'].shape[0], 17, 3)
        pred_xyz_jts_17 = pred_xyz_jts_17.cpu().data.numpy()
        assert pred_xyz_jts_17.ndim in [2, 3]
        pred_xyz_jts_17 = pred_xyz_jts_17.reshape(pred_xyz_jts_17.shape[0], 17, 3)
        for i in range(pred_xyz_jts_17.shape[0]):
            kpt_pred[int(img_ids[i])] = {
                'xyz_17': pred_xyz_jts_17[i],
            }
        kpt_all_pred.update(kpt_pred)

    eval_summary_json = gt_eval_dataset_for_scoring.evaluate_xyz_17(kpt_all_pred, os.path.join('exp', 'test_3d_kpt.json'))
    return eval_summary_json