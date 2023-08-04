import torch
import numpy as np
from types import SimpleNamespace
from tqdm import tqdm
import torch.nn as nn
import os

# import mmpose.apis
# import mmpose.utils

import utils.errors as errors
import utils.pose_processing as pose_processing
import utils.quick_plot

from core import cnet_data

# def setup_2d_model(config):
#     '''
#     '''
#     config.mmpose_root_dir = '/home/luke/lbidulka/uncertnet_poserefiner/backbones/mmpose/'
    
#     config.mmpose_config_file = config.mmpose_root_dir + 'ckpts/' + 'td-hm_hrnet-w48_udp-8xb32-210e_coco-256x192.py'
#     config.mmpose_ckpt_file = config.mmpose_root_dir + 'ckpts/' + 'td-hm_hrnet-w48_udp-8xb32-210e_coco-256x192-3feaef8f_20220913.pth'

#     # config.mmpose_config_file = config.mmpose_root_dir + 'ckpts/' + 'td-hm_hrnet-w48_8xb32-210e_coco-256x192.py'
#     # config.mmpose_ckpt_file = config.mmpose_root_dir + 'ckpts/' + 'td-hm_hrnet-w48_8xb32-210e_coco-256x192-0e67c616_20220913.pth'

#     # config.mmpose_config_file = config.mmpose_root_dir + 'ckpts/' + 'td-hm_litehrnet-30_8xb64-210e_coco-256x192.py'
#     # config.mmpose_ckpt_file = config.mmpose_root_dir + 'ckpts/' + 'litehrnet30_coco_256x192-4176555b_20210626.pth'

#     mmpose.utils.register_all_modules()
#     model = mmpose.apis.init_model(config.mmpose_config_file, config.mmpose_ckpt_file, device='cpu')  # or device='cuda:0'
#     return model

# def get_2d_preds(model_2d, inps, labels, config):
#     mmpose_results = mmpose.apis.inference_topdown(model_2d, labels['img_path'][0])   # fwd pass
#     mmpose_preds = mmpose_results[0].pred_instances.keypoints[0]

#     # convert kpt format from COCO to H36M
#     h36m_kpts = pose_processing.convert_kpts_coco_h36m(mmpose_preds)

#     # Reformat & zero hip
#     poses_2d = h36m_kpts.reshape(-1, 17, 2)
#     poses_2d -= poses_2d[:,0]  # zero hip
#     poses_2d = torch.from_numpy(poses_2d).to(config.device)

#     return poses_2d

def cnet_TTT_loss(config, backbone_pred, Rcnet_pred, corrected_pred, poses_2d,):
    '''
    CNet loss for TTT, based on config

    consistency: computes L2(proximal kpts predicted by Rcnet from cnet preds, orignal backbone preds)
    '''

    if config.TTT_loss == 'reproj_2d':
        loss = errors.loss_weighted_rep_no_scale(poses_2d[:,config.cnet_targets], 
                                                 corrected_pred[:,config.cnet_targets], 
                                                 sum_kpts=True, num_joints=len(config.cnet_targets)).sum()
    elif config.TTT_loss == 'consistency':
        loss = (backbone_pred[:,config.rcnet_targets] - Rcnet_pred[:,config.rcnet_targets])
        loss = torch.square(loss*config.TTT_errscale).mean()
        # loss = torch.abs(loss*config.TTT_errscale).mean()
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
                assert (k == 'type') or (k == 'img_path')
        with torch.no_grad():
            output = m(inps, flip_test=flip_test, bboxes=bboxes.to(config.device), img_center=labels['img_center'])

            if config.test_adapt:
                poses_2d = get_2d_preds(model_2d, inps, labels, config)
            else:
                poses_2d = None

    return output, labels, img_ids, poses_2d

def eval_gt(cnet, R_cnet, config, 
            m=None, gt_eval_dataset=None, 
            testset_path=None, backbone_scale=1.0,
            test_cnet=False, test_adapt=False, 
            use_data_file=False, mmlab_out=False):
    '''
    '''
    if test_adapt:
        batch_size = 1 # 1
    else:
        batch_size = 128
    # Data/Setup
    if use_data_file:
        if mmlab_out:
            test_file = config.mmlab_testset_path + '.npy'
        else:
            test_file = testset_path
        test_data = torch.from_numpy(np.load(test_file)).float().permute(1,0,2)
        if config.test_adapt and (use_data_file == True):
            test_data = test_data[:config.test_eval_limit]
        else:
            test_data = test_data[config.test_eval_subset, :]
        test_data *= backbone_scale
        gt_eval_dataset = torch.utils.data.TensorDataset(test_data)
    gt_eval_loader = torch.utils.data.DataLoader(gt_eval_dataset, batch_size=batch_size, shuffle=False, 
                                                #  num_workers=16, drop_last=False, pin_memory=True)
                                                drop_last=False, pin_memory=True)
    kpt_pred = {}
    kpt_all_pred = {}
    if m is not None: m.eval()
    cnet.eval()
    R_cnet.eval()
    if test_adapt and (config.TTT_loss == 'reproj_2d') and (use_data_file == False):
        model_2d = setup_2d_model(config)
    else: model_2d = None

    def set_bn_eval(module):
        ''' Batch Norm layer freezing '''
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()

    backbone_preds = []
    corr_preds = []
    gts = []
    losses = []
    for it, data in enumerate(tqdm(gt_eval_loader, dynamic_ncols=True)):
        output, labels, img_ids, poses_2d = unpack_test_data(data, m, model_2d, use_data_file, config)
        backbone_pred = output.pred_xyz_jts_17.reshape(-1, 17, 3)
        if test_cnet:
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
                        Rcnet_pred = R_cnet(corrected_pred)
                        loss = cnet_TTT_loss(config, backbone_pred, Rcnet_pred, corrected_pred, poses_2d)
                        loss.backward()
                        optimizer.step()
                        optimizer.zero_grad()

                        gt_3d = labels['target_xyz_17'].reshape(labels['target_xyz_17'].shape[0], -1, 3)
                        gt_mse = torch.square((backbone_pred[:,config.distal_kpts,:] - gt_3d[:,config.distal_kpts])*config.TTT_errscale).mean()
                        # cnet_outs = torch.flatten((backbone_pred-corrected_pred)[:, config.distal_kpts], start_dim=1)
                        # cnet_loss = cnet._loss(backbone_pred, cnet_outs, gt_3d)
                        losses.append([loss.item(), gt_mse.item()])
                        # losses.append([loss.item(), cnet_loss.item()])

                with torch.no_grad():
                    # get corrected pred, with adjusted cnet
                    cnet_in = backbone_pred
                    corrected_pred = cnet(cnet_in)
                    if test_adapt and config.split_corr_dim_trick:
                        # correct z with initial CNet, leaving x/y corrected with tuned CNet
                        corrected_pred[:, :, 2] = corrected_pred_init[:, :, 2]
                corrected_pred = corrected_pred.reshape(labels['target_xyz_17'].shape[0], -1)
                output.pred_xyz_jts_17 = corrected_pred
            else:
                output.pred_xyz_jts_17 = backbone_pred.reshape(labels['target_xyz_17'].shape[0], -1)

        backbone_preds.append(backbone_pred.detach().cpu().numpy())
        corr_preds.append(corrected_pred.reshape(labels['target_xyz_17'].shape[0], -1, 3).detach().cpu().numpy())
        gts.append(labels['target_xyz_17'].reshape(labels['target_xyz_17'].shape[0], -1, 3).detach().cpu().numpy())
        for i in range(output.pred_xyz_jts_17.shape[0]):
            pred_xyz_jts_17 = output.pred_xyz_jts_17[i].data.cpu().numpy()
            kpt_pred[int(img_ids[i])] = {
                'xyz_17': pred_xyz_jts_17.reshape(-1, 3),
            }
        kpt_all_pred.update(kpt_pred)
        # early stop if testing on subset & not using file
        if not use_data_file and (it+1) >= config.test_eval_limit:
            break

    # Investigation of TTT loss
    if test_adapt:
        losses = np.array(losses)
        if config.TTT_loss == 'consistency':
            utils.quick_plot.simple_2d_plot(losses, save_dir='../../outputs/testset/Knees_losses_consist.png', 
                                            title='Knees Consist. Loss vs GT 3D distals MSE', 
                                            xlabel='Consistency Loss', ylabel='GT 3D MSE Loss for Distals',
                                            # x_lim=[0, 25], y_lim=[0, 7500])
                                            x_lim=[0, 1000], y_lim=[0, 7500])
        if config.TTT_loss == 'reproj_2d':
            utils.quick_plot.simple_2d_plot(losses, save_dir='../../outputs/testset/losses_2d.png', 
                                            title='2D reproj. Loss vs GT 3D distals MSE', 
                                            xlabel='2D reproj Loss', ylabel='GT 3D MSE Loss for Distals',
                                            x_lim=[0,1], y_lim=[0, 7500])

    # save updated file for mmlab eval (preds, gts, ids)
    if mmlab_out:
        test_data = torch.from_numpy(np.load(test_file)).float()
        # replace preds with corrected preds
        for id in kpt_all_pred.keys():
            test_data[0, np.where(test_data[2,:,0]==id)] = torch.from_numpy(kpt_all_pred[id]['xyz_17']).reshape(1,1,51)
        # save npy file
        test_out_file = config.mmlab_testset_path + '_corrected.npy'
        print("Saving corrected preds to {} for mmlab eval...".format(test_out_file))
        np.save(test_out_file, test_data.cpu().data.numpy())    
    
    # concat all preds and gts
    backbone_preds = np.concatenate(backbone_preds, axis=0)
    corr_preds = np.concatenate(corr_preds, axis=0)
    gts = np.concatenate(gts, axis=0)
    # only keep eval joints
    backbone_preds = backbone_preds[:, config.EVAL_JOINTS]
    corr_preds = corr_preds[:, config.EVAL_JOINTS]
    gts = gts[:, config.EVAL_JOINTS]
    
    backbone_preds *= 1000
    corr_preds *= 1000
    gts *= 1000
    # get metrics
    backbone_mpjpe = np.sqrt(((backbone_preds - gts)**2).sum(2)).mean()
    corr_mpjpe = np.sqrt(((corr_preds - gts)**2).sum(2)).mean()

    backbone_pa_mpjpe = np.zeros((gts.shape[0], len(config.EVAL_JOINTS)))
    corr_pa_mpjpe = np.zeros((gts.shape[0], len(config.EVAL_JOINTS)))
    backbone_err = np.zeros((gts.shape[0], 3))
    corr_err = np.zeros((gts.shape[0], 3))
    for i, (backbone_pred, corr_pred, gt) in enumerate(zip(backbone_preds, corr_preds, gts)):
        backbone_pred_pa = pose_processing.compute_similarity_transform(backbone_pred.copy(), gt.copy())
        corr_pred_pa = pose_processing.compute_similarity_transform(corr_pred.copy(), gt.copy())

        backbone_pa_mpjpe[i] = np.sqrt(((backbone_pred_pa - gt)**2).sum(1))
        corr_pa_mpjpe[i] = np.sqrt(((corr_pred_pa - gt)**2).sum(1))

        backbone_err[i,0] = np.abs(backbone_pred[:,0] - gt[:,0]).mean()
        backbone_err[i,1] = np.abs(backbone_pred[:,1] - gt[:,1]).mean()
        backbone_err[i,2] = np.abs(backbone_pred[:,2] - gt[:,2]).mean()
        corr_err[i,0] = np.abs(corr_pred[:,0] - gt[:,0]).mean()
        corr_err[i,1] = np.abs(corr_pred[:,1] - gt[:,1]).mean()
        corr_err[i,2] = np.abs(corr_pred[:,2] - gt[:,2]).mean()

    backbone_pa_mpjpe = backbone_pa_mpjpe.mean()
    backbone_err = backbone_err.mean(0)
    corr_pa_mpjpe = corr_pa_mpjpe.mean()
    corr_err = corr_err.mean(0)

    eval_summary_json = {
        'corrected': {
            'PA-MPJPE': corr_pa_mpjpe,
            'MPJPE': corr_mpjpe,
            'x': corr_err[0],
            'y': corr_err[1],
            'z': corr_err[2],
        },
        'backbone': {
            'PA-MPJPE': backbone_pa_mpjpe,
            'MPJPE': backbone_mpjpe,
            'x': backbone_err[0],
            'y': backbone_err[1],
            'z': backbone_err[2],
        }
    }
    return eval_summary_json