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

from core import cnet_data, metrics


def cnet_TTT_loss(config, backbone_pred, Rcnet_pred, corrected_pred, poses_2d,):
    '''
    CNet loss for TTT, based on config

    consistency: computes L2(proximal kpts predicted by Rcnet from cnet preds, orignal backbone preds)
    '''

    if config.TTT_loss == 'reproj_2d':
        loss = errors.loss_weighted_rep_no_scale(poses_2d[:,config.cnet_targets], 
                                                 corrected_pred[:,config.cnet_targets], 
                                                 sum_kpts=True, num_joints=len(config.cnet_targets)).sum()
        loss *= config.TTT_errscale
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
        # if theres no img_ids, add dummies
        if data[0].shape[1] == 2:
            img_ids = torch.ones(data[0].shape[0], 1) * -1
        else:
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
            use_data_file=False, agora_out=False,
            subset=None,):
    '''
    '''
    if test_adapt:
        batch_size = 1 # 1
    else:
        batch_size = 4096
    # Data/Setup
    if use_data_file:
        test_file = testset_path
        test_data = torch.from_numpy(np.load(test_file)).float().permute(1,0,2)
        if subset is None: 
            subset = config.test_eval_subset
        test_data = test_data[subset, :]
        test_data *= backbone_scale
        gt_eval_dataset = torch.utils.data.TensorDataset(test_data)
    gt_eval_loader = torch.utils.data.DataLoader(gt_eval_dataset, batch_size=batch_size, shuffle=False, 
                                                #  num_workers=2, drop_last=False, pin_memory=True)
                                                drop_last=False)#, pin_memory=True)
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
    corr_idxs = []
    gts = []
    losses = []
    for it, data in enumerate(tqdm(gt_eval_loader, dynamic_ncols=True)):
        output, labels, img_ids, poses_2d = unpack_test_data(data, m, model_2d, use_data_file, config)
        backbone_pred = output.pred_xyz_jts_17.reshape(output.pred_xyz_jts_17.shape[0], -1, 3)
        if test_cnet:
            if config.use_cnet:
                corrected_pred_init = None
                if test_adapt:
                    # THIS IS SUPER INNEFICIENT, AND WASTES MY TIME :(
                    cnet.load_cnets(print_str=False)
                    if len(config.split_corr_dims) > 0:
                        cnet.cnet.eval()
                        corrected_pred_init = cnet(backbone_pred.detach().clone())
                    cnet.cnet.train()
                    # Setup Optimizer
                    cnet.cnet.apply(set_bn_eval)
                    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, cnet.cnet.parameters()),
                                                    lr=config.test_adapt_lr)
                    cnet_in = backbone_pred.detach().clone()
                    for i in range(config.adapt_steps):
                        optimizer.zero_grad()
                        corrected_pred, corr_idx = cnet(cnet_in, ret_corr_idxs=True)

                        # only do TTT if there are any corrected samples
                        if corr_idx.sum() > 0:
                            Rcnet_pred = R_cnet(corrected_pred)
                            loss = cnet_TTT_loss(config, backbone_pred, Rcnet_pred, corrected_pred, poses_2d)
                            loss.backward()
                            optimizer.step()

                            gt_3d = labels['target_xyz_17'].reshape(labels['target_xyz_17'].shape[0], -1, 3)
                            # gt_mse = torch.square((backbone_pred[:,config.distal_kpts,:] - gt_3d[:,config.distal_kpts])*config.TTT_errscale).mean()
                            gt_mse = torch.square((corrected_pred - gt_3d)*config.TTT_errscale).mean()
                            losses.append([loss.item(), gt_mse.item()])

                with torch.no_grad():
                    # get corrected pred, with adjusted cnet
                    cnet_in = backbone_pred
                    corrected_pred, batch_corr_idxs = cnet(cnet_in, ret_corr_idxs=True)
                    corr_idxs.append(batch_corr_idxs)
                    # Only correct specified CNet corr dims
                    if len(config.cnet_dont_corr_dims) > 0:
                        corrected_pred_init = backbone_pred.detach().clone()
                        corrected_pred[:, :, config.cnet_dont_corr_dims] = corrected_pred_init[:, :, config.cnet_dont_corr_dims]
                    # Only use TTT improved preds for specified dims
                    # if config.split_corr_dim_trick:
                    if len(config.split_corr_dims) > 0:
                        if corrected_pred_init is None:
                            corrected_pred_init = backbone_pred.detach().clone()

                        corrected_pred[:, :, config.split_corr_dims] = corrected_pred_init[:, :, config.split_corr_dims]

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
        TTT_losses = np.array(losses)
        # save losses
        TTT_losses_outpath = '../../outputs/TTT_losses/'
        dataset_name = testset_path.split('/')[-2]
        if dataset_name != 'PW3D':
            TTT_losses_outpath += dataset_name + '_'
        backbone_name = testset_path.split('/')[-1].split('.')[-2]
        TTT_losses_outpath +=  '_'.join([backbone_name, config.TTT_loss, 'losses.npy'])
        np.save(TTT_losses_outpath, TTT_losses)

    # save corrected preds for agora eval
    if agora_out:
        agora_corr_preds = np.concatenate(corr_preds, axis=0)
        # order was scrambled according to config.test_eval_subset, so we need to reverse it 
        # (since agora is expecting the original order)
        reverse_idxs = [np.where(np.array(config.test_eval_subset) == i)[0][0] for i in range(len(config.test_eval_subset))]
        agora_corr_preds = agora_corr_preds[reverse_idxs]

        # save npy file
        if test_adapt:
            test_out_file = testset_path.replace('_test.npy', '_corrected_+TTT.npy')
        else:
            test_out_file = testset_path.replace('_test.npy', '_corrected.npy')
        print("Saving corrected preds to {} for AGORA eval...".format(test_out_file))
        np.save(test_out_file, agora_corr_preds)
    
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

    corr_idxs = np.concatenate(corr_idxs, axis=0)

    # Save Energy scores and gt MSEs for plotting
    if config.use_cnet_energy:
        bb_e_scores = torch.logsumexp(torch.tensor(backbone_preds.reshape(backbone_preds.shape[0], -1)), dim=-1)
        gt_mses = np.square((backbone_preds - gts)).mean((1,2)).reshape(-1,1)
        energies_losses = np.concatenate([bb_e_scores.reshape(-1,1), gt_mses], axis=1)

        energies_outpath = '../../outputs/energies/'
        dataset_name = testset_path.split('/')[-2]
        if dataset_name != 'PW3D':
            energies_outpath += dataset_name + '_'
        backbone_name = testset_path.split('/')[-1].split('.')[-2]
        energies_outpath +=  '_'.join([backbone_name, 'energies.npy'])
        np.save(energies_outpath, energies_losses)
    
    # get metrics
    num_tails = [
        int(0.05*gts.shape[0]),
        int(0.10*gts.shape[0]),
        int(0.25*gts.shape[0]),
    ]
    corr_res, bb_res, att_res = metrics.get_P1_P2(backbone_preds, corr_preds, gts, 
                                                       corr_idxs,
                                                       num_tails, config)
    corr_pa_mpjpe_all, corr_pa_mpjpe_tails, corr_mpjpe_all, corr_mpjpe_tails, corr_err = corr_res
    bb_pa_mpjpe_all, bb_pa_mpjpe_tails, bb_mpjpe_all, bb_mpjpe_tails, backbone_err = bb_res
    att_corr_mpjpe, att_corr_pa_mpjpe, att_corr_err, att_bb_mpjpe, att_bb_pa_mpjpe, att_bb_err = att_res

    eval_summary_json = {
        'corrected': {
            'PA-MPJPE': corr_pa_mpjpe_all,
            '(P1 ^25%)': corr_pa_mpjpe_tails[2],
            '(P1 ^10%)': corr_pa_mpjpe_tails[1],
            '(P1 ^5%)': corr_pa_mpjpe_tails[0],
            'MPJPE': corr_mpjpe_all,
            '(P2 ^25%)': corr_mpjpe_tails[2],
            '(P2 ^10%)': corr_mpjpe_tails[1],
            '(P2 ^5%)': corr_mpjpe_tails[0],
            'x': corr_err[0],
            'y': corr_err[1],
            'z': corr_err[2],
        },
        'backbone': {
            'PA-MPJPE': bb_pa_mpjpe_all,
            '(P1 ^25%)': bb_pa_mpjpe_tails[2],
            '(P1 ^10%)': bb_pa_mpjpe_tails[1],
            '(P1 ^5%)': bb_pa_mpjpe_tails[0],
            'MPJPE': bb_mpjpe_all,
            '(P2 ^25%)': bb_mpjpe_tails[2],
            '(P2 ^10%)': bb_mpjpe_tails[1],
            '(P2 ^5%)': bb_mpjpe_tails[0],
            'x': backbone_err[0],
            'y': backbone_err[1],
            'z': backbone_err[2],
        },
        'attempted_backbone': {
            'PA-MPJPE': att_bb_pa_mpjpe,
            'MPJPE': att_bb_mpjpe,
            'x': att_bb_err[0],
            'y': att_bb_err[1],
            'z': att_bb_err[2],
        },
        'attempted_corr': {
            'PA-MPJPE': att_corr_pa_mpjpe,
            'MPJPE': att_corr_mpjpe,
            'x': att_corr_err[0],
            'y': att_corr_err[1],
            'z': att_corr_err[2],
            '(num_samples)': int(corr_idxs.sum().item()),
        },
    }
    return eval_summary_json