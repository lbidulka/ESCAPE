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
from core.cnet_dataset import cnet_pose_dataset, hflip_keypoints

from core import cnet_data, metrics


def cnet_TTT_loss(config, backbone_pred, Rcnet_pred, corrected_pred, poses_2d,):
    '''
    CNet loss for TTT, based on config

    consistency: computes MSE(proximal kpts predicted by Rcnet from cnet preds, orignal backbone preds)
    '''

    if config.TTT_loss == 'reproj_2d':
        loss = errors.loss_weighted_rep_no_scale(poses_2d[:,config.cnet_targets], 
                                                 corrected_pred[:,config.cnet_targets], 
                                                 sum_kpts=True, num_joints=len(config.cnet_targets)).sum()
        loss *= config.TTT_errscale
    elif config.TTT_loss == 'consistency':
        loss = (backbone_pred[:,config.rcnet_targets] - Rcnet_pred[:,config.rcnet_targets])
        loss = torch.square(loss*config.TTT_errscale).mean()
    else:
        raise NotImplementedError
    return loss

def unpack_test_data(data, config,):
    '''
    Unpack CNet test sample 
    '''
    labels = {}
    output = SimpleNamespace()
    output.pred_xyz_jts_17 = data[:,0].to(config.device)
    labels['target_xyz_17'] = data[:,1].to(config.device)
    # if theres no img_ids, add dummies
    if data.shape[1] == 2:
        img_ids = torch.ones(data.shape[0], 1) * -1
    else:
        img_ids = data[:,2,:1].to(config.device)
    poses_2d = labels['target_xyz_17'].reshape(labels['target_xyz_17'].shape[0], -1, 3)[:,:,:2] # weak reproj GT 2D

    return output, labels, img_ids, poses_2d

def eval_gt(cnet, R_cnet, config, 
            testset_path=None, backbone_scale=1.0,
            test_adapt=False, subset=None,):
    '''
    '''
    if test_adapt:
        batch_size = 1
    else:
        batch_size = 4096
    # Data/Setup
    test_file = testset_path
    test_data = torch.from_numpy(np.load(test_file)).float().permute(1,0,2)
    if subset is None: 
        subset = config.test_eval_subset
    test_data = test_data[subset, :]
    test_data *= backbone_scale
    gt_eval_dataset = cnet_pose_dataset(test_data, datasets=config.testsets, 
                                        backbones=config.test_backbones,
                                        config=config,)
    gt_eval_loader = torch.utils.data.DataLoader(gt_eval_dataset, batch_size=batch_size, shuffle=False,
                                                drop_last=False)
    kpt_pred = {}
    kpt_all_pred = {}
    cnet.eval()
    R_cnet.eval()

    def set_bn_eval(module):
        ''' Batch Norm layer freezing '''
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()

    backbone_preds = []
    corr_preds = []
    corr_idxs = []
    img_idxs = []
    gts = []
    losses = []
    for it, data in enumerate(tqdm(gt_eval_loader, dynamic_ncols=True)):
        output, labels, img_ids, poses_2d = unpack_test_data(data, config)
        backbone_pred = output.pred_xyz_jts_17.reshape(output.pred_xyz_jts_17.shape[0], -1, 3)
        if config.use_cnet:
            corrected_pred_init = None            
            
            # Only do TTT if sample is hard sample
            if config.test_adapt and config.TTT_e_thresh:
                # dont correct samples with energy above threshold
                # E = cnet._energy(backbone_pred.detach().clone())
                E = cnet._energy(backbone_pred - backbone_pred[:,:1])
                dont_TTT_idxs = E > config.energy_thresh
                TTT_idxs = ~dont_TTT_idxs
                TTT_e_thresh_condition = TTT_idxs.sum() > 0
            else:
                TTT_e_thresh_condition = True

            if test_adapt and TTT_e_thresh_condition:
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
                    corrected_pred, corr_idx = cnet(cnet_in, ret_corr_idxs=True,)

                    # only do TTT if there are some corrected samples
                    if corr_idx.sum() > 0:
                        Rcnet_pred = R_cnet(corrected_pred)
                        loss = cnet_TTT_loss(config, backbone_pred, Rcnet_pred, corrected_pred, poses_2d)
                        loss.backward()
                        optimizer.step()

                        gt_3d = labels['target_xyz_17'].reshape(labels['target_xyz_17'].shape[0], -1, 3)
                        gt_mse = torch.square((corrected_pred - gt_3d)*config.TTT_errscale).mean()

                        # log true mse between cnet predicted correction, and GT correction
                        cnet_pred_err = (backbone_pred - corrected_pred)[:,cnet.target_kpts]
                        cnet_gt_err = (backbone_pred - gt_3d)[:,cnet.target_kpts] #* config.TTT_errscale
                        cnet_mse = torch.square((cnet_pred_err - cnet_gt_err)*cnet.config.err_scale).mean()

                        losses.append([loss.item(), gt_mse.item(), cnet_mse.item()])

            with torch.no_grad():
                # get corrected pred, with adjusted cnet
                cnet_in = backbone_pred
                corrected_pred, batch_corr_idxs = cnet(cnet_in, ret_corr_idxs=True)
                
                # Log attempted TTT samples if using energy threshold for TTT
                if config.test_adapt and config.TTT_e_thresh:
                    corr_idxs.append(TTT_idxs.cpu().numpy())
                else:
                    corr_idxs.append(batch_corr_idxs)

                # Only correct specified CNet corr dims
                if len(config.cnet_dont_corr_dims) > 0:
                    corrected_pred_init = backbone_pred.detach().clone()
                    corrected_pred[:, :, config.cnet_dont_corr_dims] = corrected_pred_init[:, :, config.cnet_dont_corr_dims]
                # Only use TTT improved preds for specified dims
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
        img_idxs.append(img_ids.detach().cpu().numpy())
        for i in range(output.pred_xyz_jts_17.shape[0]):
            pred_xyz_jts_17 = output.pred_xyz_jts_17[i].data.cpu().numpy()
            kpt_pred[int(img_ids[i])] = {
                'xyz_17': pred_xyz_jts_17.reshape(-1, 3),
            }
        kpt_all_pred.update(kpt_pred)

    # Investigation of TTT loss
    if test_adapt:
        TTT_losses = np.array(losses)
        # save losses
        TTT_losses_outpath = '../../outputs/TTT_losses/'
        dataset_name = testset_path.split('/')[-2]
        # if dataset_name != 'PW3D':
        TTT_losses_outpath += dataset_name + '_'
        backbone_name = testset_path.split('/')[-1].split('.')[-2]
        TTT_losses_outpath +=  '_'.join([backbone_name, config.TTT_loss, 'losses.npy'])
        np.save(TTT_losses_outpath, TTT_losses)
    
    # concat all preds and gts
    backbone_preds = np.concatenate(backbone_preds, axis=0)
    corr_preds = np.concatenate(corr_preds, axis=0)
    gts = np.concatenate(gts, axis=0)
    img_idxs = np.concatenate(img_idxs, axis=0)
    # only keep eval joints
    backbone_preds_eval = backbone_preds[:, config.EVAL_JOINTS]
    corr_preds_eval = corr_preds[:, config.EVAL_JOINTS]
    gts_eval = gts[:, config.EVAL_JOINTS]
    
    backbone_preds_eval *= 1000
    corr_preds_eval *= 1000
    gts_eval *= 1000

    corr_idxs = np.concatenate(corr_idxs, axis=0)

    # Save BB Energy scores, gt MSEs, and gt MPJPEs for plotting
    bb_e_scores = torch.logsumexp(torch.tensor(backbone_preds_eval.reshape(backbone_preds_eval.shape[0], -1)), dim=-1)
    gt_mses = np.square((backbone_preds_eval - gts_eval)).mean((1,2)).reshape(-1,1)
    gt_mpjpe = np.sqrt(((backbone_preds_eval - gts_eval)**2).sum(2)).mean(1).reshape(-1,1)
    energies_losses = np.concatenate([bb_e_scores.reshape(-1,1), gt_mses, gt_mpjpe], axis=1)

    energies_outpath = '../../outputs/energies/'
    dataset_name = testset_path.split('/')[-2]
    energies_outpath += dataset_name + '_'
    backbone_name = testset_path.split('/')[-1].split('.')[-2]
    energies_outpath +=  '_'.join([backbone_name, 'energies.npy'])
    np.save(energies_outpath, energies_losses)

    # Save CNet Energy scores and gt MSEs for plotting
    cnet_preds = (corr_preds[:, config.cnet_targets] - backbone_preds[:, config.cnet_targets]) * 1000
    cnet_labels = (backbone_preds[:, config.cnet_targets] - gts[:, config.cnet_targets]) * 1000

    cnet_e_scores = torch.logsumexp(torch.tensor(cnet_preds.reshape(cnet_preds.shape[0], -1)), dim=-1)
    cnet_mses = np.square((cnet_labels - cnet_preds)).mean((1,2)).reshape(-1,1)
    cnet_mpjpe = np.sqrt(((cnet_labels - cnet_preds)**2).sum(2)).mean(1).reshape(-1,1)
    energies_losses = np.concatenate([cnet_e_scores.reshape(-1,1), cnet_mses, cnet_mpjpe], axis=1)

    energies_outpath = '../../outputs/energies/'
    dataset_name = testset_path.split('/')[-2]
    energies_outpath += dataset_name + '_'
    backbone_name = testset_path.split('/')[-1].split('.')[-2]
    energies_outpath +=  '_'.join([backbone_name, 'energies_cnet.npy'])
    np.save(energies_outpath, energies_losses)
    
    # get top corrections for qualitative investigation
    top_img_ids, top_gts, top_preds, top_corrs = metrics.get_top_corr(backbone_preds, corr_preds, gts, 
                                                           img_idxs, config.cnet_targets, n=50)
    
    # if dataset_name == 'PW3D':
    #     # add specific img id to the top corrections
    #     # want to add sample with img id = 22408
    #     idx_22408 = np.where(img_idxs == 22408)[0][0]
    #     top_img_ids = np.concatenate([np.array([22408])[None,:], top_img_ids])
    #     top_gts = np.concatenate([gts[idx_22408:idx_22408+1], top_gts])
    #     top_preds = np.concatenate([backbone_preds[idx_22408:idx_22408+1], top_preds])
    #     top_corrs = np.concatenate([corr_preds[idx_22408:idx_22408+1], top_corrs])

    # save top corrections
    top_corr_outpath = '../../outputs/qualitative/'
    top_corr_outpath += dataset_name + '_'
    top_corr_outpath +=  '_'.join([backbone_name, 'top_corr.npy'])
    top_qual_info = np.array([np.ones_like(top_gts)*top_img_ids.reshape(-1,1,1), 
                              top_gts, top_preds, top_corrs])
    np.save(top_corr_outpath, top_qual_info)


    # get metrics
    num_tails = [
        int(0.05*gts_eval.shape[0]),
        int(0.10*gts_eval.shape[0]),
        int(0.25*gts_eval.shape[0]),
    ]
    corr_res, bb_res, att_res = metrics.get_P1_P2(backbone_preds_eval, corr_preds_eval, gts_eval, 
                                                       corr_idxs,
                                                       num_tails, config)
    corr_pa_mpjpe_all, corr_pa_mpjpe_tails, corr_mpjpe_all, corr_mpjpe_tails, corr_err = corr_res
    bb_pa_mpjpe_all, bb_pa_mpjpe_tails, bb_mpjpe_all, bb_mpjpe_tails, backbone_err = bb_res
    att_corr_mpjpe, att_corr_pa_mpjpe, att_corr_err, att_bb_mpjpe, att_bb_pa_mpjpe, att_bb_err = att_res       

    eval_summary_json = {
        'corrected': {
            'PA-MPJPE': corr_pa_mpjpe_all,
            '(P1 ^10%)': corr_pa_mpjpe_tails[1],
            'MPJPE': corr_mpjpe_all,
            '(P2 ^10%)': corr_mpjpe_tails[1],
            'x': corr_err[0],
            'y': corr_err[1],
            'z': corr_err[2],
        },
        'backbone': {
            'PA-MPJPE': bb_pa_mpjpe_all,
            '(P1 ^10%)': bb_pa_mpjpe_tails[1],
            'MPJPE': bb_mpjpe_all,
            '(P2 ^10%)': bb_mpjpe_tails[1],
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

    # Add target kpt specific errors, if required
    if config.include_target_kpt_errs:
        bb_mpjpe = np.sqrt((((backbone_preds - gts)*1000)**2).sum(2))
        corr_mppe = np.sqrt((((corr_preds - gts)*1000)**2).sum(2))

        bb_mpjpe_tail_idxs = [np.argpartition(bb_mpjpe.mean(1), -num_tails[i])[-num_tails[i]:] for i in range(len(num_tails))]
        bb_mpjpe_tails = [bb_mpjpe[bb_mpjpe_tail_idxs[i]][:,cnet.target_kpts].mean(0) for i in range(len(num_tails))]
        corr_mpjpe_tails = [corr_mppe[bb_mpjpe_tail_idxs[i]][:,cnet.target_kpts].mean(0) for i in range(len(num_tails))]

        # errors for each of the target kpts 
        corr_target_kpt_errs =  corr_mppe[:,cnet.target_kpts]
        bb_target_kpt_errs =  bb_mpjpe[:,cnet.target_kpts]
        corr_target_kpt_errs = corr_target_kpt_errs.mean(0)
        bb_target_kpt_errs = bb_target_kpt_errs.mean(0)

        # average feet kpts and hand kpts together
        corr_target_kpt_errs = np.concatenate([corr_target_kpt_errs[:2].mean(0, keepdims=True), 
                                               corr_target_kpt_errs[2:].mean(0, keepdims=True)])
        bb_target_kpt_errs = np.concatenate([bb_target_kpt_errs[:2].mean(0, keepdims=True),
                                                bb_target_kpt_errs[2:].mean(0, keepdims=True)])
        corr_mpjpe_tails = [np.concatenate([corr_mpjpe_tails[i][:2].mean(0, keepdims=True),
                                            corr_mpjpe_tails[i][2:].mean(0, keepdims=True)]) for i in range(len(num_tails))]
        bb_mpjpe_tails = [np.concatenate([bb_mpjpe_tails[i][:2].mean(0, keepdims=True),
                                            bb_mpjpe_tails[i][2:].mean(0, keepdims=True)]) for i in range(len(num_tails))]
        
        # add to results
        titles = ['feet', 'hands']
        eval_summary_json['corrected'].update(dict(zip(titles, corr_target_kpt_errs)))
        eval_summary_json['corrected'].update(dict(zip([f'({kpt} ^10%)' for kpt in titles], corr_mpjpe_tails[1])))
        eval_summary_json['backbone'].update(dict(zip(titles, bb_target_kpt_errs)))
        eval_summary_json['backbone'].update(dict(zip([f'({kpt} ^10%)' for kpt in titles], bb_mpjpe_tails[1])))
    
    
    return eval_summary_json
    