import torch
import numpy as np

import utils.pose_processing as pose_processing

def get_PA_MPJPE(preds, gts, config):
    pa_mpjpe = np.zeros((gts.shape[0], len(config.EVAL_JOINTS)))
    err = np.zeros((gts.shape[0], 3))
    for i, (pred, gt) in enumerate(zip(preds, gts)):
        pred_pa = pose_processing.compute_similarity_transform(pred.copy(), gt.copy())
        pa_mpjpe[i] = np.sqrt(((pred_pa - gt)**2).sum(1))

        err[i,0] = np.abs(pred[:,0] - gt[:,0]).mean()
        err[i,1] = np.abs(pred[:,1] - gt[:,1]).mean()
        err[i,2] = np.abs(pred[:,2] - gt[:,2]).mean()
    return pa_mpjpe, err

def get_P1_P2(backbone_preds, corr_preds, gts, 
              corr_idxs,
              num_tails, config):
    '''
    '''
    backbone_pa_mpjpe, backbone_err = get_PA_MPJPE(backbone_preds, gts, config)
    corr_pa_mpjpe, corr_err = get_PA_MPJPE(corr_preds, gts, config)
    
    backbone_pa_mpjpe = backbone_pa_mpjpe.mean(1)
    bb_pa_mpjpe_tail_idxs = [np.argpartition(backbone_pa_mpjpe, -num_tails[i])[-num_tails[i]:] for i in range(len(num_tails))]
    bb_pa_mpjpe_tails = [backbone_pa_mpjpe[bb_pa_mpjpe_tail_idxs[i]].mean() for i in range(len(num_tails))]
    bb_pa_mpjpe_all = backbone_pa_mpjpe.mean()

    corr_pa_mpjpe = corr_pa_mpjpe.mean(1)
    corr_pa_mpjpe_tails = [corr_pa_mpjpe[bb_pa_mpjpe_tail_idxs[i]].mean() for i in range(len(num_tails))]

    # MPJPE
    backbone_mpjpe = np.sqrt(((backbone_preds - gts)**2).sum(2)).mean(1)
    bb_mpjpe_tail_idxs = [np.argpartition(backbone_mpjpe, -num_tails[i])[-num_tails[i]:] for i in range(len(num_tails))]
    bb_mpjpe_tails = [backbone_mpjpe[bb_mpjpe_tail_idxs[i]].mean() for i in range(len(num_tails))]
    bb_mpjpe_all = backbone_mpjpe.mean()

    corr_mpjpe = np.sqrt(((corr_preds - gts)**2).sum(2)).mean(1)
    corr_mpjpe_tails = [corr_mpjpe[bb_mpjpe_tail_idxs[i]].mean() for i in range(len(num_tails))]

    # only compute corrected metrics where we did not keep bb preds (idx not in keep_bb_idxs)
    if corr_idxs is None:
        att_bb_pa_mpjpe = bb_pa_mpjpe_all
        att_bb_mpjpe = bb_mpjpe_all
        att_bb_err = backbone_err.mean(0)
                
        att_corr_pa_mpjpe = corr_pa_mpjpe.mean()
        att_corr_mpjpe = corr_mpjpe.mean()
        att_corr_err = corr_err.mean(0)
    else:
        att_bb_pa_mpjpe = backbone_pa_mpjpe[corr_idxs].mean()
        att_bb_mpjpe = backbone_mpjpe[corr_idxs].mean()
        att_bb_err = backbone_err[corr_idxs].mean(0)

        att_corr_pa_mpjpe = corr_pa_mpjpe[corr_idxs].mean()
        att_corr_mpjpe = corr_mpjpe[corr_idxs].mean()
        att_corr_err = corr_err[corr_idxs].mean(0)

        # att_rel_pa_mpjpe = att_corr_pa_mpjpe - att_bb_pa_mpjpe
        # att_rel_mpjpe = att_corr_mpjpe - att_bb_mpjpe
        # att_rel_err = att_corr_err - att_backbone_err

    corr_pa_mpjpe_all = corr_pa_mpjpe.mean()
    corr_mpjpe_all = corr_mpjpe.mean()
    
    # X/Y/Z errs
    backbone_err = backbone_err.mean(0)
    corr_err = corr_err.mean(0)

    corr = [corr_pa_mpjpe_all, corr_pa_mpjpe_tails, corr_mpjpe_all, corr_mpjpe_tails, corr_err]
    attempted = [att_corr_mpjpe, att_corr_pa_mpjpe, att_corr_err,
                     att_bb_mpjpe, att_bb_pa_mpjpe, att_bb_err]
    bb = [bb_pa_mpjpe_all, bb_pa_mpjpe_tails, bb_mpjpe_all, bb_mpjpe_tails, backbone_err]
    return corr, bb, attempted

