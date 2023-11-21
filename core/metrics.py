import torch
import numpy as np

import utils.pose_processing as pose_processing

def get_PA_MPJPE(preds, gts, config):
    pa_mpjpe = np.zeros((gts.shape[0], len(config.EVAL_JOINTS)))
    err = np.zeros((gts.shape[0], 3))
    for i, (pred, gt) in enumerate(zip(preds, gts)):
        pred_pa = pose_processing.compute_similarity_transform(pred.copy(), gt.copy(), rescale=True)
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

def get_top_corr(backbone_preds, corr_preds, gts, img_idxs, kpts, n=10):
    ''' 
    Get img_ids, gt poses, and corrected poses of the n best 
    correction improvements on specified kpts over backbone predictions
    '''

    # look at top corrections for each individual target kpt:
    backbone_kpts_mpjpe = np.sqrt(((backbone_preds[:,kpts] - gts[:,kpts])**2).sum(2))
    corr_kpts_mpjpe = np.sqrt(((corr_preds[:,kpts] - gts[:,kpts])**2).sum(2))
    diffs = corr_kpts_mpjpe - backbone_kpts_mpjpe

    # get top n prediction with largest single keypoint correction (smallest value)
    num = n #*100
    # top_lh_idxs = np.argsort(-diffs[:,0])[-num:]
    # top_rh_idxs = np.argsort(-diffs[:,1])[-num:]
    # top_lf_idxs = np.argsort(-diffs[:,2])[-num:]
    # top_rf_idxs = np.argsort(-diffs[:,3])[-num:]

    # top_idxs = np.argsort(-np.vstack([diffs[:,0], diffs[:,1], diffs[:,2], diffs[:,3]]).max(axis=0))[-num:]
    # top_idxs = top_lh_idxs

    # get MPJPE of backbone and corrected predictions
    backbone_mpjpe = backbone_kpts_mpjpe.mean(1)
    corr_mpjpe = corr_kpts_mpjpe.mean(1)

    # get top n corrected predictions
    top_idxs = np.argpartition(corr_mpjpe - backbone_mpjpe, n)[:n*100]

    # top_idxs = np.argpartition(corr_mpjpe, n)[:n*100]

    # get img_ids, gt poses, and corrected poses of the top n corrected predictions
    img_ids = img_idxs[top_idxs]
    # img_ids will have many similar values due to the data, we want to keep representative 
    # ids, so we enforce a minimum distance between ids
    img_ids = np.unique(np.round(img_ids, -2), return_index=True)[1][:n]

    # replace top_idxs with top_idxs corresponding to the selected img_ids
    top_idxs = top_idxs[img_ids]
    

    img_ids = img_idxs[top_idxs]
    gt = gts[top_idxs]
    preds = backbone_preds[top_idxs]
    corr = corr_preds[top_idxs]
    return img_ids, gt, preds, corr

