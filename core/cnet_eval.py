import torch
import numpy as np
from types import SimpleNamespace
from tqdm import tqdm
import torch.nn as nn
import os

import utils.errors as errors

def cnet_rcnet_self_consistency(config, backbone_pred, Rcnet_pred, err_scale=1000):
    '''
    CNet self consistency loss for TTT

    computes L2(proximal kpts predicted by Rcnet from cnet preds, orignal backbone preds)
    '''

    # loss = nn.MSELoss(backbone_pred[:, config.proximal_kpts, :], Rcnet_pred[:, config.proximal_kpts, :])
    loss = torch.square((backbone_pred[:,config.proximal_kpts,:] - Rcnet_pred[:,config.proximal_kpts,:])*err_scale).mean()

    return loss

def unpack_test_data(data, m, use_data_file, flip_test, config):
    '''
    Unpack CNet test sample 
    '''
    if use_data_file:
        labels = {}
        output = SimpleNamespace()
        output.pred_xyz_jts_17 = data[0][:,0].to(config.device)
        labels['target_xyz_17'] = data[0][:,1].to(config.device)
        img_ids = data[0][:,2,:1].to(config.device)
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

    return output, labels, img_ids

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

    flip_test = True

    def set_bn_eval(module):
        ''' Batch Norm layer freezing '''
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()

    for it, data in enumerate(tqdm(gt_eval_loader, dynamic_ncols=True)):
        output, labels, img_ids = unpack_test_data(data, m, use_data_file, flip_test, config)
        if test_cnet:
            backbone_pred = output.pred_xyz_jts_17
            backbone_pred = backbone_pred.reshape(-1, 17, 3)

            if test_adapt:
                # THIS IS SUPER INNEFICIENT, AND WASTES MY TIME :(
                cnet.load_cnets(print_str=False)
                cnet.cnet.train()
                # Setup Optimizer
                cnet.cnet.apply(set_bn_eval)
                optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, cnet.cnet.parameters()),
                                                lr=config.test_adapt_lr)
                cnet_in = backbone_pred.detach().clone()
                for i in range(config.adapt_steps):
                    corrected_pred = cnet(cnet_in)
                    loss = cnet_rcnet_self_consistency(config, backbone_pred, R_cnet(corrected_pred))
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
            if config.use_cnet:
                with torch.no_grad():
                    # get corrected pred, with adjusted cnet
                    cnet_in = backbone_pred
                    corrected_pred = cnet(cnet_in)
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