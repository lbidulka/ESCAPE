import json
import os
import os.path
from abc import ABCMeta
from collections import OrderedDict
from typing import List, Optional, Union

import mmcv
import numpy as np
import torch

from mmhuman3d.core.conventions.keypoints_mapping import get_mapping
from mmhuman3d.core.evaluation import (
    keypoint_3d_auc,
    keypoint_3d_pck,
    keypoint_mpjpe,
    vertice_pve,
)
from mmhuman3d.data.data_structures.human_data import HumanData
from mmhuman3d.models.body_models.builder import build_body_model
from mmhuman3d.utils.demo_utils import box2cs, xyxy2xywh
from .base_dataset import BaseDataset
from .human_hybrik_dataset import HybrIKHumanImageDataset
from .builder import DATASETS


@DATASETS.register_module()
class CUSTOM_HybrIKHumanImageDataset(HybrIKHumanImageDataset):
    """Dataset for HybrIK training. The dataset loads raw features and apply
    specified transforms to return a dict containing the image tensors and
    other information.

    Args:

        data_prefix (str): Path to a directory where preprocessed datasets are
         held.
        pipeline (list[dict | callable]): A sequence of data transforms.
        dataset_name (str): accepted names include 'h36m', 'pw3d',
         'mpi_inf_3dhp', 'coco'
        ann_file (str): Name of annotation file.
        test_mode (bool): Store True when building test dataset.
         Default: False.
    """
    def get_results_and_human_gts(self,
                            outputs: list,
                            res_folder: str,
                            save_results: bool = True,):
        '''
        Builds and saves the 3D keypoints predictions and gts.

        Args:
            outputs (list): results from model inference.
            res_folder (str): path to store results.
            save_results (bool): whether to save results to res_folder.
        Returns:
            dict:
                A dict of preds, gts, & img ids in mm with h36m kpt format.
        '''
        res_file = os.path.join(res_folder, 'result_keypoints_gts.json')
        res_dict = {}
        for out in outputs:
            target_id = out['image_idx']
            batch_size = len(out['xyz_17'])
            for i in range(batch_size):
                res_dict[int(target_id[i])] = dict(
                    keypoints=out['xyz_17'][i],
                    poses=out['smpl_pose'][i],
                    betas=out['smpl_beta'][i],
                    ids=target_id[i].item(),
                )

        keypoints, poses, betas, ids = [], [], [], []
        for i in list(res_dict.keys()):
            keypoints.append(res_dict[i]['keypoints'])
            poses.append(res_dict[i]['poses'])
            betas.append(res_dict[i]['betas'])
            ids.append(res_dict[i]['ids'])

        res = dict(keypoints=keypoints, poses=poses, betas=betas,)

        """Parse gts."""
        gts = [self.data_infos[id] for id in ids]
        pred_keypoints3d = res['keypoints']
        # (B, 17, 3)
        pred_keypoints3d = np.array(pred_keypoints3d)
        factor, root_idx_17 = 1, 0
        
        _, h36m_idxs, _ = get_mapping('human_data', 'h36m') # Human has 190 kpts
        gt_keypoints3d = np.array(
            [gt['joint_relative_17'][h36m_idxs] for gt in gts])
        if self.dataset_name == 'pw3d':
            factor = 1000
        else:
            raise NotImplementedError

        pred_keypoints3d = pred_keypoints3d * (2000 / factor)
        if self.dataset_name == 'mpi_inf_3dhp':
            raise NotImplementedError
        # root joint alignment
        pred_keypoints3d = (
            pred_keypoints3d -
            pred_keypoints3d[:, None, root_idx_17]) * factor
        gt_keypoints3d = (gt_keypoints3d -
                              gt_keypoints3d[:, None, root_idx_17]) * factor
        
        """Combine"""
        res_outs = dict(preds=pred_keypoints3d, gts=gt_keypoints3d, ids=ids, poses=poses, betas=betas,)
        if save_results:
            print("\nSaving preds, gts, img ids, poses, & betas to: ", res_file)
            mmcv.dump(res_outs, res_file)
        else:
            print("\nNot saving preds, gts, img ids, poses, & betas.")

        return res_outs

    def re_evaluate(self, corr_results, 
                    metric: Optional[Union[str, List[str]]] = 'pa-mpjpe',
                    **kwargs: dict):
        '''Evaluate 3D keypoint results.

        Args:
            corr_results (dict): parsed results from model inference and then correction
            res_folder (str): path to store results.
            metric (Optional[Union[str, List(str)]]):
                the type of metric. Default: 'pa-mpjpe'
            kwargs (dict): other arguments.
        Returns:
            dict:
                A dict of all evaluation results.        
        '''
        metrics = metric if isinstance(metric, list) else [metric]
        for metric in metrics:
            if metric not in self.ALLOWED_METRICS:
                raise ValueError(f'metric {metric} is not supported')



        if self.dataset_name == 'pw3d':
            factor = 1000
        else:
            raise NotImplementedError
        
        res = dict(
            keypoints=corr_results['preds'] / (2000 / factor),
            poses=corr_results['poses'],
            betas=corr_results['betas'],
        )

        name_value_tuples = []
        for _metric in metrics:
            if _metric == 'mpjpe':
                _nv_tuples = self._report_mpjpe(res)
            elif _metric == 'pa-mpjpe':
                _nv_tuples = self._report_mpjpe(res, metric='pa-mpjpe')
            elif _metric == '3dpck':
                _nv_tuples = self._report_3d_pck(res)
            elif _metric == 'pa-3dpck':
                _nv_tuples = self._report_3d_pck(res, metric='pa-3dpck')
            elif _metric == '3dauc':
                _nv_tuples = self._report_3d_auc(res)
            elif _metric == 'pa-3dauc':
                _nv_tuples = self._report_3d_auc(res, metric='pa-3dauc')
            elif _metric == 'pve':
                _nv_tuples = self._report_pve(res)
            else:
                raise NotImplementedError
            name_value_tuples.extend(_nv_tuples)

        name_value = OrderedDict(name_value_tuples)
        return name_value


    def evaluate(self,
                 outputs: list,
                 res_folder: str,
                 metric: Optional[Union[str, List[str]]] = 'pa-mpjpe',
                 **kwargs: dict):
        """Evaluate 3D keypoint results.

        Args:
            outputs (list): results from model inference.
            res_folder (str): path to store results.
            metric (Optional[Union[str, List(str)]]):
                the type of metric. Default: 'pa-mpjpe'
            kwargs (dict): other arguments.
        Returns:
            dict:
                A dict of all evaluation results.
        """
        metrics = metric if isinstance(metric, list) else [metric]
        for metric in metrics:
            if metric not in self.ALLOWED_METRICS:
                raise ValueError(f'metric {metric} is not supported')

        res_file = os.path.join(res_folder, 'result_keypoints.json')

        res_dict = {}
        for out in outputs:
            target_id = out['image_idx']
            batch_size = len(out['xyz_17'])
            for i in range(batch_size):
                res_dict[int(target_id[i])] = dict(
                    keypoints=out['xyz_17'][i],
                    poses=out['smpl_pose'][i],
                    betas=out['smpl_beta'][i],
                )

        keypoints, poses, betas = [], [], []
        for i in range(self.num_data):
            keypoints.append(res_dict[i]['keypoints'])
            poses.append(res_dict[i]['poses'])
            betas.append(res_dict[i]['betas'])

        res = dict(keypoints=keypoints, poses=poses, betas=betas)

        pred_keypoints3d, gt_keypoints3d, gt_keypoints3d_mask = \
            self._parse_result(res, mode='keypoint')
        
        gts = []

        mmcv.dump(res, res_file)

        name_value_tuples = []
        for _metric in metrics:
            if _metric == 'mpjpe':
                _nv_tuples = self._report_mpjpe(res)
            elif _metric == 'pa-mpjpe':
                _nv_tuples = self._report_mpjpe(res, metric='pa-mpjpe')
            elif _metric == '3dpck':
                _nv_tuples = self._report_3d_pck(res)
            elif _metric == 'pa-3dpck':
                _nv_tuples = self._report_3d_pck(res, metric='pa-3dpck')
            elif _metric == '3dauc':
                _nv_tuples = self._report_3d_auc(res)
            elif _metric == 'pa-3dauc':
                _nv_tuples = self._report_3d_auc(res, metric='pa-3dauc')
            elif _metric == 'pve':
                _nv_tuples = self._report_pve(res)
            else:
                raise NotImplementedError
            name_value_tuples.extend(_nv_tuples)

        name_value = OrderedDict(name_value_tuples)
        return name_value
