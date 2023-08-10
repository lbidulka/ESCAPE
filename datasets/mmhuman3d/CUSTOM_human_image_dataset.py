import json
import os
import os.path
from abc import ABCMeta
from collections import OrderedDict
from typing import Any, List, Optional, Union

import mmcv
import numpy as np
import torch
import torch.distributed as dist
from mmcv.runner import get_dist_info

from mmhuman3d.core.conventions.keypoints_mapping import (
    convert_kps,
    get_keypoint_num,
    get_mapping,
)
from mmhuman3d.core.evaluation import (
    keypoint_3d_auc,
    keypoint_3d_pck,
    keypoint_mpjpe,
    vertice_pve,
)
from mmhuman3d.data.data_structures.human_data import HumanData
from mmhuman3d.data.data_structures.human_data_cache import (
    HumanDataCacheReader,
    HumanDataCacheWriter,
)
from mmhuman3d.models.body_models.builder import build_body_model
from .base_dataset import BaseDataset
from .human_image_dataset import HumanImageDataset
from .builder import DATASETS

from utils.convert_pose_2kps import get_smpl_l2ws
from core.cnet_data import convert_kpts_to_h36m_17s


@DATASETS.register_module()
class CUSTOM_HumanImageDataset(HumanImageDataset,):
    """Human Image Dataset.
    """
    def prepare_raw_data(self, idx: int):
        """Get item from self.human_data.
        
        Custom version due to path differences with annotations, bc I don't want to reformat
        paths and interfere with other projects
        """
        sample_idx = idx
        if self.cache_reader is not None:
            self.human_data = self.cache_reader.get_item(idx)
            idx = idx % self.cache_reader.slice_size
        info = {}
        info['img_prefix'] = None
        image_path = self.human_data['image_path'][idx]

        # Adjust path to match my setup 
        if self.dataset_name == 'mpi_inf_3dhp':
            new_path = image_path.split('/')
            new_path.insert(2, 'images')
            new_path[3] = '{}_{}_V{}'.format(new_path[0], new_path[1], new_path[3][-1:])
            new_path[4] = 'img_{}_{}'.format(new_path[3], new_path[4])
            image_path = '/'.join(new_path)

        info['image_path'] = os.path.join(self.data_prefix, 'datasets',
                                          self.dataset_name, image_path)
        if image_path.endswith('smc'):
            device, device_id, frame_id = self.human_data['image_id'][idx]
            info['image_id'] = (device, int(device_id), int(frame_id))

        info['dataset_name'] = self.dataset_name
        info['sample_idx'] = sample_idx
        if 'bbox_xywh' in self.human_data:
            info['bbox_xywh'] = self.human_data['bbox_xywh'][idx]
            x, y, w, h, s = info['bbox_xywh']
            cx = x + w / 2
            cy = y + h / 2
            w = h = max(w, h)
            info['center'] = np.array([cx, cy])
            info['scale'] = np.array([w, h])
        else:
            info['bbox_xywh'] = np.zeros((5))
            info['center'] = np.zeros((2))
            info['scale'] = np.zeros((2))

        # in later modules, we will check validity of each keypoint by
        # its confidence. Therefore, we do not need the mask of keypoints.

        if 'keypoints2d' in self.human_data:
            info['keypoints2d'] = self.human_data['keypoints2d'][idx]
            info['has_keypoints2d'] = 1
        else:
            info['keypoints2d'] = np.zeros((self.num_keypoints, 3))
            info['has_keypoints2d'] = 0
        if 'keypoints3d' in self.human_data:
            info['keypoints3d'] = self.human_data['keypoints3d'][idx]
            info['has_keypoints3d'] = 1
        else:
            info['keypoints3d'] = np.zeros((self.num_keypoints, 4))
            info['has_keypoints3d'] = 0

        if 'smpl' in self.human_data:
            smpl_dict = self.human_data['smpl']
        else:
            smpl_dict = {}

        if 'smpl' in self.human_data:
            if 'has_smpl' in self.human_data:
                info['has_smpl'] = int(self.human_data['has_smpl'][idx])
            else:
                info['has_smpl'] = 1
        else:
            info['has_smpl'] = 0
        if 'body_pose' in smpl_dict:
            info['smpl_body_pose'] = smpl_dict['body_pose'][idx]
        else:
            info['smpl_body_pose'] = np.zeros((23, 3))

        if 'global_orient' in smpl_dict:
            info['smpl_global_orient'] = smpl_dict['global_orient'][idx]
        else:
            info['smpl_global_orient'] = np.zeros((3))

        if 'betas' in smpl_dict:
            info['smpl_betas'] = smpl_dict['betas'][idx]
        else:
            info['smpl_betas'] = np.zeros((10))

        if 'transl' in smpl_dict:
            info['smpl_transl'] = smpl_dict['transl'][idx]
        else:
            info['smpl_transl'] = np.zeros((3))

        return info

    def _get_closest_mpii_3d_annot(self, pred, gt_options):
        '''
        This is a silly function.

        Return the 3d annotation from the img in question by choosing the one which most closely matches
        the prediction from my hybrik framework 

        gt_options: list of 3d annotations from the img in question [B x (candidates, 17, 3)]
        pred: preds (B, 17, 3)
        '''
        # gts = np.zeros(pred.shape)
        gts = []
        for i, options in enumerate(gt_options):
            opt = np.array(options)
            zeroed_pred = pred[i] - pred[i][0]
            dists = np.linalg.norm(zeroed_pred - opt, axis=(1,2))
            gts.append(options[np.argmin(dists)] if len(opt) > 0 else np.zeros((17, 3)))
        # # remove all 0's from gts
        # gts = [gt for gt in gts if np.sum(gt) != 0]
        gts = np.array(gts)
        return gts
        

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
            batch_size = len(out['keypoints_3d'])
            for i in range(batch_size):
                res_dict[int(target_id[i])] = dict(
                    keypoints=out['keypoints_3d'][i],
                    poses=out['smpl_pose'][i],
                    betas=out['smpl_beta'][i],
                    ids=target_id[i].item(),
                    path=out['image_path'][i],
                )

        keypoints, poses, betas, ids, paths = [], [], [], [], []
        for i in list(res_dict.keys()):
            keypoints.append(res_dict[i]['keypoints'])
            poses.append(res_dict[i]['poses'])
            betas.append(res_dict[i]['betas'])
            ids.append(res_dict[i]['ids'])
            paths.append(res_dict[i]['path'])

        res = dict(keypoints=keypoints, poses=poses, betas=betas,)

        """Parse gts."""
        # gts = [self.human_data[id] for id in ids]
        pred_keypoints3d = res['keypoints']
        # (B, 17, 3)
        pred_keypoints3d = np.array(pred_keypoints3d)

        if self.dataset_name == 'pw3d':
            gt_betas = []
            body_pose = []
            global_orient = []
            gender = []
            smpl_dict = self.human_data['smpl']
            for idx in ids:
                gt_betas.append(smpl_dict['betas'][idx])
                body_pose.append(smpl_dict['body_pose'][idx])
                global_orient.append(smpl_dict['global_orient'][idx])
                if self.human_data['meta']['gender'][idx] == 'm':
                    gender.append(0)
                else:
                    gender.append(1)
            gt_betas = torch.FloatTensor(gt_betas)
            body_pose = torch.FloatTensor(body_pose).view(-1, 69)
            global_orient = torch.FloatTensor(global_orient)
            gender = torch.Tensor(gender)
            gt_output = self.body_model(
                betas=gt_betas,
                body_pose=body_pose,
                global_orient=global_orient,
                gender=gender)
            gt_keypoints3d = gt_output['joints'].detach().cpu().numpy()
            assert pred_keypoints3d.shape[1] == 17
        elif self.dataset_name == 'mpi_inf_3dhp':
            # _, h36m_idxs, _ = get_mapping('smpl_49', 'h36m')
            _, h36m_idxs, _ = get_mapping(self.convention, 'h36m')
            gt_keypoints3d = self.human_data['keypoints3d'][ids][:, h36m_idxs, :3] # missing 'head_top'
            # duplicate head_extra (nose) in lieu of head_top kpt if needed
            if np.sum(gt_keypoints3d[:,10]) == 0:
                gt_keypoints3d[:,10] = gt_keypoints3d[:,9]
        elif self.dataset_name == 'mpii':
            # mpii only has 2D annotations, so we gotta do some tomfoolery
            
            # first we get the potential 3D annotations from the img in question
            # load all hybrik framework annotations & get the 3D
            hybrik_mpii_annot_path = '/media/ExtHDD/Mohsen_data/mpii_human_pose/mpii_cliffGT.npz'
            hybrik_framework_annots = np.load(hybrik_mpii_annot_path)
            hybrik_framework_gts = hybrik_framework_annots['pose']
            hybrik_framework_paths = hybrik_framework_annots['imgname']
            candidate_gts = np.zeros((hybrik_framework_gts.shape[0],24,3))
            for i, gt in enumerate(hybrik_framework_gts):
                candidate_gts[i] = get_smpl_l2ws(gt.reshape(24,3), scale=0.4)[:,:3,-1]
            candidate_gts -= candidate_gts[:,:1,:]
            # convert to h36m format
            h36m_idxs = [
                0,
                1, 4, 7,    # LL
                2, 5, 8,    # RL
                6, 12,      # torso, neck
                15, 15,     #               (theres no head top unfortunately)
                16, 18, 20, # LA
                17, 19, 21, # RA
            ]     
            candidate_gt_poses = candidate_gts.reshape(candidate_gts.shape[0],-1,3)[:, h36m_idxs, :3]
            # get img names of the samples
            img_names = []
            for img_path in paths:
                img_names.append(int(img_path.split('/')[-1].split('.')[0]))
            img_names = np.array(img_names).reshape(-1)
            gt_img_names = []
            for img_path in hybrik_framework_paths:
                gt_img_names.append(int(img_path.split('/')[-1].split('.')[0]))
            gt_img_names = np.array(gt_img_names).reshape(-1)
            # get the 3d annotations from the hybrik framework, where their img int matches the img int of the sample
            candidate_gts = []
            for i, img_name in enumerate(img_names):
                candidates = candidate_gt_poses[np.where(gt_img_names==img_name)]
                candidate_gts.append(candidates)
            # keep the closest one
            gt_keypoints3d = self._get_closest_mpii_3d_annot(pred_keypoints3d, candidate_gts)
        else: 
            raise NotImplementedError
        
        # root joint alignment
        pred_pelvis = pred_keypoints3d[:, 0]
        gt_pelvis = gt_keypoints3d[:, 0]
        pred_keypoints3d = (pred_keypoints3d - pred_pelvis[:, None, :]) * 1000
        gt_keypoints3d = (gt_keypoints3d - gt_pelvis[:, None, :]) * 1000

        """Combine"""
        res_outs = dict(preds=pred_keypoints3d, gts=gt_keypoints3d, 
                        ids=ids, paths=paths,
                        poses=poses, betas=betas,)
        if save_results:
            print("\nSaving preds, gts, img ids, img paths, poses, & betas to: ", res_file)
            mmcv.dump(res_outs, res_file)
        else:
            print("\nNot saving preds, gts, img ids, img paths, poses, & betas.")

        return res_outs    

    def my_evaluate(self,
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
                raise KeyError(f'metric {metric} is not supported')

        res_file = os.path.join(res_folder, 'result_keypoints.json')
        # for keeping correctness during multi-gpu test, we sort all results

        print("\nUnpacking preds... ", end='')
        res_dict = {}
        for out in outputs:
            target_id = out['image_idx']
            batch_size = len(out['keypoints_3d'])
            for i in range(batch_size):
                res_dict[int(target_id[i])] = dict(
                    keypoints=out['keypoints_3d'][i],
                    poses=out['smpl_pose'][i],
                    betas=out['smpl_beta'][i],
                )
        print("Repacking preds... ", end='')
        keypoints, poses, betas = [], [], []
        for i in list(res_dict.keys()):
            keypoints.append(res_dict[i]['keypoints'])
            poses.append(res_dict[i]['poses'])
            betas.append(res_dict[i]['betas'])        

        res = dict(keypoints=keypoints, poses=poses, betas=betas)

        mmcv.dump(res, res_file)
        print("Computing metrics... ", end='')
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
            elif _metric == 'ihmr':
                _nv_tuples = self._report_ihmr(res)
            else:
                raise NotImplementedError
            name_value_tuples.extend(_nv_tuples)

        name_value = OrderedDict(name_value_tuples)
        return name_value

    