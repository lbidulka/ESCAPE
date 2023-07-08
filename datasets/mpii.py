import cv2
import torch
import torch.utils.data as data
import numpy as np
import os
from easydict import EasyDict as edict

from hybrik.utils.presets import SimpleTransform3DSMPLCam

from utils.convert_pose_2kps import get_smpl_l2ws

class mpii_dataset(data.Dataset):
    def __init__(self, 
                 cfg,
                 annot_dir, 
                 image_dir):
        self._cfg = cfg
        self.image_dir = image_dir
        self.annot = np.load(annot_dir)
        self.pose = self.annot['pose']
        self.imgname = self.annot['imgname']
        self.center = self.annot['center']
        self.scale = self.annot['scale']
        bbox_3d_shape = getattr(self._cfg.MODEL, 'BBOX_3D_SHAPE', (2000, 2000, 2000))
        bbox_3d_shape = [item * 1e-3 for item in bbox_3d_shape]
        dummpy_set = edict({
            'joint_pairs_17': None,
            'joint_pairs_24': None,
            'joint_pairs_29': None,
            'bbox_3d_shape': bbox_3d_shape
        })        
        self.transformation= SimpleTransform3DSMPLCam(
            dummpy_set, scale_factor = self._cfg.DATASET.SCALE_FACTOR,
            color_factor = self._cfg.DATASET.COLOR_FACTOR,
            occlusion = self._cfg.DATASET.OCCLUSION,
            input_size = self._cfg.MODEL.IMAGE_SIZE,
            output_size = self._cfg.MODEL.HEATMAP_SIZE,
            depth_dim = self._cfg.MODEL.EXTRA.DEPTH_DIM,
            bbox_3d_shape = bbox_3d_shape,
            rot = self._cfg.DATASET.ROT_FACTOR, sigma = self._cfg.MODEL.EXTRA.SIGMA,
            train = False, add_dpg = False,
            loss_type = self._cfg.LOSS['TYPE'],
            focal_length = 200)
        
    def __len__(self):
        return len(self.pose)
    
    def __getitem__(self, idx):
        img_path = self.image_dir + self.imgname[idx]
        # process file name
        # img_path = os.path.join(opt.img_dir, file)
        # dirname = os.path.dirname(img_path)
        # basename = os.path.basename(img_path)
     
        # # Run Detection
        gt_pose = np.reshape(self.pose[idx], (24,3))
        gt_pose = get_smpl_l2ws(gt_pose, scale=0.4)[:,:3,-1]
        # gt_pose /= 2 # SCALE ISSUE
        gt_pose -= gt_pose[0]
        input_image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        # det_input = det_transform(input_image).to(opt.gpu)
        # det_output = det_model([det_input])[0]
        center = self.center[idx]
        scale = self.scale[idx] * 200
        xy1 = center - scale / 2
        xy2 = center + scale / 2      
        # tight_bbox = get_one_box(det_output)  # xyxy
        tight_bbox = [xy1[0], xy1[1], xy2[0], xy2[1]]
        # print(tight_bbox)
        # Run HybrIK
        # bbox: [x1, y1, x2, y2]
        pose_input, bbox, img_center = self.transformation.test_transform(
            input_image, tight_bbox) 
        sample = {'pose_input': pose_input,
                  'gt_pose': gt_pose,
                  'bbox': torch.from_numpy(np.array(bbox)),
                  'img_center': torch.from_numpy(img_center)}
        return sample