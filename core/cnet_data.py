import torch
from types import SimpleNamespace
from tqdm import tqdm
import numpy as np


def unpack_train_data(data, dataset, target_key, config):
    '''
    Unpack sample from dataloader & move to GPU
    '''
    if dataset == 'MPii':
        inps = data['pose_input']
        labels = {}
        labels['img_center'] = data['img_center']
        labels[target_key] = data['gt_pose']
        bboxes = data['bbox']
        img_ids = data['img_id'] #torch.ones(len(bboxes)) * -9  # dummy
        img_paths = data['img_path']
    else:
        (inps, labels, img_ids, bboxes) = data
        img_paths = torch.ones(len(bboxes)) * -9  # dummy

    if isinstance(inps, list):
        inps = [inp.to(config.device) for inp in inps]
    else:
        inps=inps.to(config.device)

    for k, _ in labels.items():
        try:
            labels[k] = labels[k].to(config.device)
        except AttributeError:
            assert k == 'type'
    
    return inps, labels, img_ids, img_paths, bboxes

def convert_kpts_to_h36m_17s(target_xyzs, dataset):
    '''
    Convert joint labels to H36M 17 joints
    '''
    if dataset == 'PW3D':
        return target_xyzs
    elif dataset == 'HP3D':
        EVAL_JOINTS_17 = [
            4,  
            18, 19, 20, 
            23, 24, 25,
            3, 5, # 'spine' == spine_extra, 'neck' == throat (not quite 'neck_extra' as desired)
            6, 7, # 
            9, 10, 11,
            14, 15, 16,
        ]
    if dataset == 'MPii':
        EVAL_JOINTS_17 = [
            0,
            1, 4, 7,    # LL
            2, 5, 8,    # RL
            6, 12,      # torso, neck
            15, 15,     #               (theres no head top unfortunately)
            16, 18, 20, # LA
            17, 19, 21, # RA
        ]        
    target_xyzs = [np.take(t.reshape(-1, t.shape[1], 3), EVAL_JOINTS_17, axis=1) for t in target_xyzs]
    target_xyzs = [t.reshape(-1, 51) for t in target_xyzs]
    return target_xyzs


def create_cnet_dataset_w_HybrIK(m, config, gt_dataset, dataset, task='train'):
    # Data/Setup
    gt_loader = torch.utils.data.DataLoader(gt_dataset, batch_size=64, shuffle=False, 
                                            num_workers=16, drop_last=False, pin_memory=True)
    m.eval()
    m = m.to(config.device)

    opt = SimpleNamespace()
    opt.device = config.device
    opt.flip_test = True

    target_keys = {
        'HP3D': 'target_xyz',
        'PW3D': 'target_xyz_17',
        'MPii': 'gt_pose',
    }
    target_key = target_keys[dataset]

    backbone_preds, target_xyzs, img_idss, img_pathss = [], [], [], []
    for i, data in enumerate(tqdm(gt_loader, dynamic_ncols=True)):
        inps, labels, img_ids, img_paths, bboxes = unpack_train_data(data, dataset, target_key, opt)

        m_output = m(inps, flip_test=opt.flip_test, bboxes=bboxes.to(config.device), img_center=labels['img_center'])
        backbone_pred = m_output.pred_xyz_jts_17

        backbone_preds.append(backbone_pred)
        target_xyzs.append(labels[target_key])
        img_idss.append(img_ids)
        # img_pathss.append(img_paths)

        if i > 1: break

    print("Detaching & reformatting...")
    backbone_preds = [b.detach().cpu().numpy() for b in backbone_preds]
    backbone_preds = np.concatenate(backbone_preds, axis=0)
    target_xyzs = [t.detach().cpu().numpy() for t in target_xyzs]
    target_xyz_17s = convert_kpts_to_h36m_17s(target_xyzs, dataset)
    target_xyz_17s = np.concatenate(target_xyz_17s, axis=0)
    img_idss = np.concatenate([i.detach().cpu().numpy() for i in img_idss], axis=0).reshape(-1,1)

    # img_names = []
    # for batch in img_pathss:
    #     for img_path in batch:
    #         img_name = img_path.split('/')[-1]
    #         img_name = img_name.split('.')[0]
    #         img_names.append(int(img_name))
    # # img_paths = np.concatenate([int(p.split('/')[-1].split('.')[0]) for p in img_paths], axis=0).reshape(-1,1)
    # img_pathss = np.array(img_names).reshape(-1,1)

    # normalize target magnitude to match the backbone
    scale_pred = np.linalg.norm(backbone_preds, keepdims=True)
    scale_gt = np.linalg.norm(target_xyz_17s, keepdims=True)
    target_xyz_17s /= (scale_gt / scale_pred)

    if task == 'train':
        dataset_outpath = '{}{}/{}_cnet_hybrik_train.npy'.format(config.cnet_dataset_path, dataset, config.hybrIK_version,)
    elif task == 'test':
        dataset_outpath = '{}{}/{}_cnet_hybrik_test.npy'.format(config.cnet_dataset_path, dataset, config.hybrIK_version,)
    print("Saving HybrIK pred dataset to {}".format(dataset_outpath))
    np.save(dataset_outpath, np.array([backbone_preds,
                                       target_xyz_17s,
                                       np.repeat(img_idss, backbone_pred.shape[1], axis=1),]))
                                    #    np.repeat(img_pathss, backbone_pred.shape[1], axis=1)]))
    return
