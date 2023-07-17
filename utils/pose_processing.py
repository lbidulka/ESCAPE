import numpy as np
import torch

skeleton_3D_kpt_idxs = {
        'Head': 8, 
        'Neck': 7, 
        'LShoulder': 12, 'LElbow': 13, 'LWrist': 14,
        'RShoulder': 9, 'RElbow': 10, 'RWrist': 11,
        'Hip': 0, # Centre of Pelvis
        'LHip': 1, 'LKnee': 2, 'LAnkle': 3,
        'RHip': 4, 'RKnee': 5, 'RAnkle': 6,
}

def convert_kpts_coco_h36m(kpts_coco):
    H36M_KEYPOINTS = [
        'pelvis_extra',
        'left_hip_extra',
        'left_knee',
        'left_ankle',
        'right_hip_extra',  # 4
        'right_knee',
        'right_ankle',
        'spine_extra',  # 7
        'neck_extra',
        'head_extra',
        'headtop',  # 10
        'left_shoulder',
        'left_elbow',
        'left_wrist',
        'right_shoulder',   # 14
        'right_elbow',
        'right_wrist',
    ]
    keypoints_new = np.zeros((17, kpts_coco.shape[1]), dtype=kpts_coco.dtype)
    # pelvis (root) is in the middle of l_hip and r_hip
    keypoints_new[0] = (kpts_coco[11] + kpts_coco[12]) / 2
    # thorax is in the middle of l_shoulder and r_shoulder
    keypoints_new[8] = (kpts_coco[5] + kpts_coco[6]) / 2
    # spine is in the middle of thorax and pelvis
    keypoints_new[7] = (keypoints_new[0] + keypoints_new[8]) / 2
    # in COCO, head is in the middle of l_eye and r_eye
    # in PoseTrack18, head is in the middle of head_bottom and head_top
    keypoints_new[10] = (kpts_coco[1] + kpts_coco[2]) / 2
    # rearrange other keypoints
    keypoints_new[[1, 2, 3, 4, 5, 6, 9, 11, 12, 13, 14, 15, 16]] = \
        kpts_coco[[11, 13, 15, 12, 14, 16, 0, 5, 7, 9, 6, 8, 10]]
    return keypoints_new

def zero_pose_orient(poses, flip=False):
    '''
    Align the body axes with the world axes.
    '''
    data_normal = poses.copy()
    Rhip_idx = skeleton_3D_kpt_idxs['RHip']
    Lhip_idx = skeleton_3D_kpt_idxs['LHip']
    Neck_idx = skeleton_3D_kpt_idxs['Neck']
    Hip_idx = skeleton_3D_kpt_idxs['Hip']

    data_normal -= data_normal[:, Hip_idx, None, :]

    x_vec = data_normal[:, Lhip_idx] - data_normal[:, Rhip_idx]      # L Hip - R Hip
    y_vec = data_normal[:, Neck_idx] - data_normal[:, Hip_idx]      # Neck - Pelvis
    x_vec /= np.linalg.norm(x_vec, keepdims=True, axis=-1)
    y_vec /= np.linalg.norm(y_vec, keepdims=True, axis=-1)
    z_vec = np.cross(x_vec, y_vec)
    if flip:
        # rotate 180deg around z axis
        z_vec *= -1
        

    rotation_matrix = np.ones((len(x_vec), 3, 3))
    # Rot poses back to centre
    rotation_matrix[:,:,0] = x_vec
    rotation_matrix[:,:,1] = y_vec
    rotation_matrix[:,:,2] = z_vec

    data_normal = np.matmul(data_normal, rotation_matrix)

    return data_normal

def procrustes_torch(X, Y):
    """
    Reimplementation of MATLAB's `procrustes` function to Numpy.
    """
    X1=X[:,[0,1,4,7,8,9,10,13]]
    Y1=Y[:,[0,1,4,7,8,9,10,13]]
    batch,n, m = X1.shape
    batch, ny, my = Y1.shape

    muX = torch.mean(X1,dim=1,keepdim=True)
    muY = torch.mean(Y1,dim=1,keepdim=True)

    X0 = X1 - muX
    Y0 = Y1 - muY

    # optimum rotation matrix of Y
    A = torch.matmul(torch.transpose(X0,-1,-2), Y0)
    U,s,V = torch.svd(A)
    T = torch.matmul(V, torch.transpose(U,-1,-2))

    X1=X
    Y1=Y
    muX = torch.mean(X1,dim=1,keepdim=True)
    muY = torch.mean(Y1,dim=1,keepdim=True)

    X0 = X1 - muX
    Y0 = Y1 - muY

    Z = torch.matmul(Y0, T) + muX

    return np.array(Z.cpu())
