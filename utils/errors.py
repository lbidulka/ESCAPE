import torch

def loss_weighted_rep_no_scale(p2d_in, p3d_in, confs=None, num_joints=17, scale=10):
    '''
    Reprojection loss, considering 2D backbone confidences

    Mohsen used like:
        rot_poses = rot_poses.reshape(-1, num_joints*3)
        losses.rep = loss_weighted_rep_no_scale(inp_poses, rot_poses, inp_confidences)

    with shapes:
        inp_poses: (batch, 2*num_joints)     # 2D poses from backbone, uuu ... vvv ....
        rot_poses: (batch, 3*num_joints)     # 3D poses from lifter, xxx.... yyy... zzz.... in camera space
        inp_confidences: (batch, num_joints) # 2D confidences from backbone
    '''
    if confs is None:
        confs = torch.ones(p2d_in.shape[:2]).to(p2d_in.device)
    
    p2d = p2d_in.permute(0,2,1).reshape(-1, 2*num_joints)
    p3d = p3d_in.permute(0,2,1).reshape(-1, 3*num_joints)

    # normalize by scale
    scale_p2d = torch.sqrt(p2d.square().sum(axis=1, keepdim=True) / num_joints*2)
    p2d_scaled = p2d[:, 0:num_joints*2]/scale_p2d

    # only the u,v coordinates are used and depth is ignored
    # this is a simple weak perspective projection
    scale_p3d = torch.sqrt(p3d[:, 0:num_joints*2].square().sum(axis=1, keepdim=True) / num_joints*2)
    p3d_scaled = p3d[:, 0:num_joints*2]/scale_p3d

    loss = (scale * (p2d_scaled - p3d_scaled).abs().reshape(-1, 2, num_joints).sum(axis=1) * confs).sum(axis=1, keepdims=True) / (p2d_scaled.shape[0] * p2d_scaled.shape[1])

    return loss