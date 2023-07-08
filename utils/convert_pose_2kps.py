from collections import namedtuple
from scipy.spatial.transform import Rotation
import numpy as np
import torch

Skeleton = namedtuple("Skeleton", ["joint_names", "joint_trees", "root_id", "nonroot_id", "cutoffs", "end_effectors"])

SMPLSkeleton = Skeleton(
    joint_names=[
        # 0-3
        'pelvis', 'left_hip', 'right_hip', 'spine1',
        # 4-7
        'left_knee', 'right_knee', 'spine2', 'left_ankle',
        # 8-11
        'right_ankle', 'spine3', 'left_foot', 'right_foot',
        # 12-15
        'neck', 'left_collar', 'right_collar', 'head',
        # 16-19
        'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
        # 20-23,
        'left_wrist', 'right_wrist', 'left_hand', 'right_hand'
    ],
    joint_trees=np.array(
                [0, 0, 0, 0,
                 1, 2, 3, 4,
                 5, 6, 7, 8,
                 9, 9, 9, 12,
                 13, 14, 16, 17,
                 18, 19, 20, 21]),
    root_id=0,
    nonroot_id=[i for i in range(24) if i != 0],
    cutoffs={'hip': 200, 'spine': 300, 'knee': 70, 'ankle': 70, 'foot': 40, 'collar': 100,
            'neck': 100, 'head': 120, 'shoulder': 70, 'elbow': 70, 'wrist': 60, 'hand': 60},
    end_effectors=[10, 11, 15, 22, 23],
)
smpl_rest_pose = np.array([[ 0.00000000e+00,  2.30003661e-09, -9.86228770e-08],
                           [ 1.63832515e-01, -2.17391014e-01, -2.89178602e-02],
                           [-1.57855421e-01, -2.14761734e-01, -2.09642015e-02],
                           [-7.04505108e-03,  2.50450850e-01, -4.11837511e-02],
                           [ 2.42021069e-01, -1.08830070e+00, -3.14962119e-02],
                           [-2.47206554e-01, -1.10715497e+00, -3.06970738e-02],
                           [ 3.95125849e-03,  5.94849110e-01, -4.03754264e-02],
                           [ 2.12680623e-01, -1.99382353e+00, -1.29327580e-01],
                           [-2.10857525e-01, -2.01218796e+00, -1.23002514e-01],
                           [ 9.39484313e-03,  7.19204426e-01,  2.06931755e-02],
                           [ 2.63385147e-01, -2.12222481e+00,  1.46775618e-01],
                           [-2.51970559e-01, -2.12153077e+00,  1.60450473e-01],
                           [ 3.83779174e-03,  1.22592449e+00, -9.78838727e-02],
                           [ 1.91201791e-01,  1.00385976e+00, -6.21964522e-02],
                           [-1.77145526e-01,  9.96228695e-01, -7.55542740e-02],
                           [ 1.68482102e-02,  1.38698268e+00,  2.44048554e-02],
                           [ 4.01985168e-01,  1.07928419e+00, -7.47655183e-02],
                           [-3.98825467e-01,  1.07523870e+00, -9.96334553e-02],
                           [ 1.00236952e+00,  1.05217218e+00, -1.35129794e-01],
                           [-9.86728609e-01,  1.04515052e+00, -1.40235111e-01],
                           [ 1.56646240e+00,  1.06961894e+00, -1.37338534e-01],
                           [-1.56946480e+00,  1.05935931e+00, -1.53905824e-01],
                           [ 1.75282109e+00,  1.04682994e+00, -1.68231070e-01],
                           [-1.75758195e+00,  1.04255080e+00, -1.77773550e-01]], dtype=np.float32)

def get_smpl_l2ws(pose, rest_pose=None, scale=1., skel_type=SMPLSkeleton, coord="xxx"):
    # TODO: take root as well

    def mat_to_homo(mat):
        last_row = np.array([[0, 0, 0, 1]], dtype=np.float32)        
        return np.concatenate([mat, last_row], axis=0)
    
    joint_trees = skel_type.joint_trees    
    if rest_pose is None:        # original bone parameters is in (x,-z,y), while rest_pose is in (x, y, z)
        rest_pose = smpl_rest_pose

    # apply scale
    rest_kp = rest_pose * scale
    mrots = [Rotation.from_rotvec(p).as_matrix()  for p in pose]
    mrots = np.array(mrots)

    l2ws = []
    # TODO: assume root id = 0
    # local-to-world transformation
    l2ws.append(mat_to_homo(np.concatenate([mrots[0], rest_kp[0, :, None]], axis=-1)))
    mrots = mrots[1:]
    for i in range(rest_kp.shape[0] - 1):
        idx = i + 1
        # rotation relative to parent
        joint_rot = mrots[idx-1]
        joint_j = rest_kp[idx][:, None]

        parent = joint_trees[idx]
        parent_j = rest_kp[parent][:, None]

        # transfer from local to parent coord
        joint_rel_transform = mat_to_homo(
            np.concatenate([joint_rot, joint_j - parent_j], axis=-1)
        )

        # calculate kinematic chain by applying parent transform (to global)
        l2ws.append(l2ws[parent] @ joint_rel_transform)

    l2ws = np.array(l2ws)

    return l2ws

# def get_smpl_l2ws_torch(pose, rest_pose=None, scale=1., skel_type=SMPLSkeleton, coord="xxx",axis_to_matrix=True):
#     # TODO: take root as well

#     def mat_to_homo(mat):
        
#         last_row = torch.from_numpy(np.asarray([[0., 0., 0., 1.]]))
#         last_row=last_row.to('cuda')
#         last_row=last_row.repeat(mat.shape[0],1,1)
#         return torch.cat([mat, last_row], dim=1)

#     joint_trees = skel_type.joint_trees
#     if rest_pose is None:
#         # original bone parameters is in (x,-z,y), while rest_pose is in (x, y, z)
#         rest_pose = torch.from_numpy(smpl_rest_pose.astype('float32'))
#         rest_pose=rest_pose.repeat(pose.shape[0],1,1)
#         rest_pose=rest_pose.to(device=device)


#     # apply scale
#     scale=(torch.tensor(scale)).to(device=device)
#     rest_kp = rest_pose * scale
#     if axis_to_matrix:
#         pose=pose.view(-1,3)
#         mrots = torch3d.axis_angle_to_matrix(pose)
#         mrots=mrots.view(-1,24,3,3)
#     else:
#         mrots=pose
 
#     # mrots = np.array(mrots)

#     l2ws = np.zeros((mrots.shape[0],mrots.shape[1],4,4),dtype='float')
#     l2ws=torch.from_numpy(l2ws)
#     l2ws=l2ws.to('cuda')
#     joint_rel_transform = np.zeros((mrots.shape[0],mrots.shape[1],4,4),dtype='float')
#     joint_rel_transform=torch.from_numpy(joint_rel_transform)
#     joint_rel_transform=joint_rel_transform.to('cuda')
#     # TODO: assume root id = 0
#     # local-to-world transformation
#     joint_rel_transform[:,0]=mat_to_homo(torch.cat([mrots[:,0], rest_kp[:,0, :, None]], dim=-1))
    
#     mrots = mrots[:,1:]
#     for i in range(rest_kp.shape[1] - 1):
#         idx = i + 1
#         # rotation relative to parent
#         joint_rot = mrots[:,idx-1]
#         joint_j = rest_kp[:,idx,:, None]

#         parent = joint_trees[idx]
#         parent_j = rest_kp[:,parent,:, None]
  
#         # transfer from local to parent coord
#         joint_rel_transform[:,idx] = mat_to_homo(
#             torch.cat([joint_rot, joint_j - parent_j], dim=-1)
#         )
#         # l2ws[:,idx]=joint_rel_transform @ joint_rel_transform
        
#         torch.cuda.empty_cache()
#         # calculate kinematic chain by applying parent transform (to global)
#     # print(l2ws[:,0].shape)
    
#     l2ws[:,0]= joint_rel_transform[:,0]
#     l2ws[:,1]= joint_rel_transform[:,0]@joint_rel_transform[:,1]
#     l2ws[:,2]= joint_rel_transform[:,0]@joint_rel_transform[:,2]
#     l2ws[:,3]= joint_rel_transform[:,0]@joint_rel_transform[:,3]
#     l2ws[:,4]= joint_rel_transform[:,0]@joint_rel_transform[:,1]@joint_rel_transform[:,4]
#     l2ws[:,5]= joint_rel_transform[:,0]@joint_rel_transform[:,2]@joint_rel_transform[:,5]
#     l2ws[:,6]= joint_rel_transform[:,0]@joint_rel_transform[:,3]@joint_rel_transform[:,6]
#     l2ws[:,7]= joint_rel_transform[:,0]@joint_rel_transform[:,1]@joint_rel_transform[:,4]@joint_rel_transform[:,7]
#     l2ws[:,8]= joint_rel_transform[:,0]@joint_rel_transform[:,2]@joint_rel_transform[:,5]@joint_rel_transform[:,8]
#     l2ws[:,9]= joint_rel_transform[:,0]@joint_rel_transform[:,3]@joint_rel_transform[:,6]@joint_rel_transform[:,9]
#     l2ws[:,10]=joint_rel_transform[:,0]@joint_rel_transform[:,1]@joint_rel_transform[:,4]@joint_rel_transform[:,7]@joint_rel_transform[:,10]
#     l2ws[:,11]=joint_rel_transform[:,0]@joint_rel_transform[:,2]@joint_rel_transform[:,5]@joint_rel_transform[:,8]@joint_rel_transform[:,11]
#     l2ws[:,12]=joint_rel_transform[:,0]@joint_rel_transform[:,3]@joint_rel_transform[:,6]@joint_rel_transform[:,9]@joint_rel_transform[:,12]
#     l2ws[:,13]=joint_rel_transform[:,0]@joint_rel_transform[:,3]@joint_rel_transform[:,6]@joint_rel_transform[:,9]@joint_rel_transform[:,13]
#     l2ws[:,14]=joint_rel_transform[:,0]@joint_rel_transform[:,3]@joint_rel_transform[:,6]@joint_rel_transform[:,9]@joint_rel_transform[:,14]
#     l2ws[:,15]=joint_rel_transform[:,0]@joint_rel_transform[:,3]@joint_rel_transform[:,6]@joint_rel_transform[:,9]@joint_rel_transform[:,12]@joint_rel_transform[:,15]
#     l2ws[:,16]=joint_rel_transform[:,0]@joint_rel_transform[:,3]@joint_rel_transform[:,6]@joint_rel_transform[:,9]@joint_rel_transform[:,13]@joint_rel_transform[:,16]
#     l2ws[:,18]=joint_rel_transform[:,0]@joint_rel_transform[:,3]@joint_rel_transform[:,6]@joint_rel_transform[:,9]@joint_rel_transform[:,13]@joint_rel_transform[:,16]@joint_rel_transform[:,18]
#     l2ws[:,20]=joint_rel_transform[:,0]@joint_rel_transform[:,3]@joint_rel_transform[:,6]@joint_rel_transform[:,9]@joint_rel_transform[:,13]@joint_rel_transform[:,16]@joint_rel_transform[:,18]@joint_rel_transform[:,20]
#     l2ws[:,22]=joint_rel_transform[:,0]@joint_rel_transform[:,3]@joint_rel_transform[:,6]@joint_rel_transform[:,9]@joint_rel_transform[:,13]@joint_rel_transform[:,16]@joint_rel_transform[:,18]@joint_rel_transform[:,20]@joint_rel_transform[:,22]
#     l2ws[:,17]=joint_rel_transform[:,0]@joint_rel_transform[:,3]@joint_rel_transform[:,6]@joint_rel_transform[:,9]@joint_rel_transform[:,14]@joint_rel_transform[:,17]
#     l2ws[:,19]=joint_rel_transform[:,0]@joint_rel_transform[:,3]@joint_rel_transform[:,6]@joint_rel_transform[:,9]@joint_rel_transform[:,14]@joint_rel_transform[:,17]@joint_rel_transform[:,19]
#     l2ws[:,21]=joint_rel_transform[:,0]@joint_rel_transform[:,3]@joint_rel_transform[:,6]@joint_rel_transform[:,9]@joint_rel_transform[:,14]@joint_rel_transform[:,17]@joint_rel_transform[:,19]@joint_rel_transform[:,21]
#     l2ws[:,23]=joint_rel_transform[:,0]@joint_rel_transform[:,3]@joint_rel_transform[:,6]@joint_rel_transform[:,9]@joint_rel_transform[:,14]@joint_rel_transform[:,17]@joint_rel_transform[:,19]@joint_rel_transform[:,21]@joint_rel_transform[:,23]
#     return l2ws