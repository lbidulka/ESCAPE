import torch
from torch.utils.data.dataset import Dataset
import numpy as np

class cnet_pose_dataset(Dataset):
    '''
    
    '''
    def __init__(self, data, transform=None):
        '''
        args:
            data: numpy array of shape ()
        '''
        self.transform = transform
        self.data = data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        if self.transform:
            if np.random.rand() < self.transform.p:
                sample = self.transform(sample)
        return sample

class hflip_keypoints():
    '''
    Horizontally flip keypoints
    '''
    def __init__(self, p=0.5, flip_pairs=[[1,4], # hips
                                    [2,5], # knees
                                    [3,6], # ankles
                                    [11,14], # shoulders
                                    [12,15], # elbows
                                    [13,16], # wrists
                                    [8,9], # neck          
                                    ]):
        '''  '''
        self.flip_pairs = flip_pairs
        self.p = p

    def flip(self, keypoints, flip_pairs, img_width=None):
        """Flip human joints horizontally.

        Note:
            num_keypoints: K
            num_dimension: D
        Args:
            keypoints (np.ndarray([K, D])): Coordinates of keypoints.
            flip_pairs (list[tuple()]): Pairs of keypoints which are mirrored
                (for example, left ear -- right ear).
            img_width (int | None, optional): The width of the original image.
                To flip 2D keypoints, image width is needed. To flip 3D keypoints,
                we simply negate the value of x-axis. Default: None.
        Returns:
            keypoints_flipped
        """
        keypoints_flipped = keypoints.detach().clone()

        # Swap left-right parts
        for left, right in flip_pairs:
            keypoints_flipped[left, :] = keypoints[right, :]
            keypoints_flipped[right, :] = keypoints[left, :]

        # Flip horizontally
        if img_width is None:
            keypoints_flipped[:, 0] = -keypoints_flipped[:, 0]
        else:
            keypoints_flipped[:, 0] = img_width - 1 - keypoints_flipped[:, 0]

        return keypoints_flipped

    def foo(self,a,b):
        return a+b

    def __call__(self, sample):
        '''  '''
        backbone_preds = sample[0]
        target_xyz_17s = sample[1]

        flipped_backbone_preds = self.flip(backbone_preds, self.flip_pairs)
        flipped_target_xyz_17s = self.flip(target_xyz_17s, self.flip_pairs)

        return torch.cat([flipped_backbone_preds, 
                          flipped_target_xyz_17s, 
                          sample[2]]).reshape(3,-1,3)