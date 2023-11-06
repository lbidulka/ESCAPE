import torch
from torch.utils.data.dataset import Dataset
import numpy as np

class cnet_pose_dataset(Dataset):
    '''
    
    '''
    def __init__(self, data, datasets, backbones, config, train=False, transforms=None, use_feats=False):
        '''
        args:
            data: numpy array of shape ()
        '''
        
        self.transforms = transforms
        self.data = data
        self.train = train
        self.datasets = datasets
        self.backbones = backbones
        self.data_path = config.cnet_dataset_path
        self.config = config

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        if self.transforms:
            for transform in self.transforms:
                if np.random.rand() < transform.p:
                    sample = transform(sample)

        # use id of sample to load feature maps if required
        if self.config.use_features:
            if sample.shape[0] < 4:
                f = 5
            id = int(sample[2].flatten()[0].item())
            # load feature maps from file
            dataset_idx = int(sample[3].flatten()[0].item())
            dataset = self.datasets[dataset_idx]
            backbone_idx = int(sample[4].flatten()[0].item())
            backbone = self.backbones[backbone_idx] if self.train else self.backbones[backbone_idx]
            framework = 'mmlab' if backbone in self.config.mmlab_backbones else 'bedlam'

            # feat_path = self.data_path.split('mmlab')[0] + f'feature_maps/{self.data_path.split(".")[0].split("/")[-1]}/{id}.npy'
            feat_path = f'{self.data_path}{dataset}/feature_maps/{framework}_{backbone}_{"train" if self.train else "test"}/{id}.npy'
            feats = np.load(feat_path)
            sample = {'data': sample,
                      'feats': feats}        
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

    def __call__(self, sample):
        '''  '''
        backbone_preds = sample[0]
        target_xyz_17s = sample[1]

        flipped_backbone_preds = self.flip(backbone_preds, self.flip_pairs)
        flipped_target_xyz_17s = self.flip(target_xyz_17s, self.flip_pairs)

        if sample.shape[0] == 2:
            return torch.cat([flipped_backbone_preds, 
                              flipped_target_xyz_17s]).reshape(2,-1,3)
        elif sample.shape[0] == 5:
            # features case
            return torch.cat([flipped_backbone_preds.reshape(1,-1,3), 
                          flipped_target_xyz_17s.reshape(1,-1,3), 
                          sample[2:]])
        else:
            return torch.cat([flipped_backbone_preds, 
                          flipped_target_xyz_17s, 
                          sample[2]]).reshape(3,-1,3)
    
class rescale_keypoints():
    '''
    Rescale pose keypoints
    '''
    def __init__(self, p=1.0, range=[0.9, 1.1]):
        '''  '''
        self.range = range
        self.p = p
    
    def rescale(self, keypoints, range):
        '''  '''
        keypoints_rescaled = keypoints.detach().clone()
        keypoints_rescaled *= np.random.uniform(range[0], range[1])
        return keypoints_rescaled

    def __call__(self, sample):
        ''' '''
        sample[0] = self.rescale(sample[0], self.range) # backbone preds
        sample[1] = self.rescale(sample[1], self.range) # targets
        return sample
