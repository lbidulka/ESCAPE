import torch
import torch.nn as nn

from uncertnet.cnet_all import adapt_net

# H36M_KEYPOINTS = [
        #     'pelvis_extra',    # 0
        #     'left_hip_extra', 'left_knee', 'left_ankle',
        #     'right_hip_extra', 'right_knee', 'right_ankle', #6
        #     'spine_extra', 'neck_extra',
        #     'head_extra', 'headtop',         # 10
        #     'left_shoulder', 'left_elbow', 'left_wrist',
        #     'right_shoulder', 'right_elbow', 'right_wrist', # 16
        # ]

class multi_distal():
    '''
    Wrapper to have a CNet for each limb individually
    '''
    def __init__(self, config) -> None:

        config.cnet_ckpt_path += 'sep_limbs/'
        self.config = config

        # pred_errs = config.pred_errs   # True: predict distal joint errors, False: predict 3d-joints directly
        # 10,9,8,7,0,1,4,
        LL_kpts = [0, 1, 2, 3]  # 0, 1, 2, 3
        LL_target = [3,]
        RL_kpts = [0, 4, 5, 6]  # 0, 4, 5, 6
        RL_target = [6,]
        LA_kpts = [7, 11, 12, 13]   # 7, 11, 12, 13
        LA_target = [13,]
        RA_kpts = [7, 14, 15, 16]   # 7, 14, 15, 16
        RA_target = [16,]

        if 'LL' in config.limbs:
            self.cnet_LL = adapt_net(config, #pred_errs=pred_errs,
                                     target_kpts=LL_target, in_kpts=LL_kpts)
            self.cnet_LL.config.cnet_ckpt_path += 'LL_'
        if 'RL' in config.limbs:
            self.cnet_RL = adapt_net(config, #pred_errs=pred_errs, 
                                     target_kpts=RL_target, in_kpts=RL_kpts)
            self.cnet_RL.config.cnet_ckpt_path += 'RL_'
        if 'LA' in config.limbs:
            self.cnet_LA = adapt_net(config, #pred_errs=pred_errs, 
                                     target_kpts=LA_target, in_kpts=LA_kpts)
            self.cnet_LA.config.cnet_ckpt_path += 'LA_'
        if 'RA' in config.limbs:
            self.cnet_RA = adapt_net(config, #pred_errs=pred_errs, 
                                     target_kpts=RA_target, in_kpts=RA_kpts)
            self.cnet_RA.config.cnet_ckpt_path += 'RA_'
    
    def __call__(self, input):
        corr_input = input.detach().clone()
        if 'LL' in self.config.limbs:
            corr_input = self.cnet_LL(corr_input)
        if 'RL' in self.config.limbs:
            corr_input = self.cnet_RL(corr_input)
        if 'LA' in self.config.limbs:
            corr_input = self.cnet_LA(corr_input)
        if 'RA' in self.config.limbs:
            corr_input = self.cnet_RA(corr_input)
        return corr_input

    def train(self,):
        if 'LL' in self.config.limbs:
            print("\n --- LL ---")
            self.cnet_LL.train()
        if 'RL' in self.config.limbs:
            print("\n --- RL ---")
            self.cnet_RL.train()
        if 'LA' in self.config.limbs:
            print("\n --- LA ---")
            self.cnet_LA.train()
        if 'RA' in self.config.limbs:
            print("\n --- RA ---")
            self.cnet_RA.train()
    
    def load_cnets(self, print_str=True):
        if 'LL' in self.config.limbs:
            self.cnet_LL.load_cnets(print_str=print_str)
        if 'RL' in self.config.limbs:
            self.cnet_RL.load_cnets(print_str=print_str)
        if 'LA' in self.config.limbs:
            self.cnet_LA.load_cnets(print_str=print_str)
        if 'RA' in self.config.limbs:
            self.cnet_RA.load_cnets(print_str=print_str)
    
    def save(self, path):
        if 'LL' in self.config.limbs:
            self.cnet_LL.save(path)
        if 'RL' in self.config.limbs:
            self.cnet_RL.save(path)
        if 'LA' in self.config.limbs:
            self.cnet_LA.save(path)
        if 'RA' in self.config.limbs:
            self.cnet_RA.save(path)
    
    def eval(self, ):
        if 'LL' in self.config.limbs:
            self.cnet_LL.eval()
        if 'RL' in self.config.limbs:
            self.cnet_RL.eval()
        if 'LA' in self.config.limbs:
            self.cnet_LA.eval()
        if 'RA' in self.config.limbs:
            self.cnet_RA.eval()

