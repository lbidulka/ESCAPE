import torch
import torch.nn as nn

from .residual import Linear

class Featmap_Model(nn.Module):
    def __init__(self, linear_size=1024, num_stages=2, p_dropout=0.5, 
                 num_in_kpts=17, featmap_shape=[2048, 8, 6], num_out_kpts=4):
        """
        Modified BaseLine model to additionally take in CLIFF feature maps as input
        
        Args:
            linear_size (int, optional): Number of nodes in the linear layers. Defaults to 1024.
            num_stages (int, optional): Number to repeat the linear block. Defaults to 2.
            p_dropout (float, optional): Dropout probability. Defaults to 0.5.
        """
        super(Featmap_Model, self).__init__()
        self.num_kpts = num_in_kpts

        kpt_insize = num_in_kpts * 3          # Input 3d-joints.
        featmap_insize = featmap_shape[0] * featmap_shape[1] * featmap_shape[2] # Input feature maps
        output_size = num_out_kpts * 3          # Output distal joint errs

        embed_size_fm = 512
        embed_size_common = 256

        # kpt stream
        self.emb_kpt = nn.Linear(kpt_insize, embed_size_common)    

        # Feature map stream
        self.emb_fm = nn.Linear(featmap_insize, embed_size_fm) 
        self.fm1 = nn.Linear(embed_size_fm, embed_size_common)
        # self.fm2 = nn.Linear(embed_size_fm, embed_size_fm)
        self.fm1_bn = nn.BatchNorm1d(embed_size_common)
        # self.fm2_bn = nn.BatchNorm1d(embed_size_fm)        

        # Common stream
        self.w1 = nn.Linear(embed_size_common * 2, linear_size)
        self.bn1 = nn.BatchNorm1d(linear_size)
        self.linear_stages = [Linear(linear_size, p_dropout) for _ in range(num_stages)]
        self.linear_stages = nn.ModuleList(self.linear_stages)

        self.w2 = nn.Linear(linear_size, output_size)

        self.relu = nn.ReLU(inplace=True)
        # self.l_relu = nn.LeakyReLU(inplace=True)
        self.dropout = nn.Dropout(p_dropout)

    def forward(self, x):
        """Forward operations of the linear block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            y (torch.Tensor): Output tensor.
        """
        # kpt stream
        kpt_embed = self.emb_kpt(x[0])
        
        # Feature map stream
        fm_embed = self.emb_fm(x[1])
        fm_embed = self.fm1_bn(self.fm1(fm_embed))
        fm_embed = self.dropout(self.relu(fm_embed))        
        
        embed = torch.cat((kpt_embed, fm_embed), dim=1)
        y = self.w1(embed)
        y = self.bn1(y)
        y = self.relu(y)
        y = self.dropout(y)

        # linear blocks
        for linear in self.linear_stages:
            y = linear(y)

        y = self.w2(y)

        return y