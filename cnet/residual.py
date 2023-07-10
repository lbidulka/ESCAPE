import torch
import torch.nn as nn


class Linear(nn.Module):
    def __init__(self, linear_size, p_dropout):
        """

        Args:
            linear_size (int): Number of nodes in the linear layers.
            p_dropout (float): Dropout probability.
        """
        super(Linear, self).__init__()

        self.relu = nn.ReLU(inplace=True)
        self.l_relu = nn.LeakyReLU(inplace=True)
        self.dropout = nn.Dropout(p_dropout)

        self.w1 = nn.Linear(linear_size, linear_size)
        self.bn1 = nn.BatchNorm1d(linear_size)

        self.w2 = nn.Linear(linear_size, linear_size)
        self.bn2 = nn.BatchNorm1d(linear_size)

    def forward(self, x):
        """Forward operations of the linear block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            y (torch.Tensor): Output tensor.
        """
        h = self.w1(x)
        h = self.bn1(h)
        h = self.relu(h)
        # h = self.l_relu(h)
        h = self.dropout(h)

        h = self.w2(h)
        h = self.bn2(h)
        h = self.relu(h)
        # h = self.l_relu(h)
        h = self.dropout(h)

        y = x + h
        return y

class BaselineModel(nn.Module):
    def __init__(self, linear_size=1024, num_stages=2, p_dropout=0.5, 
                 num_in_kpts=17, num_out_kpts=4):
        """

        Args:
            linear_size (int, optional): Number of nodes in the linear layers. Defaults to 1024.
            num_stages (int, optional): Number to repeat the linear block. Defaults to 2.
            p_dropout (float, optional): Dropout probability. Defaults to 0.5.
            predict_14 (bool, optional): Whether to predict 14 3d-joints. Defaults to False.
        """
        super(BaselineModel, self).__init__()
        self.num_kpts = num_in_kpts

        input_size = num_in_kpts * 3          # Input 3d-joints.
        output_size = num_out_kpts * 3          # Output distal joint errs

        self.w1 = nn.Linear(input_size, linear_size)
        self.bn1 = nn.BatchNorm1d(linear_size)

        self.linear_stages = [Linear(linear_size, p_dropout) for _ in range(num_stages)]
        self.linear_stages = nn.ModuleList(self.linear_stages)

        self.w2 = nn.Linear(linear_size, output_size)

        self.relu = nn.ReLU(inplace=True)
        self.l_relu = nn.LeakyReLU(inplace=True)
        self.dropout = nn.Dropout(p_dropout)

    def forward(self, x):
        """Forward operations of the linear block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            y (torch.Tensor): Output tensor.
        """
        y = self.w1(x)
        y = self.bn1(y)
        y = self.relu(y)
        # y = self.l_relu(y)
        y = self.dropout(y)

        # linear blocks
        for linear in self.linear_stages:
            y = linear(y)

        y = self.w2(y)

        return y