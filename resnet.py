import torch
import torch.nn as nn

import random
import numpy as np

seed = 2025
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_dropout, dropout_prob=0.3):
        super(ResidualBlock, self).__init__()
        self.use_dropout = use_dropout
        self.dropout_prob = dropout_prob
        self.match_dimensions = (in_channels != out_channels)
        self.linear1 = nn.Linear(in_channels, out_channels)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(out_channels, out_channels)
        if self.match_dimensions:
            self.match_dim_linear = nn.Linear(in_channels, out_channels)
        if self.use_dropout:
            self.dropout = nn.Dropout(p=self.dropout_prob)
    def forward(self, x):
        identity = x
        out = self.linear1(x)
        out = self.relu(out)
        if self.use_dropout:
            out = self.dropout(out)
        out = self.linear2(out)
        out = self.relu(out)
        if self.use_dropout:
            out = self.dropout(out)
        if self.match_dimensions:
            identity = self.match_dim_linear(identity)
        out = out + identity
        out = self.relu(out)
        return out

use_dropout=False

# 最终的网络
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.model = nn.Sequential(
            ResidualBlock(6, 16, use_dropout),
            ResidualBlock(16, 32, use_dropout),
            ResidualBlock(32, 64, use_dropout),
            ResidualBlock(64, 32, use_dropout),
            ResidualBlock(32, 16, use_dropout),
            nn.Linear(16, 1),  # 最后的输出层不包含激活
        )
    def forward(self, x):
        return self.model(x)

