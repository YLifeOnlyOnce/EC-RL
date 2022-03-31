'''
Author: yaoyaoyu
Date: 2022-03-28 21:06:00
Desc: DQN Network
'''

import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    def __init__(self, state_space, action_space ):
        super(DQN, self).__init__()
        self.l0 = nn.Linear(state_space,256)
        self.l1 = nn.Linear(256, 128)
        self.l2 = nn.Linear(128, 64)
        self.out = nn.Linear(64, action_space)

    def forward(self, x):
        x = F.leaky_relu(self.l0(x))
        x = F.leaky_relu(self.l1(x))
        x = F.leaky_relu(self.l2(x))
        x = F.leaky_relu(self.out(x))
        return x

