'''
Author: yaoyaoyu
Date: 2022-04-02 21:06:00
Desc: DQN Network
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class Dueling_DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Dueling_DQN, self).__init__()
        self.bn = nn.BatchNorm1d(state_dim)
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.f1 = nn.Linear(state_dim, 512)
        self.f2 = nn.Linear(512, 256)

        self.val_hidden = nn.Linear(256, 128)
        self.adv_hidden = nn.Linear(256, 128)

        self.val = nn.Linear(128, 1)

        self.adv = nn.Linear(128, action_dim)

    def forward(self, x):
        x = self.f1(x)
        x = F.leaky_relu(x)
        x = self.f2(x)
        x = F.leaky_relu(x)
        val_hidden = self.val_hidden(x)
        val_hidden = F.leaky_relu(val_hidden)

        adv_hidden = self.adv_hidden(x)
        adv_hidden = F.leaky_relu(adv_hidden)

        val = self.val(val_hidden)

        adv = self.adv(adv_hidden)

        adv_ave = torch.mean(adv, dim=1, keepdim=True)

        x = adv + val - adv_ave

        return x
