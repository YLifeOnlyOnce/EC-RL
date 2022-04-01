#### 算法参数
```python
USER_NUM = 1
TASKS_NUM = 1
EDGENODE_NUM = 5

for index in range(USER_NUM):
    task = Task('task' + str(index), 100 + int(index), 1000, 5000)
    users.append(User('user' + str(index), 5, 45, 45, task))

for index in range(EDGENODE_NUM):
    edgenode = EdgeNode(32, 45, 45, 300, 2000, energy_factor=0.2, power_max=(80 + int(index)*10) )
    edgeNodes.append(edgenode)

EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 150000
BATCH_SIZE = 1000
GAMMA = 0.999

# 设置memory
memory = ReplayMemory(20000)

# 训练轮次
num_episodes = 50
# 每一轮多少步
count = 2000
```

#### 环境参数
```python
lamda = 0.12  # 能耗因子
p_idle = 0.65  # 空闲时 功耗占比
p_sleep = 0.3  # 休眠时 功耗占比
```


#### nn参数
```python

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


```