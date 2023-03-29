'''
Author: yaoyaoyu
Date: 2022-03-31 13:20:00
Desc: agent 与环境交互
'''

import sys
import numpy as np
import random
import math

import torch
import torch.nn as nn
import torch.optim as optim
from DQN.dqn import DQN
from DQN.duelingDqn import Dueling_DQN
from torch.utils.tensorboard import SummaryWriter

from collections import namedtuple, deque

from edge_env.env_energy_static import EdgeEnv
from edge_env.edgenode import EdgeNode
from edge_env.task import Task
from edge_env.user import User

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# log记录
writer = SummaryWriter("logstrain")


# --------- 初始化 用户和节点 -------------
users = []
tasks = []
edgeNodes = []

USER_NUM = 2
TASKS_NUM = 1
EDGENODE_NUM = 5

for index in range(USER_NUM):
    task = Task('task' + str(index), 100 + int(index), 1000, 5000)
    users.append(User('user' + str(index), 5, 45, 45, task))

for index in range(EDGENODE_NUM):
    edgenode = EdgeNode(32, 45, 45, 300, 2000, energy_factor=0.2, power_max=(80 + int(index)*10) )
    edgeNodes.append(edgenode)
# ---------- 初始化用户和节点 end -----------


# - -- - -- - - - - - -- DQN ---- - - - - - - - - - - -
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 300000  # 400-10000-8000   #600- 10000 - 8000  # 800--60000-20000不够
BATCH_SIZE = 300
GAMMA = 0.99  #0.999
TARGET_UPDATE = 3

state_space = USER_NUM * (EDGENODE_NUM + 1) + EDGENODE_NUM * USER_NUM
action_space = USER_NUM * (1 + 2*EDGENODE_NUM)
policy_net = Dueling_DQN(state_space, action_space).to(device)
target_net = Dueling_DQN(state_space, action_space).to(device)
# 预训练权重加载
target_net.load_state_dict(policy_net.state_dict())
# 优化器
optimizer = optim.RMSprop(policy_net.parameters(), lr=0.01)

steps_done = 0


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                    math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if steps_done % 3000 == 0:
        print("eps:", eps_threshold)
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return torch.tensor([torch.argmax(policy_net(state.float()))])
    else:
        return torch.tensor([random.randrange(action_space)], device=device, dtype=torch.long)


# ---------------------- DQN agent done --------------------------

# ---------------------- 优化器 loss----------------------------------
def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                       if s is not None])

    non_final_next_states = torch.reshape(non_final_next_states, (BATCH_SIZE, state_space))
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_batch = torch.reshape(state_batch, (BATCH_SIZE, state_space))
    action_batch = torch.reshape(action_batch, (BATCH_SIZE, 1))

    state_action_values = policy_net(state_batch.float()).gather(1, action_batch)
    # print(policy_net(state_batch.float()))
    # print("s_a", state_action_values[1])
    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch
    if steps_done % 6000 == 0:
        print('state_action_values', state_action_values[:100,0])
    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
    # print("loss",loss)
    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


# 设置memory
memory = ReplayMemory(60000)

# 训练轮次
num_episodes = 1000
# 每一轮多少步
count = 300
# 训练
for i_episode in range(num_episodes):
    # 初始化环境
    env = EdgeEnv(edgeNodes, users, edgeNum=EDGENODE_NUM, userNum=USER_NUM)
    print('-------------- 回合 {} 开始------------------'.format(i_episode))
    state = env.initState()
    init_energy = env.energy_computing_total()
    init_time = env.time_computing_total()
    if i_episode == 0:
        reward, next_state, done = env.step(0)
        writer.add_scalar("train_reward", reward, i_episode)
        writer.add_scalar("train_reward_time",init_time, i_episode)
        writer.add_scalar("train_reward_energy", init_energy, i_episode)
    f_reword = 0
    for step in range(count):
        # policy network
        action = select_action(state)
        # action = np.random.randint(0, 18)
        reward, next_state, done = env.step(action=action)
        f_reword = f_reword + reward
        # Store the transition in memory
        memory.push(state, action, next_state, torch.tensor([reward]))
        # Move to the next state
        state = next_state
        # 优化器 优化模型参数
        optimize_model()
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())
    energy = env.energy_computing_total()
    time = env.time_computing_total()
    print("state", state)
    print("f_reward:", f_reword)
    writer.add_scalar("train_reward", f_reword, i_episode+1)
    writer.add_scalar("train_reward_time", time, i_episode+1)
    writer.add_scalar("train_reward_energy", energy, i_episode+1)

writer.close()
print("done ")
