'''
Author: yaoyaoyu
Date: 2022-03-28 16:20:02
'''

import sys

sys.path.append('/Users/xiangbo/ReinforcementLearning/edge_env')
from edgeNode import EdgeNode
from task import Task
from edge_env_train import EdgeEnv
from user import User

USER_NUM = 1
TASKS_NUM = 1
EDGENODE_NUM = 5

users = []
tasks = []
edgeNodes = []

for index in range(USER_NUM):
    task = Task('task' + str(index), 4000, 1000, 5000)
    users.append(User('user' + str(index), 5000, 45, 45, task))

for index in range(EDGENODE_NUM):
    edgenode = EdgeNode(24000, 45, 45, 300, 2000, energy_factor=0.2)
    edgeNodes.append(edgenode)

env = EdgeEnv(edgeNodes, users, edgeNum=EDGENODE_NUM, userNum=USER_NUM)

env.showDetail()

state = env.initState()
env.step(8)
env.step(8)
r, state_next, done = env.step(8)

print("s", state, "a", 8, "reward", r, "s_t+1", state_next)

print('main')
