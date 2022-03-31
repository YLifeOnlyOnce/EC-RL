'''
Author: yaoyaoyu
Date: 2022-03-31 13:20:00
Desc: agent 与环境交互
'''

from edge_env.env_energy_static import EdgeEnv
from edge_env.edgenode import EdgeNode
from edge_env.task import Task
from edge_env.user import User

# --------- 初始化 用户和节点 -------------
users = []
tasks = []
edgeNodes = []

USER_NUM = 3
TASKS_NUM = 1
EDGENODE_NUM = 5

for index in range(USER_NUM):
    task = Task('task' + str(index), 100 + int(index), 1000, 5000)
    users.append(User('user' + str(index), 5, 45, 45, task))

for index in range(EDGENODE_NUM):
    edgenode = EdgeNode(24, 45, 45, 300, 2000, energy_factor=0.2)
    edgeNodes.append(edgenode)
# ---------- 初始化用户和节点 end -----------

# 初始化环境
env = EdgeEnv(edgeNodes=edgeNodes, users=users, userNum=USER_NUM, edgeNum=EDGENODE_NUM)

env.initState()

env.step(26)
env.step(26)
env.step(26)
env.step(26)
env.step(26)
env.step(8)
env.step(8)
env.step(8)
env.step(9)

env.energy_computing_total()
