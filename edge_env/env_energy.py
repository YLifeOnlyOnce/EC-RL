'''
Author: yaoyaoyu
Date: 2022-03-31 13:20:00
Desc: edge env To Energy
'''

import torch
import torch.nn as nn
import torch.nn.functional as F


class EdgeEnv:
    def __init__(self, edgeNodes, users, edgeNum, userNum):
        self.edgeNodes = edgeNodes  # 边缘节点
        self.users = users  # 用户列表，包括各自的用户任务
        self.edgeNum = edgeNum  # 边缘节点 个数
        self.userNum = userNum  # 边缘节点的用户数

        self.state = []  # 状态空间

        self.cov_matrix = []  # 边缘节点 与 用户的覆盖关系
        self.edgeComputingAllocation_matrix = []  # 边缘节点-任务 计算资源分发
        self.offload_percent_matrix = []  # 任务-边缘节点 卸载百分比
        self.offload_computing_resource = []  # 任务-边缘节点 卸载计算量

    def initEnv(self):
        # 初始化边缘节点，用户及任务等参数。
        return None

    def initState(self):
        # 计算边缘节点可卸载位置 - 初始化  - 全允许的状态
        self.cov_matrix = torch.ones((self.userNum, self.edgeNum), dtype=torch.int32)
        # 边缘节点卸载百分比，边缘节点
        self.offload_percent_matrix = torch.zeros((self.userNum, self.edgeNum + 1), dtype=torch.float)
        self.offload_percent_matrix[:, 0] = 1
        # 计算资源分配 int
        self.edgeComputingAllocation_matrix = torch.zeros(self.userNum, self.edgeNum, dtype=torch.float)
        self.initComputingAllocation()

        return None

    def initComputingAllocation(self):
        # 初始化计算资源分配
        # 是否有分配计算资源的矩阵
        # offload_matrix = self.offload_percent_matrix[:, 1:]
        # isOffload_matrix = F.relu(offload_matrix)

        # 计算资源为一个确定的值，先不去分配
        self.edgeComputingAllocation_matrix = self.edgeComputingAllocation_matrix + torch.randint(14, 24, (
            self.userNum, self.edgeNum))
        return None

    def step(self, action):
        # 修改卸载资源百分比
        self.changeOffloadPercent(action)
        # 修载卸载到边缘的计算资源的量
        self.changeOffloadComputingResource()
        r = 0
        next_state = 0
        is_done = False
        return r, next_state, is_done

    # 计算任务所需能耗
    def energy_computing_total(self):

        # 总体执行时间 - 两个矩阵做除法 矩阵求最大
        offload_computing_time_matrix = (self.offload_computing_resource / self.edgeComputingAllocation_matrix)
        edge_computing_time_total = torch.max(offload_computing_time_matrix)
        # 每个边缘节点的执行时间 - 每个任务，也就是每一行 row 求最大
        edge_computing_time_task = torch.amax(offload_computing_time_matrix, 1)
        # 对于每一个边缘节点来说，其 休眠状态时间 = 边缘计算总时间 - 该边缘节点的执行时间
        # 对每个边缘节点进行单独处理
        #  从  offload_computing_time_matrix 中的一列，依此找top min
        for i in range(self.edgeNum):
            # 每一列
            temp_time_i = offload_computing_time_matrix[:,i]
            lowest_min = torch.min(temp_time_i)
            # CPU 利用率 = 当前分配出去的计算资源 / 边缘服务器的总计算资源
            resource_now = torch.sum(self.edgeComputingAllocation_matrix[:, i])
            cpu_rate = resource_now / self.edgeNodes[i].computing_resource
            print(cpu_rate)
        E = 0

        return E

    # 计算任务所需能耗
    def energy_computing_total_none_sleep(self):

        # 总体执行时间 - 两个矩阵做除法 矩阵求最大
        offload_computing_time_matrix = (self.offload_computing_resource / self.edgeComputingAllocation_matrix)
        edge_computing_time_total = torch.max(offload_computing_time_matrix)
        # 每个边缘节点的执行时间 - 每个任务，也就是每一行 row 求最大
        edge_computing_time_task = torch.amax(offload_computing_time_matrix, 1)
        # 对于每一个边缘节点来说，其 休眠状态时间 = 边缘计算总时间 - 该边缘节点的执行时间
        # 对每个边缘节点进行单独处理
        #  从  offload_computing_time_matrix 中的一列，依此找top min
        for i in range(self.edgeNum):
            # 每一列
            temp_time_i = offload_computing_time_matrix[:, i]
            lowest_min = torch.min(temp_time_i)
            # CPU 利用率 = 当前分配出去的计算资源 / 边缘服务器的总计算资源
            resource_now = torch.sum(self.edgeComputingAllocation_matrix[:, i])
            cpu_rate = resource_now / self.edgeNodes[i].computing_resource
            print(cpu_rate)
            E = 0

            return E

    '''
    function：奖励函数
    '''

    def reward(self):
        r = self.energy_computing_total()
        return r

    # 执行 action 修改卸载百分比
    def changeOffloadPercent(self, action):
        userpart = int(action / (3 * (self.edgeNum + 1)))
        # print("第{}个任务".format(userpart))
        # 第几个任务的第几个动作，3*6 = 18
        act_index = action % (3 * (self.edgeNum + 1))
        # print("第个任务的{}个动作".format(act_index))
        # 第几个边缘节点
        edgeindex = act_index % (self.edgeNum + 1)
        # print('第{}个卸载位置'.format(edgeindex))
        # 哪一种动作
        act = int(act_index / (self.edgeNum + 1))
        # self.offload_percent_matrix[]
        # print(act)

        if edgeindex != 0:
            # 执行动作。P[userpart][edgeindex] act + -
            if act == 0:
                pass
            elif act == 1:
                self.offload_percent_matrix[userpart, edgeindex] = min(
                    self.offload_percent_matrix[userpart, edgeindex] + 0.01, 1.00)
                self.offload_percent_matrix[userpart, 0] = max(self.offload_percent_matrix[userpart, 0] - 0.01, 0.00)
            elif act == 2:
                self.offload_percent_matrix[userpart, edgeindex] = max(
                    self.offload_percent_matrix[userpart, edgeindex] - 0.01, 0.00)
                self.offload_percent_matrix[userpart, 0] = min(self.offload_percent_matrix[userpart, 0] + 0.01, 1.00)
        print(self.offload_percent_matrix)

    def changeOffloadComputingResource(self):
        A = lambda x: self.fuc(x)
        tasks_required_computing_resource = torch.tensor([A(x) for x in self.users])
        self.offload_computing_resource = self.offload_percent_matrix[:, 1:] * torch.reshape(
            tasks_required_computing_resource, (-1, 1))
        return None

    @classmethod
    def fuc(self, x):
        return x.task.required_computing_resource
