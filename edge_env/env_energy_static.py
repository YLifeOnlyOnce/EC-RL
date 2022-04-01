'''
Author: yaoyaoyu
Date: 2022-03-31 13:20:00
Desc: edge env To Energy
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class EdgeEnv:
    def __init__(self, edgeNodes, users, edgeNum, userNum):
        self.edgeNodes = edgeNodes  # 边缘节点
        self.users = users  # 用户列表，包括各自的用户任务
        self.edgeNum = edgeNum  # 边缘节点 个数
        self.userNum = userNum  # 边缘节点的用户数

        self.state = []  # 状态空间

        self.cov_matrix = []  # 边缘节点 与 用户的覆盖关系
        self.init_edgeComputingAllocation_matrix = []  # 初始化的 边缘计算资源分发
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
        # 任务卸载的计算资源
        self.offload_computing_resource = torch.zeros(self.userNum, self.edgeNum, dtype=torch.float)
        # 计算资源分配 int
        self.edgeComputingAllocation_matrix = torch.zeros(self.userNum, self.edgeNum, dtype=torch.float)
        self.initComputingAllocation()
        # print(self.edgeComputingAllocation_matrix)
        self.state = torch.cat([self.offload_percent_matrix.flatten(), self.edgeComputingAllocation_matrix.flatten()])
        return self.state

    def initComputingAllocation(self):
        # 初始化计算资源分配
        # 是否有分配计算资源的矩阵
        offload_matrix = self.offload_percent_matrix[:, 1:]
        isOffload_matrix = F.relu(offload_matrix)
        # 计算资源为一个确定的值，先不去分配, 随机一个范围，然后均分
        self.edgeComputingAllocation_matrix = self.edgeComputingAllocation_matrix + torch.randint(22, 24, (
            1, self.edgeNum))
        self.init_edgeComputingAllocation_matrix = self.edgeComputingAllocation_matrix

        r = 0
        self.state = torch.cat([self.offload_percent_matrix.flatten(), self.edgeComputingAllocation_matrix.flatten()])
        return r, self.state, False

    def step(self, action):

        # 计算原有能耗
        e = self.energy_computing_total()
        # 修改卸载资源百分比
        self.changeOffloadPercent(action)
        # 修载卸载到边缘的计算资源的量
        self.changeOffloadComputingResource()
        # 修改边缘节点对任务的计算资源分配
        self.changeEdgeComputingAllocation()
        # 计算现有能耗
        e_next = self.energy_computing_total()
        r = self.reward(e, e_next)
        self.state = torch.cat([self.offload_percent_matrix.flatten(), self.edgeComputingAllocation_matrix.flatten()])
        is_done = False
        return r, self.state, is_done

    # 计算任务所需能耗
    def energy_computing_total(self):
        lamda = 0.12  # 能耗因子
        p_idle = 0.65  # 空闲时 功耗占比
        p_sleep = 0.3  # 休眠时 功耗占比
        E = 0  # 总能耗

        # 用户任务本地执行时间
        E_user = 0
        Sys_time_total = 0
        for u_i in range(self.userNum):
            time_user_comuting = (self.offload_percent_matrix[u_i, 0] * self.users[
                u_i].task.required_computing_resource) / self.users[u_i].computing_resource
            Sys_time_total = max(Sys_time_total, time_user_comuting)
            E_user = E_user + time_user_comuting * 80
        # 总体执行时间 - 两个矩阵做除法 矩阵求最大
        offload_computing_time_matrix = self.offload_computing_resource / self.edgeComputingAllocation_matrix
        offload_computing_time_matrix = torch.where(torch.isnan(offload_computing_time_matrix),
                                                    torch.full_like(offload_computing_time_matrix, 0),
                                                    offload_computing_time_matrix)
        edge_computing_time_total = torch.max(offload_computing_time_matrix)
        Sys_time_total = max(Sys_time_total, edge_computing_time_total)
        # print("total",edge_computing_time_total)
        # 每个边缘节点的执行时间 - 每个任务，也就是每一行 row 求最大
        edge_computing_time_task = torch.amax(offload_computing_time_matrix, 0)
        # print("time",edge_computing_time_task)
        # 对于每一个边缘节点来说，其 休眠状态时间 = 边缘计算总时间 - 该边缘节点的执行时间
        # 对每个边缘节点进行单独处理
        #  从  offload_computing_time_matrix 中的一列，依此找top min
        for i in range(self.edgeNum):
            edge_i_energy = 0
            # 执行时间
            time_running = edge_computing_time_task[i]
            # 休息时间
            time_idle = Sys_time_total - time_running
            # CPU 利用率 = 当前分配出去的计算资源 / 边缘服务器的总计算资源
            resource_now = torch.sum(self.edgeComputingAllocation_matrix[:, i])
            # print("resource_nonw",resource_now)
            if resource_now != 0:
                cpu_rate = resource_now / self.edgeNodes[i].computing_resource
                # 有任务负载时
                edge_i_energy = time_running * ((cpu_rate * lamda + p_idle) * self.edgeNodes[i].power_max)
                # 无任务负载时
                edge_i_energy = edge_i_energy + time_idle * (p_idle * self.edgeNodes[i].power_max)
            else:
                # 没有任务卸载到该服务器，则用总时间✖️待机功耗
                edge_i_energy = Sys_time_total * (p_idle * self.edgeNodes[i].power_max)
            E = E + edge_i_energy

        return E + E_user

    '''
    Reward function：奖励函数
    '''

    def reward(self, e, e_next):
        r = 0
        if e > e_next:
            r = 1
        elif e < e_next:
            r = -1
        elif e == e_next:
            r = -0.1
        return r

    # 执行 action 修改卸载百分比
    def changeOffloadPercent(self, action):
        act_range = 0.01
        userpart = int(action / (2 * self.edgeNum + 1))
        # print("第{}个任务".format(userpart))
        # 第几个任务的第几个动作，2*(5+1) = 18
        act_index = int(action % (2 * self.edgeNum + 1))
        # print("第个任务的{}个动作".format(act_index))
        if act_index == 0:
            # 选择了任务 userpart的本地 则PASS
            pass
        elif 0 < act_index <= self.edgeNum:
            # 加法 - 边缘节点加， 用户本地减
            edge_index = act_index  # 第4个的话， 索引是 4-1
            if self.offload_percent_matrix[userpart, edge_index] != 1 and self.offload_percent_matrix[userpart, 0] >= act_range:
                self.offload_percent_matrix[userpart, 0] = self.offload_percent_matrix[userpart, 0] - act_range
                self.offload_percent_matrix[userpart, edge_index] = self.offload_percent_matrix[
                                                                        userpart, edge_index] + act_range
        elif self.edgeNum < act_index <= 2 * self.edgeNum:
            # 减法 - 边缘节点减， 用户本地加
            edge_index = act_index - self.edgeNum
            if self.offload_percent_matrix[userpart, edge_index] >= act_range and self.offload_percent_matrix[userpart, 0] <= 1-act_range:
                self.offload_percent_matrix[userpart, 0] = self.offload_percent_matrix[userpart, 0] + act_range
                self.offload_percent_matrix[userpart, edge_index] = self.offload_percent_matrix[
                                                                        userpart, edge_index] - act_range

    # 任务卸载到边缘节点的【计算资源】的矩阵
    def changeOffloadComputingResource(self):
        A = lambda x: self.fuc(x)
        tasks_required_computing_resource = torch.tensor([A(x) for x in self.users])
        self.offload_computing_resource = self.offload_percent_matrix[:, 1:] * torch.reshape(
            tasks_required_computing_resource, (-1, 1))
        return None

    # 边缘节点的【计算资源】分配
    def changeEdgeComputingAllocation(self):
        # 边缘中的 卸载矩阵
        offload_matrix = self.offload_percent_matrix[:, 1:]
        # 是否有卸载 生成0-1矩阵
        isOffload_matrix = (offload_matrix != 0).type(torch.float)
        # 计算资源为一个确定的值，先不去分配, 随机一个范围，然后均分
        self.edgeComputingAllocation_matrix = torch.div(self.init_edgeComputingAllocation_matrix,
                                                        torch.sum(isOffload_matrix, dim=0)) * isOffload_matrix
        # 去除 inf 的无穷值
        self.edgeComputingAllocation_matrix = torch.where(torch.isnan(self.edgeComputingAllocation_matrix),
                                                          torch.full_like(self.edgeComputingAllocation_matrix, 0),
                                                          self.edgeComputingAllocation_matrix)
        # print(self.edgeComputingAllocation_matrix)

    @classmethod
    def fuc(self, x):
        return x.task.required_computing_resource
