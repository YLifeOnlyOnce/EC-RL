import torch
import numpy as np


class EdgeEnv:
    def __init__(self, edgeNodes, users, edgeNum, userNum):
        self.edgeNodes = edgeNodes  # 边缘节点
        self.users = users  # 用户列表，包括各自的用户任务
        self.edgeNum = edgeNum  # 边缘节点 个数
        self.userNum = userNum  # 边缘节点的用户数

        self.state = []  # 状态空间

        self.cov_matrix = []  # 边缘节点 与 用户的覆盖关系
        self.edgeComputingAllocation_matrix = []  # 边缘节点-任务 计算资源分发
        self.offload_percent_matrix = []  # 卸载百分比

    def showDetail(self):
        # print(self.edgeNodes, self.users)
        return

    def initState(self):

        # 计算每个任务可卸载的边缘节点位置
        self.cov_matrix = torch.ones([self.userNum, self.edgeNum], dtype=torch.int16)
        self.edgeComputingAllocation_matrix = torch.zeros([self.userNum, self.edgeNum], dtype=torch.int32)
        self.offload_percent_matrix = torch.zeros([self.userNum, self.edgeNum + 1], dtype=torch.float)
        self.offload_percent_matrix[:, 0] = 1.00
        # print(self.edgeComputingAllocation_matrix)
        # print(self.cov_matrix)
        # print(self.offload_percent_matrix)
        # 修改边缘节点的计算资源分配
        self.changeComputingAllocation()
        # 构造 state
        self.state = torch.cat([self.offload_percent_matrix.flatten(), self.edgeComputingAllocation_matrix.flatten()])
        return self.state

    # ------------------
    # 输入动作
    # 返回 状态 state 和 reward
    # -------------------
    def step(self, action):
        # 执行action ，卸载百分比更改，边缘服务器分配计算资源更改，
        # 单个任务 1，2，3，4，5 ｜ 6，7，8，9，10 ｜11，12，13，14，15 ｜｜ 16，17，18
        # 第几个任务

        self.changeOffloadPercent(action)

        r = self.reward(action)
        self.state = torch.cat([self.offload_percent_matrix.flatten(), self.edgeComputingAllocation_matrix.flatten()])

        return r, self.state, False

    def getCovMatrix(self):
        return

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
        # print(self.offload_percent_matrix)

    def changeComputingAllocation(self):
        self.edgeComputingAllocation_matrix = self.edgeComputingAllocation_matrix + torch.randint(14, 24, (
        self.userNum, self.edgeNum))

        return

    # ------ 能耗计算 ----------
    def computing_energy_total(self):
        energy_total = 0
        # 能耗计算方法
        # 本地执行能耗 与 边缘计算能耗
        # 预期是本地执行时间不够，边缘执行才能满足时间， 因此不管本地能耗会有多小，都不影响必须要卸载到边缘

        return energy_total

    def computing_time_task(self, i):
        # 时间计算 1。单个任务时间，一个函数
        # 2。 任务总体完成时间，一个函数
        t_local = (self.offload_percent_matrix[i, 0] * self.users[i].task.required_computing_resource) / self.users[
            i].computing_resource

        # 卸载的子任务
        A = lambda x: self.fuc(x)
        tasks_required_computing_resource = [A(x) for x in self.users]

        offload_computing_resource_matrix = self.offload_percent_matrix[i, 1:] * torch.tensor(
            tasks_required_computing_resource)
        # print("卸载的计算资源:{}".format( offload_computing_resource_matrix )  )

        # 远程计算时间计算
        t_edge_total = torch.max(offload_computing_resource_matrix[i] / self.edgeComputingAllocation_matrix[i])
        # print(t_edge_total)

        t_total_i = torch.max(t_local, t_edge_total)
        return t_total_i

    @classmethod
    def fuc(self, x):
        return x.task.required_computing_resource

    def computing_time_total(self):

        return

    def reward(self, action):
        # reward function
        # 计算当前能耗

        # 计算当前时延

        # computing_time_alltask
        r = - self.computing_time_task(0)
        return r
