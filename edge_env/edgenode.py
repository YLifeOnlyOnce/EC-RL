'''
Author: yaoyaoyu
Date: 2022-03-28 14:59:18
Desc: edge computing node
'''


class EdgeNode:
    def __init__(self, computing_resource, location_x, location_y, access_range, bandwidth, energy_factor) -> None:
        self.computing_resource = computing_resource                    # 计算资源总量
        self.available_computing_resource = computing_resource          # 可用计算资源
        self.location_x = location_x                                    # 坐标 x
        self.location_y = location_y                                    # 坐标 y
        self.bandwidth = bandwidth                                      # 可用带宽
        self.access_range = access_range                                # 通信范围
        self.computing_task = []                                        # 所有的计算任务
        self.energy_factor = energy_factor                              # 能耗因子
