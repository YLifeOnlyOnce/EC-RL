'''
Author: yaoyaoyu
Date: 2022-03-28 14:59:18
Desc: edge computing node
'''

'''
desc：边缘节点
params: 计算资源，地址坐标x，地址坐标y
'''

class EdgeNode:
    def __init__(self, computing_resource, location_x, location_y, communication_range, bandwidth, energy_factor ) -> None:
        self.computing_resource = computing_resource
        self.available_computing_resource = computing_resource
        self.location_x = location_x
        self.location_y = location_y
        self.bandwidth = bandwidth
        self.communication_range = communication_range
        self.computing_task = []
        self.energy_factor = energy_factor
        
        
    def offloadTask(self,task):
        # 是否需要判断能否在 期限内完成

        # 为任务分配的计算资源
        
        return
    
    
    # 可用计算资源
    def getAvailableComputingResource(self):
        return self.available_computing_resource;
    
    def getEnergy(self):
        return 10000
      