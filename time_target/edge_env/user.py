'''
Author: yaoyaoyu
Date: 2022-03-28 16:11:59
'''

# 用户编号，用户本地计算资源，用户地理位置 (location_x,location_y), 用户任务task
class User:
    def __init__(self, name, computing_resource, location_x, location_y, task) -> None:
        self.name = name
        self.computing_resource = computing_resource
        self.location_x = location_x
        self.location_y = location_y
        self.task = task
