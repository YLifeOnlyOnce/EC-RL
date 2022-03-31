'''
Author: yaoyaoyu
Date: 2022-03-28 14:59:18
Desc: edge task
'''


class Task:
    def __init__(self, name, required_computing_resource, trans_data_size, hard_deadline) -> None:
        self.name = name
        self.required_computing_resource = required_computing_resource
        self.trans_data_size = trans_data_size
        self.hard_deadline = hard_deadline
        
        
