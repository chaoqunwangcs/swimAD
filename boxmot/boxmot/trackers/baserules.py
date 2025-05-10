from abc import ABC
from itertools import islice

import pdb

class BaseRules(ABC):
    def __init__(self):
        pass

    def rule1(self, history_observations) -> bool:
        # 连续10秒中，超过80%的帧数人都在水下
        window_size = 20
        cls_thres = 0.8
        if len(history_observations) > window_size:
            history_observations_list = list(islice(history_observations, len(history_observations) - window_size, len(history_observations)))
            clses = [x[-1]==2 for x in history_observations_list]
            if sum(clses) / window_size >= cls_thres:
                return True
        
        return False
    
    def rule2(self, history_observations) -> bool:
        # 10秒内的移动距离小于100个像素，且判别为水下的人的概率大于50%
        window_size = 20
        dist_thres = 100
        cls_thres = 0.5
        if len(history_observations) > window_size:
            history_observations_list = list(islice(history_observations, len(history_observations) - window_size, len(history_observations)))
            clses = [x[-1]==2 for x in history_observations]
            box_0 = history_observations_list[0]
            box_1 = history_observations_list[1]
            distance = self.calc_dist(box_0, box_1)
            if distance <= dist_thres and sum(clses) / window_size >= cls_thres:
                return True
        
        return False