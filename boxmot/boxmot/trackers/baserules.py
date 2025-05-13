from abc import ABC
from itertools import islice
from copy import deepcopy
from boxmot.utils import logger as LOGGER
import numpy as np 


import pdb

class BaseRules(ABC):
    def __init__(self):
        pass

    def calc_dist(self, box1, box2):
        center1 = np.array([(box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2])
        center2 = np.array([(box2[0] + box2[2]) / 2, (box2[1] + box2[3]) / 2])
        distance = np.linalg.norm(center1 - center2)
        return distance

    def rule1(self, history_observations, track_id) -> bool:
        # 连续10秒中，超过80%的帧数人都在水下

        INFO = dict()
        TIME_PERIOD = 10
        FPS = 2
        RELATIVE_RATIO = 2
        CLS_THRES = 0.8
        window_size = int(TIME_PERIOD * FPS)
        UNDER_WATER_CLS = 2

        # init_info
        INFO['id'] = track_id
        INFO['traj_len'] = min(len(history_observations), window_size)
        INFO['max_dist'] = -1
        INFO['min_dist'] = -1
        INFO['move_dist'] = -1
        INFO['avg_scale'] = -1
        INFO['cls_list'] = []
        INFO['scale_list'] = []

        if len(history_observations) > window_size:
            history_observations_list = list(islice(history_observations, len(history_observations) - window_size, len(history_observations)))
            box_scales = np.array([((x[2]-x[0])+(x[3]-x[1]))/2.0 for x in history_observations_list]) # (w+h)/2
            box_location = np.array([((x[0]+x[2])/2.0,(x[1]+x[3])/2.0) for x in history_observations_list])
            avg_box_scale = sum(box_scales) / len(box_scales)
            distances = np.sqrt((box_location[:, np.newaxis, 0] - box_location[:, 0])**2 + (box_location[:, np.newaxis, 1] - box_location[:, 1])**2)
            step_distances = np.linalg.norm(box_location[1:] - box_location[:-1], axis=1)
            max_dist = np.max(distances)
            DIST_FLAG = (max_dist < avg_box_scale * RELATIVE_RATIO)  # moving distances lower than 2 average box scale

            clses = [x[-1] for x in history_observations_list]
            CLS_FLAG = (sum([x==UNDER_WATER_CLS for x in clses]) / window_size >= CLS_THRES)

            # pdb.set_trace()
            # update the info
            INFO['max_dist'] = max_dist
            INFO['min_dist'] = np.min(step_distances)
            INFO['move_dist'] = np.sqrt((box_location[0][0]-box_location[-1][0])**2 + (box_location[0][1]-box_location[-1][1])**2)
            INFO['avg_scale'] = avg_box_scale
            INFO['cls_list'] = np.around(np.array(clses), decimals=0).tolist()
            INFO['scale_list'] = np.around(box_scales, decimals=2).tolist()

            
            return (DIST_FLAG and CLS_FLAG), INFO
        
        return False, INFO