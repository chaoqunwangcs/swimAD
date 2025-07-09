from abc import ABC
from itertools import islice
from copy import deepcopy
from boxmot.utils import logger as LOGGER
import numpy as np 


import pdb

class BaseRules(ABC):
    Window_Size = 10
    
    def __init__(self):
        pass

    def min_dist(self, history_observations, track_id):
        """window_size下的各点间最小欧氏距离"""
        if len(history_observations) == 1:
            return 0
        else:
            obs = list(history_observations)
            window = obs[-self.Window_Size:]
            data = np.stack(window)
            midpoints = np.mean(data[:,:4].reshape(data.shape[0], 2, 2), axis=1)
            distances = np.sqrt(np.sum((midpoints[:, np.newaxis] - midpoints)**2, axis=2))
            return np.min(distances)
    
    def max_dist(self, history_observations, track_id):
        """window_size下的各点间最大欧氏距离"""
        if len(history_observations) == 1:
            return 0
        else:
            obs = list(history_observations)
            window = obs[-self.Window_Size:]
            data = np.stack(window)
            midpoints = np.mean(data[:,:4].reshape(data.shape[0], 2, 2), axis=1)
            distances = np.sqrt(np.sum((midpoints[:, np.newaxis] - midpoints)**2, axis=2))
            return np.max(distances)
    
    def class_label(self, history_observations, track_id):
        return int(history_observations[-1][-1])
    
    def costheta(self, history_observations, track_id):
        """方向一致性的cosθ指标,计算box最近两帧之间的cos值"""
        if len(history_observations) < 3:
            return 0
        else:
            obs = list(history_observations)
            window = obs[-self.Window_Size:]
            centers = np.array([[(x[0]+x[2])/2, (x[1]+x[3])/2] for x in window])
            vecs = centers[1:] - centers[:-1]    # shape=(N-1,2) 计算相邻帧之间的位移向量
            v_prev, v_cur = vecs[-2], vecs[-1] #  只取最后两段向量
            
            dot = float(np.dot(v_prev, v_cur))#  点积
            norm_val = float(np.linalg.norm(v_prev) * np.linalg.norm(v_cur))#模长
            if norm_val <= 1e-6:
                return 0.0
        cos_val = dot / norm_val       
        return float(np.clip(cos_val, -1.0, 1.0))# 裁剪到 [-1,1] 并返回单个值
        '''
        v_prev, v_cur = vecs[:-1], vecs[1:]
        dot = np.sum(v_prev * v_cur, axis=1) #点积
        norm = np.linalg.norm(v_prev, axis=1) * np.linalg.norm(v_cur, axis=1)#模长
        cos = np.ones_like(dot)# 计算余弦值
        valid = norm > 1e-6
        cos[valid] = dot[valid] / norm[valid]
        #return np.clip(cos, -1.0, 1.0).tolist() #  裁剪到[-1,1]并返回列表               
        '''
        
    
    def move_dist(self, history_observations, track_id):
        """window_size下的最后两帧中心点移动距离"""
        if len(history_observations) == 1:
            return 0
        else:
            obs = list(history_observations)
            window = obs[-self.Window_Size:]
            b1, b2 = window[-2][:4], window[-1][:4]
            c1 = np.array([(b1[0] + b1[2]) / 2.0, (b1[1] + b1[3]) / 2.0])
            c2 = np.array([(b2[0] + b2[2]) / 2.0, (b2[1] + b2[3]) / 2.0])
            return float(np.linalg.norm(c2 - c1))    

    def avg_scale(self, history_observations, track_id):
        """滑窗内平均检测框大小：(w+h)/2 平均值"""
        if len(history_observations) == 1:
            return 0
        else:            
            obs = list(history_observations)
            window = obs[-self.Window_Size:]
            scales = [((x[2]-x[0])+(x[3]-x[1]))/2.0 for x in window]
            return float(np.mean(scales)) if scales else 0.0

    def condition_A_triggered(self, history_observations, track_id):
        pass

    def condition_B_triggered(self, history_observations, track_id):
        pass

    def final_ema_magnitude(self, history_observations, track_id):
        pass



    def calc_dist(self, box1, box2):
        center1 = np.array([(box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2])
        center2 = np.array([(box2[0] + box2[2]) / 2, (box2[1] + box2[3]) / 2]) 
        distance = np.linalg.norm(center1 - center2)
        return distance

    def rule1(self, history_observations, track_id) -> tuple[bool, dict]:
        # Original parameters
        INFO = dict()
        TIME_PERIOD = 10  # seconds
        FPS = 2
        # RELATIVE_RATIO = 2 # Original parameter for DIST_FLAG, may not be central to new logic
        CLS_THRES = 0.8
        window_size = int(TIME_PERIOD * FPS)
        UNDER_WATER_CLS = 2

        # New parameters for Mobile Vector Trend Analysis
        ALPHA_EMA = 0.2  # Decay factor for EMA of magnitude
        COS_SIM_LOW_THRESHOLD = 0.5  # Cosine similarity: if lower, considered fluctuating
        # Threshold for "small magnitude" for an individual displacement vector, relative to average box scale
        MAG_SMALL_FACTOR = 0.3
        # Threshold for "EMA magnitude approaches zero", relative to average box scale
        EMA_MAG_LOW_FACTOR = 0.1
        # Ratio of frames that must show (fluctuating angle AND small magnitude) for Condition A
        FLUCT_SMALL_FRAMES_RATIO = 0.7
        MIN_DISPLACEMENT_VECTORS_FOR_COS_THETA = 2 # Need at least 2 vectors for 1 cosine angle
        MIN_POINTS_FOR_VECTOR_ANALYSIS = MIN_DISPLACEMENT_VECTORS_FOR_COS_THETA + 1 # Need N points for N-1 vectors

        # init_info
        INFO['id'] = track_id
        INFO['traj_len'] = min(len(history_observations), window_size)
        INFO['max_dist'] = -1       # From original logic
        INFO['min_dist'] = -1       # From original logic (step_distances)
        INFO['move_dist'] = -1      # From original logic (start to end of window)
        INFO['avg_scale'] = -1
        INFO['cls_list'] = []
        INFO['scale_list'] = []
        INFO['is_AD'] = False       # AD for Anomaly Detection / Drowning

        # New INFO fields for drowning detection logic
        INFO['cos_theta_values'] = []
        INFO['displacement_magnitudes_list'] = []
        INFO['final_ema_magnitude'] = -1.0
        INFO['condition_A_triggered'] = False
        INFO['condition_B_triggered'] = False
        INFO['new_movement_drowning_flag'] = False
        INFO['cls_flag_triggered'] = False


        if len(history_observations) >= window_size : # Use >= for consistency with islice, though > was in original
            # Slice to get the current window of observations
            history_observations_list = list(islice(history_observations, len(history_observations) - window_size, len(history_observations)))

            box_scales = np.array([((x[2]-x[0])+(x[3]-x[1]))/2.0 for x in history_observations_list]) # (w+h)/2
            box_location = np.array([((x[0]+x[2])/2.0,(x[1]+x[3])/2.0) for x in history_observations_list])
            
            if len(box_scales) == 0: # Should not happen if window_size > 0 and history_observations_list is populated
                return False, INFO
                
            avg_box_scale = np.mean(box_scales) if len(box_scales) > 0 else 1.0 # Avoid division by zero if no scales

            # Update common INFO fields
            INFO['avg_scale'] = avg_box_scale
            clses = [x[-1] for x in history_observations_list]
            INFO['cls_list'] = [int(c) for c in clses] # Ensure integers
            INFO['scale_list'] = np.around(box_scales, decimals=2).tolist()

            # --- CLS_FLAG (Underwater classification) ---
            CLS_FLAG = (sum([x==UNDER_WATER_CLS for x in clses]) / window_size >= CLS_THRES)
            INFO['cls_flag_triggered'] = CLS_FLAG

            # --- New Drowning Detection Logic based on Image ---
            condition_A_flag = False
            condition_B_flag = False
            
            # Ensure enough points for vector analysis
            if len(box_location) >= MIN_POINTS_FOR_VECTOR_ANALYSIS:
                displacement_vectors = box_location[1:] - box_location[:-1]
                magnitudes = np.linalg.norm(displacement_vectors, axis=1)
                INFO['displacement_magnitudes_list'] = magnitudes.tolist()

                # 1. Direction Consistency Evaluation (Condition A)
                if len(displacement_vectors) >= MIN_DISPLACEMENT_VECTORS_FOR_COS_THETA:
                    v_current = displacement_vectors[1:]
                    v_previous = displacement_vectors[:-1]
                    
                    mag_current = magnitudes[1:]
                    mag_previous = magnitudes[:-1]
                    
                    dot_products = np.sum(v_current * v_previous, axis=1)
                    denominators = mag_current * mag_previous
                    
                    cos_thetas = np.ones(len(denominators)) # Default to 1 (no change)
                    valid_indices = denominators > 1e-9 # Avoid division by zero or near-zero
                    
                    cos_thetas[valid_indices] = dot_products[valid_indices] / denominators[valid_indices]
                    cos_thetas = np.clip(cos_thetas, -1.0, 1.0) # Ensure valid cosine values
                    INFO['cos_theta_values'] = cos_thetas.tolist()

                    # Check for "angle fluctuation large AND magnitude small"
                    # Magnitudes to check are mag_current (corresponding to v_t, which forms angle with v_{t-1})
                    fluctuating_and_small_count = 0
                    # Number of angles calculated = len(cos_thetas)
                    # Corresponding magnitudes for v_t are magnitudes[1:]
                    for i in range(len(cos_thetas)):
                        angle_fluctuates = cos_thetas[i] < COS_SIM_LOW_THRESHOLD
                        # Use magnitude of the current vector in the pair that forms the angle
                        # v_previous = displacement_vectors[i], v_current = displacement_vectors[i+1]
                        # Angle is between these. Magnitude small refers to current movements.
                        # Let's use magnitudes of v_current, which is magnitudes[i+1]
                        current_magnitude_is_small = magnitudes[i+1] < (MAG_SMALL_FACTOR * avg_box_scale)
                        if angle_fluctuates and current_magnitude_is_small:
                            fluctuating_and_small_count += 1
                    
                    if len(cos_thetas) > 0: # Avoid division by zero if no angles
                        if (fluctuating_and_small_count / len(cos_thetas)) >= FLUCT_SMALL_FRAMES_RATIO:
                            condition_A_flag = True
                INFO['condition_A_triggered'] = condition_A_flag

                # 2. Magnitude Attenuation Detection (Condition B)
                if len(magnitudes) > 0:
                    current_ema_magnitude = magnitudes[0] # Initialize EMA
                    for i in range(1, len(magnitudes)):
                        current_ema_magnitude = ALPHA_EMA * magnitudes[i] + (1 - ALPHA_EMA) * current_ema_magnitude
                    
                    INFO['final_ema_magnitude'] = current_ema_magnitude
                    if current_ema_magnitude < (EMA_MAG_LOW_FACTOR * avg_box_scale):
                        condition_B_flag = True
                INFO['condition_B_triggered'] = condition_B_flag

            new_movement_drowning_flag = condition_A_flag or condition_B_flag
            INFO['new_movement_drowning_flag'] = new_movement_drowning_flag
            
            # Final Drowning Determination
            INFO['is_AD'] = (new_movement_drowning_flag and CLS_FLAG)
            
            # For completeness, recalculate original INFO fields if desired or remove if not needed.
            # Example: Re-calculate max_dist based on the current window
            if len(box_location) > 1:
                # This calculates distance from each point to all other points in the window
                # More computationally intensive than just start-to-end or step distances
                # Consider what max_dist truly represents in your context.
                # The original code's distances matrix can be large: window_size x window_size
                # For now, let's compute overall displacement as in original 'move_dist'
                # and max step distance.
                INFO['move_dist'] = np.linalg.norm(box_location[0] - box_location[-1])
                step_distances = np.linalg.norm(box_location[1:] - box_location[:-1], axis=1)
                if step_distances.size > 0:
                    INFO['max_dist'] = np.max(step_distances) # Max instantaneous speed proxy
                    INFO['min_dist'] = np.min(step_distances) # Min instantaneous speed proxy
                else:
                    INFO['max_dist'] = 0
                    INFO['min_dist'] = 0
            elif len(box_location) == 1:
                 INFO['move_dist'] = 0
                 INFO['max_dist'] = 0
                 INFO['min_dist'] = 0


            return INFO['is_AD'], INFO
        
        # If not enough history_observations for a full window
        return False, INFO
    
'''
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
        INFO['is_AD'] = False

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
            INFO['is_AD'] = (DIST_FLAG and CLS_FLAG)

            
            return (DIST_FLAG and CLS_FLAG), INFO
        
        return False, INFO
'''
