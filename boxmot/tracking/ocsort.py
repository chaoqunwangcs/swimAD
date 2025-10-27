# Mikel Broström 🔥 Yolo Tracking 🧾 AGPL-3.0 license

"""
    This script is adopted from the SORT script by Alex Bewley alex@bewley.ai
"""
from __future__ import absolute_import, division

import sys
import hashlib
import colorsys
import yaml

import cv2 as cv
import numpy as np
import numpy.linalg as linalg

from copy import deepcopy
from itertools import islice
from collections import deque
from abc import ABC, abstractmethod
from math import log, exp, sqrt, pi 
from filterpy.stats import logpdf
from filterpy.common import pretty_str, reshape_z
from numpy import dot, zeros, eye, isscalar, shape

from boxmot.utils import logger as LOGGER

import pdb

def linear_assignment(cost_matrix):
    try:
        import lap
        _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
        return np.array([[y[i], i] for i in x if i >= 0])  #
    except ImportError:
        from scipy.optimize import linear_sum_assignment
        x, y = linear_sum_assignment(cost_matrix)
        return np.array([list(zip(x, y))])

def speed_direction(bbox1, bbox2):
    cx1, cy1 = (bbox1[0] + bbox1[2]) / 2.0, (bbox1[1] + bbox1[3]) / 2.0
    cx2, cy2 = (bbox2[0] + bbox2[2]) / 2.0, (bbox2[1] + bbox2[3]) / 2.0
    speed = np.array([cy2 - cy1, cx2 - cx1])
    norm = np.sqrt((cy2 - cy1) ** 2 + (cx2 - cx1) ** 2) + 1e-6
    return speed / norm

def speed_direction_batch(dets, tracks):
    tracks = tracks[..., np.newaxis]
    CX1, CY1 = (dets[:, 0] + dets[:, 2]) / 2.0, (dets[:, 1] + dets[:, 3]) / 2.0
    CX2, CY2 = (tracks[:, 0] + tracks[:, 2]) / 2.0, (tracks[:, 1] + tracks[:, 3]) / 2.0
    dx = CX1 - CX2
    dy = CY1 - CY2
    norm = np.sqrt(dx**2 + dy**2) + 1e-6
    dx = dx / norm
    dy = dy / norm
    return dy, dx  # size: num_track x num_det

def speed_direction_obb(bbox1, bbox2):
    cx1, cy1 = bbox1[0], bbox1[1]
    cx2, cy2 = bbox2[0], bbox2[1]
    speed = np.array([cy2 - cy1, cx2 - cx1])
    norm = np.sqrt((cy2 - cy1) ** 2 + (cx2 - cx1) ** 2) + 1e-6
    return speed / norm

def k_previous_obs(observations, cur_age, k, is_obb=False):
    if len(observations) == 0:
        if is_obb:
            return [-1, -1, -1, -1, -1, -1]
        else :
            return [-1, -1, -1, -1, -1]
    for i in range(k):
        dt = k - i
        if cur_age - dt in observations:
            return observations[cur_age - dt]
    max_age = max(observations.keys())
    return observations[max_age]

def convert_x_to_bbox(x, score=None):
    """
    Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
      [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
    """
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w
    if score is None:
        return np.array(
            [x[0] - w / 2.0, x[1] - h / 2.0, x[0] + w / 2.0, x[1] + h / 2.0]
        ).reshape((1, 4))
    else:
        return np.array(
            [x[0] - w / 2.0, x[1] - h / 2.0, x[0] + w / 2.0, x[1] + h / 2.0, score]
        ).reshape((1, 5))


def associate(
    detections,
    trackers,
    asso_func,
    iou_threshold,
    velocities,
    previous_obs,
    vdc_weight,
    w,
    h,
    emb_cost=None,
    w_assoc_emb=None,
    aw_off=None,
    aw_param=None,
    
):
    if len(trackers) == 0:
        return (
            np.empty((0, 2), dtype=int),
            np.arange(len(detections)),
            np.empty((0, 5), dtype=int),
        )
    Y, X = speed_direction_batch(detections, previous_obs)
    inertia_Y, inertia_X = velocities[:, 0], velocities[:, 1]
    inertia_Y = np.repeat(inertia_Y[:, np.newaxis], Y.shape[1], axis=1)
    inertia_X = np.repeat(inertia_X[:, np.newaxis], X.shape[1], axis=1)
    diff_angle_cos = inertia_X * X + inertia_Y * Y
    diff_angle_cos = np.clip(diff_angle_cos, a_min=-1, a_max=1)
    diff_angle = np.arccos(diff_angle_cos)
    diff_angle = (np.pi / 2.0 - np.abs(diff_angle)) / np.pi

    valid_mask = np.ones(previous_obs.shape[0])
    valid_mask[np.where(previous_obs[:, 4] < 0)] = 0

    iou_matrix = asso_func(detections, trackers)
    #iou_matrix = iou_batch(detections, trackers)
    scores = np.repeat(detections[:, -1][:, np.newaxis], trackers.shape[0], axis=1)
    # iou_matrix = iou_matrix * scores # a trick sometiems works, we don't encourage this
    valid_mask = np.repeat(valid_mask[:, np.newaxis], X.shape[1], axis=1)

    angle_diff_cost = (valid_mask * diff_angle) * vdc_weight
    angle_diff_cost = angle_diff_cost.T
    angle_diff_cost = angle_diff_cost * scores

    if min(iou_matrix.shape):
        a = (iou_matrix > iou_threshold).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            if emb_cost is None:
                emb_cost = 0
            else:
                emb_cost = emb_cost
                emb_cost[iou_matrix <= 0] = 0
                if not aw_off:
                    emb_cost = compute_aw_max_metric(emb_cost, w_assoc_emb, bottom=aw_param)
                else:
                    emb_cost *= w_assoc_emb

            final_cost = -(iou_matrix + angle_diff_cost + emb_cost)
            matched_indices = linear_assignment(final_cost)
            if matched_indices.size == 0:
                matched_indices = np.empty(shape=(0, 2))

    else:
        matched_indices = np.empty(shape=(0, 2))

    unmatched_detections = []
    for d, det in enumerate(detections):
        if d not in matched_indices[:, 0]:
            unmatched_detections.append(d)
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if t not in matched_indices[:, 1]:
            unmatched_trackers.append(t)

    # filter out matched with low IOU
    matches = []
    for m in matched_indices:
        if iou_matrix[m[0], m[1]] < iou_threshold:
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))
    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)

def xyxy2xysr(x):
    """
    Converts bounding box coordinates from (x1, y1, x2, y2) format to (x, y, s, r) format.

    Args:
        bbox (np.ndarray) or (torch.Tensor): The input bounding box coordinates in (x1, y1, x2, y2) format.
    Returns:
        z (np.ndarray) or (torch.Tensor): The bounding box coordinates in (x, y, s, r) format, where
                                          x, y is the center of the box,
                                          s is the scale (area), and
                                          r is the aspect ratio.
    """
    x = x[0:4]
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    w = y[..., 2] - y[..., 0]  # width
    h = y[..., 3] - y[..., 1]  # height
    y[..., 0] = y[..., 0] + w / 2.0            # x center
    y[..., 1] = y[..., 1] + h / 2.0            # y center
    y[..., 2] = w * h                                  # scale (area)
    y[..., 3] = w / (h + 1e-6)                         # aspect ratio
    y = y.reshape((4, 1))
    return y

class AssociationFunction:
    def __init__(self, w, h, asso_mode="iou"):
        """
        Initializes the AssociationFunction class with the necessary parameters for bounding box operations.
        The association function is selected based on the `asso_mode` string provided during class creation.
        
        Parameters:
        w (int): The width of the frame, used for normalizing centroid distance.
        h (int): The height of the frame, used for normalizing centroid distance.
        asso_mode (str): The association function to use (e.g., "iou", "giou", "centroid", etc.).
        """
        self.w = w
        self.h = h
        self.asso_mode = asso_mode
        self.asso_func = self._get_asso_func(asso_mode)

    @staticmethod
    def iou_batch(bboxes1, bboxes2) -> np.ndarray:
        bboxes2 = np.expand_dims(bboxes2, 0)
        bboxes1 = np.expand_dims(bboxes1, 1)

        xx1 = np.maximum(bboxes1[..., 0], bboxes2[..., 0])
        yy1 = np.maximum(bboxes1[..., 1], bboxes2[..., 1])
        xx2 = np.minimum(bboxes1[..., 2], bboxes2[..., 2])
        yy2 = np.minimum(bboxes1[..., 3], bboxes2[..., 3])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        wh = w * h
        o = wh / (
            (bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1]) +
            (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1]) -
            wh
        )
        return o    

    @staticmethod
    def iou_batch_obb(bboxes1, bboxes2) -> np.ndarray:

        N, M = len(bboxes1), len(bboxes2)

        def wrapper(i, j):
            return iou_obb_pair(i, j, bboxes1, bboxes2)
        
        iou_matrix = np.fromfunction(np.vectorize(wrapper), shape=(N, M), dtype=int)
        return iou_matrix


    @staticmethod
    def run_asso_func(self, bboxes1, bboxes2):
        """
        Runs the selected association function (based on the initialization string) on the input bounding boxes.
        
        Parameters:
        bboxes1: First set of bounding boxes.
        bboxes2: Second set of bounding boxes.
        """
        return self.asso_func(bboxes1, bboxes2)

    def _get_asso_func(self, asso_mode):
        """
        Returns the corresponding association function based on the provided mode string.
        
        Parameters:
        asso_mode (str): The association function to use (e.g., "iou", "giou", "centroid", etc.).
        
        Returns:
        function: The appropriate function for the association calculation.
        """
        ASSO_FUNCS = {
            "iou": AssociationFunction.iou_batch
        }
        return ASSO_FUNCS[self.asso_mode]

class KalmanFilterXYSR(object):
    """ Implements a Kalman filter. You are responsible for setting the
    various state variables to reasonable values; the defaults will
    not give you a functional filter.
    """

    def __init__(self, dim_x, dim_z, dim_u=0, max_obs=50):
        if dim_x < 1:
            raise ValueError('dim_x must be 1 or greater')
        if dim_z < 1:
            raise ValueError('dim_z must be 1 or greater')
        if dim_u < 0:
            raise ValueError('dim_u must be 0 or greater')

        self.dim_x = dim_x
        self.dim_z = dim_z
        self.dim_u = dim_u

        self.x = zeros((dim_x, 1))        # state
        self.P = eye(dim_x)               # uncertainty covariance
        self.Q = eye(dim_x)               # process uncertainty
        self.B = None                     # control transition matrix
        self.F = eye(dim_x)               # state transition matrix
        self.H = zeros((dim_z, dim_x))    # measurement function
        self.R = eye(dim_z)               # measurement uncertainty
        self._alpha_sq = 1.               # fading memory control
        self.M = np.zeros((dim_x, dim_z)) # process-measurement cross correlation
        self.z = np.array([[None]*self.dim_z]).T

        # gain and residual are computed during the innovation step. We
        # save them so that in case you want to inspect them for various
        # purposes
        self.K = np.zeros((dim_x, dim_z)) # kalman gain
        self.y = zeros((dim_z, 1))
        self.S = np.zeros((dim_z, dim_z)) # system uncertainty
        self.SI = np.zeros((dim_z, dim_z)) # inverse system uncertainty

        # identity matrix. Do not alter this.
        self._I = np.eye(dim_x)

        # these will always be a copy of x,P after predict() is called
        self.x_prior = self.x.copy()
        self.P_prior = self.P.copy()

        # these will always be a copy of x,P after update() is called
        self.x_post = self.x.copy()             
        self.P_post = self.P.copy()

        # Only computed only if requested via property
        self._log_likelihood = log(sys.float_info.min)
        self._likelihood = sys.float_info.min
        self._mahalanobis = None

        # keep all observations 
        self.max_obs = max_obs
        self.history_obs = deque([], maxlen=self.max_obs)

        self.inv = np.linalg.inv

        self.attr_saved = None
        self.observed = False
        self.last_measurement = None

    def predict(self, u=None, B=None, F=None, Q=None):
        """
        Predict next state (prior) using the Kalman filter state propagation
        equations.
        Parameters
        ----------
        u : np.array, default 0
            Optional control vector.
        B : np.array(dim_x, dim_u), or None
            Optional control transition matrix; a value of None
            will cause the filter to use `self.B`.
        F : np.array(dim_x, dim_x), or None
            Optional state transition matrix; a value of None
            will cause the filter to use `self.F`.
        Q : np.array(dim_x, dim_x), scalar, or None
            Optional process noise matrix; a value of None will cause the
            filter to use `self.Q`.
        """
        if B is None:
            B = self.B
        if F is None:
            F = self.F
        if Q is None:
            Q = self.Q
        elif isscalar(Q):
            Q = eye(self.dim_x) * Q

        # x = Fx + Bu
        if B is not None and u is not None:
            self.x = dot(F, self.x) + dot(B, u)
        else:
            self.x = dot(F, self.x)

        # P = FPF' + Q
        self.P = self._alpha_sq * dot(dot(F, self.P), F.T) + Q

        # save prior
        self.x_prior = self.x.copy()
        self.P_prior = self.P.copy()

    def freeze(self):
        """
            Save the parameters before non-observation forward
        """
        self.attr_saved = deepcopy(self.__dict__)

    def unfreeze(self):
        if self.attr_saved is not None:
            new_history = deepcopy(list(self.history_obs))
            self.__dict__ = self.attr_saved
            self.history_obs = deque(list(self.history_obs)[:-1], maxlen=self.max_obs)
            occur = [int(d is None) for d in new_history]
            indices = np.where(np.array(occur) == 0)[0]
            index1, index2 = indices[-2], indices[-1]
            box1, box2 = new_history[index1], new_history[index2]
            x1, y1, s1, r1 = box1
            w1, h1 = np.sqrt(s1 * r1), np.sqrt(s1 / r1)
            x2, y2, s2, r2 = box2
            w2, h2 = np.sqrt(s2 * r2), np.sqrt(s2 / r2)
            time_gap = index2 - index1
            dx, dy = (x2 - x1) / time_gap, (y2 - y1) / time_gap
            dw, dh = (w2 - w1) / time_gap, (h2 - h1) / time_gap

            for i in range(index2 - index1):
                x, y = x1 + (i + 1) * dx, y1 + (i + 1) * dy
                w, h = w1 + (i + 1) * dw, h1 + (i + 1) * dh
                s, r = w * h, w / float(h)
                new_box = np.array([x, y, s, r]).reshape((4, 1))
                self.update(new_box)
                if not i == (index2 - index1 - 1):
                    self.predict()
                    self.history_obs.pop()
            self.history_obs.pop()

    def update(self, z, R=None, H=None):
        """
        Add a new measurement (z) to the Kalman filter. If z is None, nothing is changed.
        Parameters
        ----------
        z : np.array
            Measurement for this update. z can be a scalar if dim_z is 1,
            otherwise it must be a column vector.
        R : np.array, scalar, or None
            Measurement noise. If None, the filter's self.R value is used.
        H : np.array, or None
            Measurement function. If None, the filter's self.H value is used.
        """
        
        # set to None to force recompute
        self._log_likelihood = None
        self._likelihood = None
        self._mahalanobis = None

        # append the observation
        self.history_obs.append(z)
        
        if z is None:
            if self.observed:
                """
                Got no observation so freeze the current parameters for future
                potential online smoothing.
                """
                self.last_measurement = self.history_obs[-2]
                self.freeze()
            self.observed = False
            self.z = np.array([[None] * self.dim_z]).T
            self.x_post = self.x.copy()
            self.P_post = self.P.copy()
            self.y = zeros((self.dim_z, 1))
            return

        # self.observed = True
        if not self.observed:
            """
            Get observation, use online smoothing to re-update parameters
            """
            self.unfreeze()
        self.observed = True
        
        if R is None:
            R = self.R
        elif isscalar(R):
            R = eye(self.dim_z) * R
        if H is None:
            z = reshape_z(z, self.dim_z, self.x.ndim)
            H = self.H

        # y = z - Hx
        # error (residual) between measurement and prediction
        self.y = z - dot(H, self.x)

        # common subexpression for speed
        PHT = dot(self.P, H.T)

        # S = HPH' + R
        self.S = dot(H, PHT) + R
        self.SI = self.inv(self.S)

        # K = PH'inv(S)
        self.K = PHT.dot(self.SI)

        # x = x + Ky
        self.x = self.x + dot(self.K, self.y)

        # P = (I-KH)P(I-KH)' + KRK'
        I_KH = self._I - dot(self.K, H)
        self.P = dot(dot(I_KH, self.P), I_KH.T) + dot(dot(self.K, R), self.K.T)

        # save measurement and posterior state
        self.z = deepcopy(z)
        self.x_post = self.x.copy()
        self.P_post = self.P.copy()

        # save history of observations
        self.history_obs.append(z)


class KalmanBoxTrackerOBB(object):
    """
    This class represents the internal state of individual tracked objects observed as oriented bbox.
    """

    count = 0

    def __init__(self, bbox, cls, det_ind, delta_t=3, max_obs=50, Q_xy_scaling = 0.01, Q_a_scaling = 0.01):
        """
        Initialises a tracker using initial bounding box.
        """
        # define constant velocity model
        self.det_ind = det_ind

        self.Q_xy_scaling = Q_xy_scaling
        self.Q_a_scaling = Q_a_scaling

        self.kf = KalmanFilterXYWHA(dim_x=10, dim_z=5, max_obs=max_obs)
        self.kf.F = np.array(
            [
                [1, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # cx = cx + vx
                [0, 1, 0, 0, 0, 0, 1, 0, 0, 0],  # cy = cy + vy
                [0, 0, 1, 0, 0, 0, 0, 1, 0, 0],  # w = w + vw
                [0, 0, 0, 1, 0, 0, 0, 0, 1, 0],  # h = h + vh
                [0, 0, 0, 0, 1, 0, 0, 0, 0, 1],  # a = a + va
                [0, 0, 0, 0, 0, 1, 0, 0, 0, 0], 
                [0, 0, 0, 0, 0, 0, 1, 0, 0, 0], 
                [0, 0, 0, 0, 0, 0, 0, 1, 0, 0], 
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 0], 
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
        ] 
    )
        self.kf.H = np.array(
            [
                [1, 0, 0, 0, 0, 0, 0, 0, 0 ,0],  # cx
                [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # cy
                [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # w
                [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # h
                [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # angle
    ]
        )

        self.kf.R[2:, 2:] *= 10.0
        self.kf.P[
            5:, 5:
        ] *= 1000.0  # give high uncertainty to the unobservable initial velocities
        self.kf.P *= 10.0

        self.kf.Q[5:7, 5:7] *= self.Q_xy_scaling
        self.kf.Q[-1, -1] *= self.Q_a_scaling

        self.kf.x[:5] = bbox[:5].reshape((5, 1)) # x, y, w, h, angle   (dont take confidence score)
        self.time_since_update = 0
        self.id = KalmanBoxTrackerOBB.count
        KalmanBoxTrackerOBB.count += 1
        self.max_obs = max_obs
        self.history = deque([], maxlen=self.max_obs)
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        self.conf = bbox[-1]
        self.cls = cls
        """
        NOTE: [-1,-1,-1,-1,-1] is a compromising placeholder for non-observation status, the same for the return of
        function k_previous_obs. It is ugly and I do not like it. But to support generate observation array in a
        fast and unified way, which you would see below k_observations = np.array([k_previous_obs(...]]),
        let's bear it for now.
        """
        self.last_observation = np.array([-1, -1, -1, -1, -1, -1])  #WARNING : -1 is a valid angle value 
        self.observations = dict()
        self.history_observations = deque([], maxlen=self.max_obs)
        self.velocity = None
        self.delta_t = delta_t

    def update(self, bbox, cls, det_ind):
        """
        Updates the state vector with observed bbox.
        """
        self.det_ind = det_ind
        if bbox is not None:
            self.conf = bbox[-1]
            self.cls = cls
            if self.last_observation.sum() >= 0:  # no previous observation
                previous_box = None
                for i in range(self.delta_t):
                    dt = self.delta_t - i
                    if self.age - dt in self.observations:
                        previous_box = self.observations[self.age - dt]
                        break
                if previous_box is None:
                    previous_box = self.last_observation
                """
                  Estimate the track speed direction with observations \Delta t steps away
                """
                self.velocity = speed_direction_obb(previous_box, bbox)

            """
              Insert new observations. This is a ugly way to maintain both self.observations
              and self.history_observations. Bear it for the moment.
            """
            self.last_observation = bbox
            self.observations[self.age] = bbox
            self.history_observations.append(bbox)

            self.time_since_update = 0
            self.hits += 1
            self.hit_streak += 1
            self.kf.update(bbox[:5].reshape((5, 1))) # x, y, w, h, angle as column vector   (dont take confidence score)
        else:
            self.kf.update(bbox)

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        if (self.kf.x[7] + self.kf.x[2]) <= 0: # Negative width
            self.kf.x[7] *= 0.0
        if (self.kf.x[8] + self.kf.x[3]) <= 0: # Negative Height
            self.kf.x[8] *= 0.0
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(self.kf.x[0:5].reshape((1, 5)))
        return self.history[-1]

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return self.kf.x[0:5].reshape((1, 5))


class KalmanBoxTracker(object):
    """
    This class represents the internal state of individual tracked objects observed as bbox.
    """

    count = 0

    def __init__(self, bbox, cls, det_ind, delta_t=3, max_obs=50, Q_xy_scaling = 0.01, Q_s_scaling = 0.0001):
        """
        Initialises a tracker using initial bounding box.

        """
        # define constant velocity model
        self.det_ind = det_ind

        self.Q_xy_scaling = Q_xy_scaling
        self.Q_s_scaling = Q_s_scaling

        self.kf = KalmanFilterXYSR(dim_x=7, dim_z=4, max_obs=max_obs)
        self.kf.F = np.array(
            [
                [1, 0, 0, 0, 1, 0, 0],
                [0, 1, 0, 0, 0, 1, 0],
                [0, 0, 1, 0, 0, 0, 1],
                [0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 1],
            ]
        )
        self.kf.H = np.array(
            [
                [1, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0],
            ]
        )

        self.kf.R[2:, 2:] *= 10.0
        self.kf.P[
            4:, 4:
        ] *= 1000.0  # give high uncertainty to the unobservable initial velocities
        self.kf.P *= 10.0

        self.kf.Q[4:6, 4:6] *= self.Q_xy_scaling
        self.kf.Q[-1, -1] *= self.Q_s_scaling

        self.kf.x[:4] = xyxy2xysr(bbox)
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.max_obs = max_obs
        self.history = deque([], maxlen=self.max_obs)
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        self.conf = bbox[-1]
        self.cls = cls
        """
        NOTE: [-1,-1,-1,-1,-1] is a compromising placeholder for non-observation status, the same for the return of
        function k_previous_obs. It is ugly and I do not like it. But to support generate observation array in a
        fast and unified way, which you would see below k_observations = np.array([k_previous_obs(...]]),
        let's bear it for now.
        """
        self.last_observation = np.array([-1, -1, -1, -1, -1])  # placeholder
        self.observations = dict()
        self.history_observations = deque([], maxlen=self.max_obs)
        self.velocity = None
        self.delta_t = delta_t

    def update(self, bbox, cls, det_ind):
        """
        Updates the state vector with observed bbox.
        """
        self.det_ind = det_ind
        if bbox is not None:
            self.conf = bbox[-1]
            self.cls = cls
            if self.last_observation.sum() >= 0:  # no previous observation
                previous_box = None
                for i in range(self.delta_t):
                    dt = self.delta_t - i
                    if self.age - dt in self.observations:
                        previous_box = self.observations[self.age - dt]
                        break
                if previous_box is None:
                    previous_box = self.last_observation
                """
                  Estimate the track speed direction with observations \Delta t steps away
                """
                self.velocity = speed_direction(previous_box, bbox)

            """
              Insert new observations. This is a ugly way to maintain both self.observations
              and self.history_observations. Bear it for the moment.
            """
            self.last_observation = bbox
            self.observations[self.age] = bbox
            self.history_observations.append(np.append(bbox, cls))  # add the cls label for swimAD

            self.time_since_update = 0
            self.hits += 1
            self.hit_streak += 1
            self.kf.update(xyxy2xysr(bbox))
        else:
            self.kf.update(bbox)

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0

        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return convert_x_to_bbox(self.kf.x)


class OcSort(ABC):
    """
    OCSort Tracker: A tracking algorithm that utilizes motion-based tracking.

    Args:
        per_class (bool, optional): Whether to perform per-class tracking. If True, tracks are maintained separately for each object class.
        det_thresh (float, optional): Detection confidence threshold. Detections below this threshold are ignored in the first association step.
        max_age (int, optional): Maximum number of frames to keep a track alive without any detections.
        min_hits (int, optional): Minimum number of hits required to confirm a track.
        asso_threshold (float, optional): Threshold for the association step in data association. Controls the maximum distance allowed between tracklets and detections for a match.
        delta_t (int, optional): Time delta for velocity estimation in Kalman Filter.
        asso_func (str, optional): Association function to use for data association. Options include "iou" for IoU-based association.
        inertia (float, optional): Weight for inertia in motion modeling. Higher values make tracks less responsive to changes.
        use_byte (bool, optional): Whether to use BYTE association in the second association step.
        Q_xy_scaling (float, optional): Scaling factor for the process noise covariance in the Kalman Filter for position coordinates.
        Q_s_scaling (float, optional): Scaling factor for the process noise covariance in the Kalman Filter for scale coordinates.
    """
    def __init__(
        self,
        per_class: bool = False,
        min_conf: float = 0.1,
        det_thresh: float = 0.2,
        max_age: int = 30,
        min_hits: int = 3,
        asso_threshold: float = 0.3,
        delta_t: int = 3,
        asso_func: str = "iou",
        inertia: float = 0.2,
        use_byte: bool = False,
        Q_xy_scaling: float = 0.01,
        Q_s_scaling: float = 0.0001,
        iou_threshold: float = 0.3,
        max_obs: int = 50,
        nr_classes: int = 80,
        is_obb: bool = False
    ):
        """
        Sets key parameters for SORT
        """
        self.det_thresh = det_thresh
        self.max_age = max_age
        self.max_obs = max_obs
        self.min_hits = min_hits
        self.per_class = per_class  # Track per class or not
        self.nr_classes = nr_classes
        self.iou_threshold = iou_threshold
        self.last_emb_size = None
        self.asso_func_name = asso_func+"_obb" if is_obb else asso_func
        self.is_obb = is_obb
        self.window_size = 10
        self.frame_count = 0
        self.active_tracks = []  # This might be handled differently in derived classes
        self.per_class_active_tracks = None
        self._first_frame_processed = False  # Flag to track if the first frame has been processed
        self._first_dets_processed = False

        self.cls_map = {
            '0': 'ashore',
            '1': 'above',
            '2': 'under'
        }

        # Initialize per-class active tracks
        if self.per_class:
            self.per_class_active_tracks = {}
            for i in range(self.nr_classes):
                self.per_class_active_tracks[i] = []
        
        if self.max_age >= self.max_obs:
            LOGGER.warning("Max age > max observations, increasing size of max observations...")
            self.max_obs = self.max_age + 5
            print("self.max_obs", self.max_obs)

        self.min_conf = min_conf
        self.asso_threshold = asso_threshold
        self.delta_t = delta_t
        self.inertia = inertia
        self.use_byte = use_byte
        self.Q_xy_scaling = Q_xy_scaling
        self.Q_s_scaling = Q_s_scaling
        KalmanBoxTracker.count = 0

        self.Window_Size = 50


    # @staticmethod
    def setup_decorator(method):
        """
        Decorator to perform setup on the first frame only.
        This ensures that initialization tasks (like setting the association function) only
        happen once, on the first frame, and are skipped on subsequent frames.
        """
        def wrapper(self, *args, **kwargs):
            # If setup hasn't been done yet, perform it
            # Even if dets is empty (e.g., shape (0, 7)), this check will still pass if it's Nx7
            if not self._first_dets_processed:
                dets = args[0]
                if dets is not None:
                    if dets.ndim == 2 and dets.shape[1] == 6:
                        self.is_obb = False
                        self._first_dets_processed = True
                    elif dets.ndim == 2 and dets.shape[1] == 7:
                        self.is_obb = True
                        self._first_dets_processed = True

            if not self._first_frame_processed:
                img = args[1]
                self.h, self.w = img.shape[0:2]
                self.asso_func = AssociationFunction(w=self.w, h=self.h, asso_mode=self.asso_func_name).asso_func

                # Mark that the first frame setup has been done
                self._first_frame_processed = True

            # Call the original method (e.g., update)
            return method(self, *args, **kwargs)
        
        return wrapper

    # @staticmethod
    def per_class_decorator(update_method):
        """
        Decorator for the update method to handle per-class processing.
        """
        def wrapper(self, dets: np.ndarray, img: np.ndarray, embs: np.ndarray = None):
            
            #handle different types of inputs
            if dets is None or len(dets) == 0:
                dets = np.empty((0, 6))
            
            if self.per_class:
                # Initialize an array to store the tracks for each class
                per_class_tracks = []
                
                # same frame count for all classes
                frame_count = self.frame_count

                for cls_id in range(self.nr_classes):
                    # Get detections and embeddings for the current class
                    class_dets, class_embs = self.get_class_dets_n_embs(dets, embs, cls_id)
                    
                    LOGGER.debug(f"Processing class {int(cls_id)}: {class_dets.shape} with embeddings {class_embs.shape if class_embs is not None else None}")

                    # Activate the specific active tracks for this class id
                    self.active_tracks = self.per_class_active_tracks[cls_id]
                    
                    # Reset frame count for every class
                    self.frame_count = frame_count
                    
                    # Update detections using the decorated method
                    tracks = update_method(self, dets=class_dets, img=img, embs=class_embs)

                    # Save the updated active tracks
                    self.per_class_active_tracks[cls_id] = self.active_tracks

                    if tracks.size > 0:
                        per_class_tracks.append(tracks)
                
                # Increase frame count by 1
                self.frame_count = frame_count + 1

                return np.vstack(per_class_tracks) if per_class_tracks else np.empty((0, 8))
            else:
                # Process all detections at once if per_class is False
                return update_method(self, dets=dets, img=img, embs=embs)
        return wrapper

    @setup_decorator
    @per_class_decorator
    def update(self, dets: np.ndarray, img: np.ndarray, embs: np.ndarray = None) -> np.ndarray:
        """
        Params:
          dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
        Requires: this method must be called once for each frame even with empty detections
        (use np.empty((0, 5)) for frames without detections).
        Returns the a similar array, where the last column is the object ID.
        NOTE: The number of objects returned may differ from the number of detections provided.
        """
        self.check_inputs(dets, img)

        self.frame_count += 1
        h, w = img.shape[0:2]

        dets = np.hstack([dets, np.arange(len(dets)).reshape(-1, 1)])
        confs = dets[:, 4+self.is_obb] 

        inds_low = confs > self.min_conf
        inds_high = confs < self.det_thresh
        inds_second = np.logical_and(
            inds_low, inds_high
        )  # self.det_thresh > score > 0.1, for second matching
        dets_second = dets[inds_second]  # detections for second matching
        remain_inds = confs > self.det_thresh
        dets = dets[remain_inds]

        # get predicted locations from existing trackers.
        trks = np.zeros((len(self.active_tracks), 5+self.is_obb))
        to_del = []
        ret = []
        for t, trk in enumerate(trks):
            pos = self.active_tracks[t].predict()[0]
            trk[:] = [pos[i] for i in range(4+self.is_obb)] + [0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.active_tracks.pop(t)

        velocities = np.array(
            [
                trk.velocity if trk.velocity is not None else np.array((0, 0))
                for trk in self.active_tracks
            ]
        )
        last_boxes = np.array([trk.last_observation for trk in self.active_tracks])

        k_observations = np.array(
            [
                k_previous_obs(trk.observations, trk.age, self.delta_t, is_obb=self.is_obb)
                for trk in self.active_tracks
            ]
        )

        """
            First round of association
        """
        matched, unmatched_dets, unmatched_trks = associate(
            dets[:, 0:5+self.is_obb], trks, self.asso_func, self.asso_threshold, velocities, k_observations, self.inertia, w, h
        )
        for m in matched:
            self.active_tracks[m[1]].update(dets[m[0], :-2], dets[m[0], -2], dets[m[0], -1])

        """
            Second round of associaton by OCR
        """
        # BYTE association
        if self.use_byte and len(dets_second) > 0 and unmatched_trks.shape[0] > 0:
            u_trks = trks[unmatched_trks]
            iou_left = self.asso_func(
                dets_second, u_trks
            )  # iou between low score detections and unmatched tracks
            iou_left = np.array(iou_left)
            if iou_left.max() > self.asso_threshold:
                """
                NOTE: by using a lower threshold, e.g., self.asso_threshold - 0.1, you may
                get a higher performance especially on MOT17/MOT20 datasets. But we keep it
                uniform here for simplicity
                """
                matched_indices = linear_assignment(-iou_left)
                to_remove_trk_indices = []
                for m in matched_indices:
                    det_ind, trk_ind = m[0], unmatched_trks[m[1]]
                    if iou_left[m[0], m[1]] < self.asso_threshold:
                        continue
                    self.active_tracks[trk_ind].update(
                        dets_second[det_ind, :-2], dets_second[det_ind, -2], dets_second[det_ind, -1]
                    )
                    to_remove_trk_indices.append(trk_ind)
                unmatched_trks = np.setdiff1d(
                    unmatched_trks, np.array(to_remove_trk_indices)
                )

        if unmatched_dets.shape[0] > 0 and unmatched_trks.shape[0] > 0:
            left_dets = dets[unmatched_dets]
            left_trks = last_boxes[unmatched_trks]
            iou_left = self.asso_func(left_dets, left_trks)
            iou_left = np.array(iou_left)
            if iou_left.max() > self.asso_threshold:
                """
                NOTE: by using a lower threshold, e.g., self.asso_threshold - 0.1, you may
                get a higher performance especially on MOT17/MOT20 datasets. But we keep it
                uniform here for simplicity
                """
                rematched_indices = linear_assignment(-iou_left)
                to_remove_det_indices = []
                to_remove_trk_indices = []
                for m in rematched_indices:
                    det_ind, trk_ind = unmatched_dets[m[0]], unmatched_trks[m[1]]
                    if iou_left[m[0], m[1]] < self.asso_threshold:
                        continue
                    self.active_tracks[trk_ind].update(dets[det_ind, :-2], dets[det_ind, -2], dets[det_ind, -1])
                    to_remove_det_indices.append(det_ind)
                    to_remove_trk_indices.append(trk_ind)
                unmatched_dets = np.setdiff1d(
                    unmatched_dets, np.array(to_remove_det_indices)
                )
                unmatched_trks = np.setdiff1d(
                    unmatched_trks, np.array(to_remove_trk_indices)
                )

        for m in unmatched_trks:
            self.active_tracks[m].update(None, None, None)

        # create and initialise new trackers for unmatched detections
        for i in unmatched_dets:
            if self.is_obb:
                trk = KalmanBoxTrackerOBB(dets[i, :-2], dets[i, -2], dets[i, -1], delta_t=self.delta_t, Q_xy_scaling=self.Q_xy_scaling, Q_a_scaling=self.Q_s_scaling, max_obs=self.max_obs)
            else:
                trk = KalmanBoxTracker(dets[i, :5], dets[i, 5], dets[i, 6], delta_t=self.delta_t, Q_xy_scaling=self.Q_xy_scaling, Q_s_scaling=self.Q_s_scaling, max_obs=self.max_obs)
            self.active_tracks.append(trk)
        i = len(self.active_tracks)
        for trk in reversed(self.active_tracks):
            if trk.last_observation.sum() < 0:
                d = trk.get_state()[0]
            else:
                """
                this is optional to use the recent observation or the kalman filter prediction,
                we didn't notice significant difference here
                """
                d = trk.last_observation[:4+self.is_obb]
            if (trk.time_since_update < 1) and (
                trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits
            ):
                # +1 as MOT benchmark requires positive
                ret.append(
                    np.concatenate((d, [trk.id + 1], [trk.conf], [trk.cls], [trk.det_ind])).reshape(
                        1, -1
                    )
                )
            i -= 1
            # remove dead tracklet
            if trk.time_since_update > self.max_age:
                self.active_tracks.pop(i)
        if len(ret) > 0:
            return np.concatenate(ret)
        return np.array([])

    def get_class_dets_n_embs(self, dets, embs, cls_id):
        # Initialize empty arrays for detections and embeddings
        class_dets = np.empty((0, 6))
        class_embs = np.empty((0, self.last_emb_size)) if self.last_emb_size is not None else None

        # Check if there are detections
        if dets.size > 0:
            class_indices = np.where(dets[:, 5] == cls_id)[0]
            class_dets = dets[class_indices]
            
            if embs is not None:
                # Assert that if embeddings are provided, they have the same number of elements as detections
                assert dets.shape[0] == embs.shape[0], "Detections and embeddings must have the same number of elements when both are provided"
                
                if embs.size > 0:
                    class_embs = embs[class_indices]
                    self.last_emb_size = class_embs.shape[1]  # Update the last known embedding size
                else:
                    class_embs = None
        return class_dets, class_embs

    def check_inputs(self, dets, img):
        assert isinstance(
            dets, np.ndarray
        ), f"Unsupported 'dets' input format '{type(dets)}', valid format is np.ndarray"
        assert isinstance(
            img, np.ndarray
        ), f"Unsupported 'img_numpy' input format '{type(img)}', valid format is np.ndarray"
        assert (
            len(dets.shape) == 2
        ), "Unsupported 'dets' dimensions, valid number of dimensions is two"
        if self.is_obb:
            assert (
                dets.shape[1] == 7
            ), "Unsupported 'dets' 2nd dimension lenght, valid lenghts is 6 (cx,cy,w,h,angle,conf,cls)"
        else :
            assert (
                dets.shape[1] == 6
            ), "Unsupported 'dets' 2nd dimension lenght, valid lenghts is 6 (x1,y1,x2,y2,conf,cls)"


    def id_to_color(self, id: int, saturation: float = 0.75, value: float = 0.95) -> tuple:
        """
        Generates a consistent unique BGR color for a given ID using hashing.

        Parameters:
        - id (int): Unique identifier for which to generate a color.
        - saturation (float): Saturation value for the color in HSV space.
        - value (float): Value (brightness) for the color in HSV space.

        Returns:
        - tuple: A tuple representing the BGR color.
        """

        # Hash the ID to get a consistent unique value
        hash_object = hashlib.sha256(str(id).encode())
        hash_digest = hash_object.hexdigest()
        
        # Convert the first few characters of the hash to an integer
        # and map it to a value between 0 and 1 for the hue
        hue = int(hash_digest[:8], 16) / 0xffffffff
        
        # Convert HSV to RGB
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        
        # Convert RGB from 0-1 range to 0-255 range and format as hexadecimal
        rgb_255 = tuple(int(component * 255) for component in rgb)
        hex_color = '#%02x%02x%02x' % rgb_255
        # Strip the '#' character and convert the string to RGB integers
        rgb = tuple(int(hex_color.strip('#')[i:i+2], 16) for i in (0, 2, 4))
        
        # Convert RGB to BGR for OpenCV
        bgr = rgb[::-1]
        
        return bgr


    def plot_trackers_trajectories(self, img: np.ndarray, box: tuple, conf: float, cls: int, id: int, thickness: int = 2, fontscale: float = 0.5) -> np.ndarray:
        """
        Draws a bounding box with ID, confidence, and class information on an image.

        Parameters:
        - img (np.ndarray): The image array to draw on.
        - box (tuple): The bounding box coordinates as (x1, y1, x2, y2).
        - conf (float): Confidence score of the detection.
        - cls (int): Class ID of the detection.
        - id (int): Unique identifier for the detection.
        - thickness (int): The thickness of the bounding box.
        - fontscale (float): The font scale for the text.

        Returns:
        - np.ndarray: The image array with the bounding box drawn on it.
        """

        if self.is_obb:
            
            angle = box[4] * 180.0 / np.pi  # Convert radians to degrees
            box_poly = ((box[0], box[1]), (box[2], box[3]), angle)
            # print((width, height))
            rotrec = cv.boxPoints(box_poly)
            box_poly = np.int_(rotrec)  # Convert to integer

            # Draw the rectangle on the image
            img = cv.polylines(img, [box_poly], isClosed=True, color=self.id_to_color(id), thickness=thickness)

            
            img = cv.putText(
                img,
                f'{int(id)}, c: {int(cls)}',
                (int(box[0]), int(box[1]) - 10),
                cv.FONT_HERSHEY_SIMPLEX,
                fontscale,
                self.id_to_color(id),
                thickness
            )
        else:
            img = cv.rectangle(
                img,
                (int(box[0]), int(box[1])),
                (int(box[2]), int(box[3])),
                self.id_to_color(id),  
                thickness
            )
            img = cv.putText(
                img,
                f'id: {int(id)}, conf: {conf:.2f}, c: {int(cls)}',
                (int(box[0]), int(box[1]) - 10),
                cv.FONT_HERSHEY_SIMPLEX,
                fontscale,
                self.id_to_color(id),
                thickness
            )
        return img


    def plot_plain_box_on_img(self, img: np.ndarray, box: tuple, conf: float, cls: int, id: int, thickness: int = 2, fontscale: float = 0.5) -> np.ndarray:
        """
        Draws a bounding box with ID, confidence, and class information on an image.

        Parameters:
        - img (np.ndarray): The image array to draw on.
        - box (tuple): The bounding box coordinates as (x1, y1, x2, y2).
        - conf (float): Confidence score of the detection.
        - cls (int): Class ID of the detection.
        - id (int): Unique identifier for the detection.
        - thickness (int): The thickness of the bounding box.
        - fontscale (float): The font scale for the text.

        Returns:
        - np.ndarray: The image array with the bounding box drawn on it.
        """
        
        if self.is_obb:
            
            angle = box[4] * 180.0 / np.pi  # Convert radians to degrees
            box_poly = ((box[0], box[1]), (box[2], box[3]), angle)
            # print((width, height))
            rotrec = cv.boxPoints(box_poly)
            box_poly = np.int_(rotrec)  # Convert to integer

            # Draw the rectangle on the image
            img = cv.polylines(img, [box_poly], isClosed=True, color=self.id_to_color(id), thickness=thickness)

            
            img = cv.putText(
                img,
                f'{int(id)}, {self.cls_map[str(cls)]}',
                (int(box[0]), int(box[1]) - 10),
                cv.FONT_HERSHEY_SIMPLEX,
                fontscale,
                self.id_to_color(id),
                thickness
            )
        else:
            img = cv.rectangle(
                img,
                (int(box[0]), int(box[1])),
                (int(box[2]), int(box[3])),
                self.id_to_color(id),  
                thickness
            )
            txt = f'{int(id)}, {self.cls_map[str(int(cls))]}'
            text_size, _ = cv.getTextSize(txt, cv.FONT_HERSHEY_SIMPLEX, fontscale, thickness)
            cv.rectangle(
                img,
                (int(box[0]), int(box[1]) - 10 - text_size[1]),
                (int(box[0])+text_size[0], int(box[1]) + 10),
                (255,255,255),
                -1
            )
            img = cv.putText(
                img,
                txt,
                (int(box[0]), int(box[1]) - 10),
                cv.FONT_HERSHEY_SIMPLEX,
                fontscale,
                self.id_to_color(id),
                thickness
            )
        return img

    def plot_box_on_img(self, img: np.ndarray, box: tuple, conf: float, cls: int, id: int, thickness: int = 2, fontscale: float = 0.5) -> np.ndarray:
        """
        Draws a bounding box with ID, confidence, and class information on an image.

        Parameters:
        - img (np.ndarray): The image array to draw on.
        - box (tuple): The bounding box coordinates as (x1, y1, x2, y2).
        - conf (float): Confidence score of the detection.
        - cls (int): Class ID of the detection.
        - id (int): Unique identifier for the detection.
        - thickness (int): The thickness of the bounding box.
        - fontscale (float): The font scale for the text.

        Returns:
        - np.ndarray: The image array with the bounding box drawn on it.
        """
        if self.is_obb:
            
            angle = box[4] * 180.0 / np.pi  # Convert radians to degrees
            box_poly = ((box[0], box[1]), (box[2], box[3]), angle)
            # print((width, height))
            rotrec = cv.boxPoints(box_poly)
            box_poly = np.int_(rotrec)  # Convert to integer

            # Draw the rectangle on the image
            img = cv.polylines(img, [box_poly], isClosed=True, color=self.id_to_color(id), thickness=thickness)

            
            img = cv.putText(
                img,
                f'id: {int(id)}, conf: {conf:.2f}, c: {int(cls)}',
                (int(box[0]), int(box[1]) - 10),
                cv.FONT_HERSHEY_SIMPLEX,
                fontscale,
                self.id_to_color(id),
                thickness
            )
        else:
            img = cv.rectangle(
                img,
                (int(box[0]), int(box[1])),
                (int(box[2]), int(box[3])),
                self.id_to_color(id),  
                thickness
            )
            img = cv.putText(
                img,
                f'id: {int(id)}, conf: {conf:.2f}, c: {int(cls)}',
                (int(box[0]), int(box[1]) - 10),
                cv.FONT_HERSHEY_SIMPLEX,
                fontscale,
                self.id_to_color(id),
                thickness
            )
        return img


    def plot_box_on_img_with_rule(self, img: np.ndarray, box: tuple, conf: float, cls: int, id: int, thickness: int = 2, fontscale: float = 0.5, rule=None) -> np.ndarray:
        """
        Draws a bounding box with ID, confidence, and class information on an image.

        Parameters:
        - img (np.ndarray): The image array to draw on.
        - box (tuple): The bounding box coordinates as (x1, y1, x2, y2).
        - conf (float): Confidence score of the detection.
        - cls (int): Class ID of the detection.
        - id (int): Unique identifier for the detection.
        - thickness (int): The thickness of the bounding box.
        - fontscale (float): The font scale for the text.

        Returns:
        - np.ndarray: The image array with the bounding box drawn on it.
        """
        if self.is_obb:
            
            angle = box[4] * 180.0 / np.pi  # Convert radians to degrees
            box_poly = ((box[0], box[1]), (box[2], box[3]), angle)
            # print((width, height))
            rotrec = cv.boxPoints(box_poly)
            box_poly = np.int_(rotrec)  # Convert to integer

            # Draw the rectangle on the image
            img = cv.polylines(img, [box_poly], isClosed=True, color=(0,0,255), thickness=thickness)

            img = cv.putText(
                img,
                f'{int(id)}, {self.cls_map[str(cls)]}, True',
                (int(box[0]), int(box[1]) - 10),
                cv.FONT_HERSHEY_SIMPLEX,
                fontscale,
                (0,0,255),   # red for AD objects
                thickness
            )
        else:
            img = cv.rectangle(
                img,
                (int(box[0]), int(box[1])),
                (int(box[2]), int(box[3])),
                (0,0,255),   # red for AD objects
                thickness
            )
            img = cv.putText(
                img,
                f'{int(id)}, {self.cls_map[str(int(cls))]}, True',
                (int(box[0]), int(box[1]) - 10),
                cv.FONT_HERSHEY_SIMPLEX,
                fontscale,
                (0,0,255),
                thickness
            )
        return img


    def plot_trackers_trajectories(self, img: np.ndarray, observations: list, id: int) -> np.ndarray:
        """
        Draws the trajectories of tracked objects based on historical observations. Each point
        in the trajectory is represented by a circle, with the thickness increasing for more
        recent observations to visualize the path of movement.

        Parameters:
        - img (np.ndarray): The image array on which to draw the trajectories.
        - observations (list): A list of bounding box coordinates representing the historical
        observations of a tracked object. Each observation is in the format (x1, y1, x2, y2).
        - id (int): The unique identifier of the tracked object for color consistency in visualization.

        Returns:
        - np.ndarray: The image array with the trajectories drawn on it.
        """
        for i, box in enumerate(observations):
            trajectory_thickness = int(np.sqrt(float (i + 1)) * 1.2)
            if self.is_obb:
                img = cv.circle(
                    img,
                    (int(box[0]), int(box[1])),
                    2,
                    self.id_to_color(id),   # red for AD objects
                    thickness=trajectory_thickness 
                )
            else:

                img = cv.circle(
                    img,
                    (int((box[0] + box[2]) / 2),
                    int((box[1] + box[3]) / 2)), 
                    2,
                    self.id_to_color(id),   # red for AD objects
                    thickness=trajectory_thickness
                )
        return img


    def plot_plain_results(self, img: np.ndarray, show_trajectories: bool, thickness: int = 2, fontscale: float = 0.5) -> np.ndarray:
        """
        Visualizes the trajectories of all active tracks on the image. For each track,
        it draws the latest bounding box and the path of movement if the history of
        observations is longer than two. This helps in understanding the movement patterns
        of each tracked object.

        Parameters:
        - img (np.ndarray): The image array on which to draw the trajectories and bounding boxes.
        - show_trajectories (bool): Whether to show the trajectories.
        - thickness (int): The thickness of the bounding box.
        - fontscale (float): The font scale for the text.

        Returns:
        - np.ndarray: The image array with trajectories and bounding boxes of all active tracks.
        """

        # if values in dict
        if self.per_class_active_tracks is not None:
            for k in self.per_class_active_tracks.keys():
                active_tracks = self.per_class_active_tracks[k]
                for a in active_tracks:
                    if int(a.cls) == 0: continue    # ignore the ashore person
                    if a.history_observations:
                        if len(a.history_observations) > 2:
                            box = a.history_observations[-1]
                            img = self.plot_plain_box_on_img(img, box, a.conf, a.cls, a.id, thickness, fontscale)
                            if show_trajectories:
                                img = self.plot_trackers_trajectories(img, a.history_observations, a.id)
        else:
            for a in self.active_tracks:
                if int(a.cls) == 0: continue    # ignore the ashore person
                if a.history_observations:
                    if len(a.history_observations) > 2:
                        box = a.history_observations[-1]
                        img = self.plot_plain_box_on_img(img, box, a.conf, a.cls, a.id, thickness, fontscale)
                        if show_trajectories:
                            img = self.plot_trackers_trajectories(img, a.history_observations, a.id)
                
        return img

    def plot_multi_view_results(self, img: np.ndarray, show_trajectories: bool, thickness: int = 2, fontscale: float = 0.5) -> np.ndarray:
        """
        Visualizes the trajectories of all active tracks on the image. For each track,
        it draws the latest bounding box and the path of movement if the history of
        observations is longer than two. This helps in understanding the movement patterns
        of each tracked object.

        Parameters:
        - img (np.ndarray): The image array on which to draw the trajectories and bounding boxes.
        - show_trajectories (bool): Whether to show the trajectories.
        - thickness (int): The thickness of the bounding box.
        - fontscale (float): The font scale for the text.

        Returns:
        - np.ndarray: The image array with trajectories and bounding boxes of all active tracks.
        """

        # if values in dict
        if self.per_class_active_tracks is not None:
            for k in self.per_class_active_tracks.keys():
                active_tracks = self.per_class_active_tracks[k]
                for a in active_tracks:
                    if a.history_observations:
                        if len(a.history_observations) > 2:
                            box = a.history_observations[-1]
                            img = self.plot_plain_box_on_img(img, box, a.conf, a.cls, a.id, thickness, fontscale)
                            if show_trajectories:
                                img = self.plot_trackers_trajectories(img, a.history_observations, a.id)
        else:
            for a in self.active_tracks:
                if a.history_observations:
                    if len(a.history_observations) > 2:
                        box = a.history_observations[-1]
                        img = self.plot_plain_box_on_img(img, box, a.conf, a.cls, a.id, thickness, fontscale)
                        if show_trajectories:
                            img = self.plot_trackers_trajectories(img, a.history_observations, a.id)
                
        return img

    def plot_results(self, img: np.ndarray, show_trajectories: bool, thickness: int = 2, fontscale: float = 0.5) -> np.ndarray:
        """
        Visualizes the trajectories of all active tracks on the image. For each track,
        it draws the latest bounding box and the path of movement if the history of
        observations is longer than two. This helps in understanding the movement patterns
        of each tracked object.

        Parameters:
        - img (np.ndarray): The image array on which to draw the trajectories and bounding boxes.
        - show_trajectories (bool): Whether to show the trajectories.
        - thickness (int): The thickness of the bounding box.
        - fontscale (float): The font scale for the text.

        Returns:
        - np.ndarray: The image array with trajectories and bounding boxes of all active tracks.
        """

        # if values in dict
        if self.per_class_active_tracks is not None:
            for k in self.per_class_active_tracks.keys():
                active_tracks = self.per_class_active_tracks[k]
                for a in active_tracks:
                    if a.history_observations:
                        if len(a.history_observations) > 2:
                            box = a.history_observations[-1]
                            img = self.plot_box_on_img(img, box, a.conf, a.cls, a.id, thickness, fontscale)
                            if show_trajectories:
                                img = self.plot_trackers_trajectories(img, a.history_observations, a.id)
        else:
            for a in self.active_tracks:
                if a.history_observations:
                    if len(a.history_observations) > 2:
                        box = a.history_observations[-1]
                        img = self.plot_box_on_img(img, box, a.conf, a.cls, a.id, thickness, fontscale)
                        if show_trajectories:
                            img = self.plot_trackers_trajectories(img, a.history_observations, a.id)
                
        return img


    def detect_AD(self) -> list[np.ndarray]:
        """
        detect the AD swimmer with rules, all the rules function is named as rule+num num~[1,99]        
        """

        AD_list = []
        info_list = []
        for a in self.active_tracks:
            match_rules = []
            if a.history_observations:
                for i in range(1, 99):
                    rule_name = f"rule{i}"
                    if hasattr(self, rule_name):
                        func = getattr(self, rule_name)
                        flag, info = func(a.history_observations, a.id)
                        info_list.append({f"rule{i}":info})
                        if flag:
                            match_rules.append(rule_name)
                
                if len(match_rules) > 0:
                    rules = " ".join(match_rules)
                    AD_list.append([a, rules])

        return AD_list, info_list


    def plot_AD_results(self, img: np.ndarray, show_trajectories: bool, AD_list: list, thickness: int = 4, fontscale: float = 0.5) -> np.ndarray:
        """
        Visualizes the trajectories of all active tracks on the image. For each track,
        it draws the latest bounding box and the path of movement if the history of
        observations is longer than two. This helps in understanding the movement patterns
        of each tracked object.

        Parameters:
        - img (np.ndarray): The image array on which to draw the trajectories and bounding boxes.
        - show_trajectories (bool): Whether to show the trajectories.
        - thickness (int): The thickness of the bounding box.
        - fontscale (float): The font scale for the text.

        Returns:
        - np.ndarray: The image array with trajectories and bounding boxes of all active tracks.
        """
        for a in AD_list:
            rule = a[-1]
            a = a[0]
            if a.history_observations:
                if len(a.history_observations) > 2:
                    box = a.history_observations[-1]
                    img = self.plot_box_on_img_with_rule(img, box, a.conf, a.cls, a.id, thickness, fontscale, rule=rule)
                    if show_trajectories:
                        img = self.plot_trackers_trajectories(img, a.history_observations, a.id)
                
        return img

    def detect_AD_v2(self, object_map, metrics=[]) -> list[np.ndarray]:
        """
        detect the AD swimmer with rules, all the rules function is named as rule+num num~[1,99]        
        """
        
        results = dict()
        for a in self.active_tracks:
            match_rules = []
            if a.history_observations:
                box = a.history_observations[-1][:4]
                box_xyxy = f'{int(box[0]):d}_{int(box[1]):d}_{int(box[2]):d}_{int(box[3]):d}'
                assert box_xyxy in object_map
                result = {'view': object_map[box_xyxy][0], 
                        'bbox_left_top': [object_map[box_xyxy][1][0], object_map[box_xyxy][1][1]], 
                        'bbox_right_bottom': [object_map[box_xyxy][1][2], object_map[box_xyxy][1][3]]}
                for metric in metrics:
                    assert hasattr(self, metric), f"The {metric} is not implemented."
                    func = getattr(self, metric)
                    value = func(a.history_observations, a.id)
                    result[f'{metric}'] = value
                results[f'obj{a.id}'] = result

        return results


    def min_dist(self, history_observations, track_id):
        """window_size下的各点间最小欧氏距离"""
        if len(history_observations) < 2:
            return 0
        else:
            obs = list(history_observations)[-self.Window_Size:]
            data = np.stack(obs)
            midpoints = np.mean(data[:,:4].reshape(data.shape[0], 2, 2), axis=1)
            distances = np.sqrt(np.sum((midpoints[:, np.newaxis] - midpoints)**2, axis=2))
            np.fill_diagonal(distances,np.inf)# 把对角线（自身距离）变成 +inf
            return float(np.min(distances))# 取非自身的最小值
    
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



def create_tracker(
        tracker_type, tracker_config=None, reid_weights=None, device=None, half=None, per_class=None,
        evolve_param_dict=None,
):
    """
    Creates and returns an instance of the specified tracker type.
    
    Parameters:
    - tracker_type: The type of the tracker (e.g., 'strongsort', 'ocsort').
    - tracker_config: Path to the tracker configuration file.
    - reid_weights: Weights for ReID (re-identification).
    - device: Device to run the tracker on (e.g., 'cpu', 'cuda').
    - half: Boolean indicating whether to use half-precision.
    - per_class: Boolean for class-specific tracking (optional).
    - evolve_param_dict: A dictionary of parameters for evolving the tracker.
    
    Returns:
    - An instance of the selected tracker.
    """

    # Load configuration from file or use provided dictionary
    if evolve_param_dict is None:
        with open(tracker_config, "r") as f:
            yaml_config = yaml.load(f, Loader=yaml.FullLoader)
            tracker_args = {param: details['default'] for param, details in yaml_config.items()}
    else:
        tracker_args = evolve_param_dict

    tracker_args['per_class'] = per_class
    return OcSort(**tracker_args)