import numpy as np
import pandas as pd
import math
import logging
from enum import IntEnum
from dataclasses import dataclass
from itertools import groupby
from typing import List, Dict, Optional, Callable

from tracking.track_window_manager import TrackObservation


# ====== 类别 ID（例如水上/水下）======
class ClassID(IntEnum):
    HEAD_ABOVE = 1  # 水上头部（cls_id = 1）
    HEAD_BELOW = 2  # 水下头部（cls_id = 2）


# ====== 规则编号 ======
class RuleID(IntEnum):
    HEAD_UNDERWATER_RATIO = 1
    LOW_VELOCITY_AND_ACC = 2
    JITTER_ANGLE = 3
    LONG_UNDERWATER_STREAK = 4
    DROP_EVENT = 5
    LOW_POSITION_VARIANCE = 6
    REPEATED_LOCATIONS = 7
    FREQUENT_DIR_CHANGE = 8
    PATH_RATIO = 9


# ====== 规则描述映射 ======
RULE_DESC_MAP = {
    RuleID.HEAD_UNDERWATER_RATIO: "Head underwater ratio >= 70%",
    RuleID.LOW_VELOCITY_AND_ACC: "Velocity and acceleration both low",
    RuleID.JITTER_ANGLE: "Jitter angle > 0.4 (irregular movement)",
    RuleID.LONG_UNDERWATER_STREAK: "Underwater streak > 1s",
    RuleID.DROP_EVENT: "Drop event with sudden underwater persistence",
    RuleID.LOW_POSITION_VARIANCE: "Position variance low (stuck in place)",
    RuleID.REPEATED_LOCATIONS: "Revisited same location >=3 times",
    RuleID.FREQUENT_DIR_CHANGE: "Frequent direction change (angle > 1 rad)",
    RuleID.PATH_RATIO: "Path/Displacement ratio > 2.5 (zigzag motion)"
}


# ====== 每条规则的结构化结果 ======
@dataclass
class RuleResult:
    rule_id: RuleID  # 枚举标识
    rule_name: str  # 人类可读描述
    value: object  # 规则值（可为 float、tuple）
    triggered: bool  # 是否触发


# ====== 溺水规则检测器核心类 ======
class DrowningDetector:
    """
    基于规则的溺水检测器，适用于滑窗轨迹检测系统（与 TrackWindowManager 联动）

    支持特性：
    - 自动分析轨迹轨道是否存在异常
    - 每条规则编号、描述、结构化结果
    - 日志可调试
    """

    def __init__(self, fps: int = 20, use_original_bbox: bool = False):
        """
        :param fps: 帧率（用于速度/时间计算）
        :param use_original_bbox: 是否使用原始图像坐标（True）或 projected bbox（False）
        """
        self.fps = fps
        self.use_original_bbox = use_original_bbox
        self.df: Optional[pd.DataFrame] = None

        self.logger = logging.getLogger("DrowningDetector")
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

    def _estimate_fps(self, observations: List[TrackObservation]) -> float:
        if len(observations) < 2:
            return self.fps  # 回退到默认
        duration = observations[-1].timestamp - observations[0].timestamp
        return (len(observations) - 1) / duration if duration > 1e-3 else self.fps

    def as_hook_fn(self) -> Callable[[str, List[TrackObservation]], Optional[Dict]]:
        def hook_fn(track_id: str, observations: List[TrackObservation]) -> Optional[Dict]:
            result = self.evaluate(observations)

            # 可选：添加漂亮日志输出，仅当真的异常
            if result is not None:
                rule_names = [r['rule_name'] for r in result.get('triggered_rules', [])]
                self.logger.info(f"[DrowningDetector] track_id={track_id} 异常触发: {rule_names}")

            return result

        return hook_fn

    def evaluate(self, observations: List[TrackObservation]) -> Optional[Dict]:
        """
        Returns
        -------
        return {
            "alarm": True,
            "triggered_rules": [
                {
                    "rule_id": 5,
                    "rule_name": "Drop event with sudden underwater persistence",
                    "value": (0.12, 0.91, 1.23),
                    "triggered": True
                },
                ...
            ]
        }
        """
        if len(observations) < max(5, self.fps):
            self.logger.warning(f"轨迹帧数不足（仅 {len(observations)} 帧），期望至少 {self.fps} 帧（约1秒）用于评估，跳过。")
            return None

        duration = observations[-1].timestamp - observations[0].timestamp
        if duration < 2.0:
            self.logger.warning(f"轨迹时长仅 {duration:.2f}s，不足 2 秒，跳过评估。")
            return None

        est_fps = self._estimate_fps(observations)
        self.logger.debug(f"[FPS Estimate] 使用动态 FPS={est_fps:.2f} 替代设定值 {self.fps}")

        self._convert_to_dataframe(observations)
        if self.df is None:
            self.logger.warning("数据转换失败。")
            return None

        # 执行所有规则
        results = [
            self.rule_1_head_ratio(),
            self.rule_2_velocity_acc(),
            self.rule_3_jitter(),
            self.rule_4_underwater_streak(),
            self.rule_5_drop_event(),
            self.rule_6_low_variance(),
            self.rule_7_repeat_location(),
            self.rule_8_direction_change(),
            self.rule_9_path_ratio(),
        ]

        triggered = [r for r in results if r.triggered]

        # 判断逻辑：规则 ≥ 4 或 DropEvent 单独触发
        alarm = (
                len(triggered) >= 4 or
                any(r.rule_id == RuleID.DROP_EVENT for r in triggered)
        )

        return {
            "alarm": alarm,
            "triggered_rules": [r.__dict__ for r in triggered]
        } if alarm else None

    def _convert_to_dataframe(self, observations: List[TrackObservation]):
        """
        将滑窗中的轨迹观测转换为 pandas.DataFrame 结构，用于规则计算
        """
        data = []
        for obs in observations:
            entry = obs.entry
            x1, y1, x2, y2 = entry.original_bbox if self.use_original_bbox else (entry.x1, entry.y1, entry.x2, entry.y2)
            x = (x1 + x2) / 2
            y = (y1 + y2) / 2
            head_above = 1 if entry.cls_id == ClassID.HEAD_ABOVE else 0
            data.append({"x": x, "y": y, "head_above": head_above})

        self.df = pd.DataFrame(data) if data else None

    # ====== 规则定义 ======

    def rule_1_head_ratio(self) -> RuleResult:
        ratio = (self.df['head_above'] == 0).mean()
        self.logger.debug(f"[Rule 1] head_underwater_ratio = {ratio:.3f}")
        return RuleResult(RuleID.HEAD_UNDERWATER_RATIO, RULE_DESC_MAP[RuleID.HEAD_UNDERWATER_RATIO], ratio,
                          ratio >= 0.7)

    def rule_2_velocity_acc(self) -> RuleResult:
        pos = self.df[['x', 'y']].values
        v = np.diff(pos, axis=0) * self.fps
        v_mag = np.linalg.norm(v, axis=1)
        a = np.diff(v_mag) * self.fps
        mean_v, mean_a = v_mag.mean(), np.mean(np.abs(a))
        self.logger.debug(f"[Rule 2] velocity = {mean_v:.3f}, acc = {mean_a:.3f}")
        return RuleResult(RuleID.LOW_VELOCITY_AND_ACC, RULE_DESC_MAP[RuleID.LOW_VELOCITY_AND_ACC],
                          (mean_v, mean_a), mean_v < 0.2 and mean_a < 0.1)

    def rule_3_jitter(self) -> RuleResult:
        pos = self.df[['x', 'y']].values
        angles = [math.atan2(pos[i][1] - pos[i - 1][1], pos[i][0] - pos[i - 1][0]) for i in range(1, len(pos))]
        jitter = np.mean([abs(angles[i] - angles[i - 1]) for i in range(1, len(angles))]) if len(angles) >= 2 else 0
        self.logger.debug(f"[Rule 3] jitter = {jitter:.3f}")
        return RuleResult(RuleID.JITTER_ANGLE, RULE_DESC_MAP[RuleID.JITTER_ANGLE], jitter, jitter > 0.4)

    def rule_4_underwater_streak(self) -> RuleResult:
        head = (self.df['head_above'] == 0).values
        max_len = curr = 0
        for h in head:
            curr = curr + 1 if h else 0
            max_len = max(max_len, curr)
        duration = max_len / self.fps
        self.logger.debug(f"[Rule 4] max_underwater_duration = {duration:.2f}s")
        return RuleResult(RuleID.LONG_UNDERWATER_STREAK, RULE_DESC_MAP[RuleID.LONG_UNDERWATER_STREAK],
                          duration, duration >= 1.0)

    def rule_5_drop_event(self) -> RuleResult:
        h = self.df['head_above'].values
        half = len(h) // 2
        f, b = (h[:half] == 0).mean(), (h[half:] == 0).mean()
        max_len = max([sum(1 for _ in g) for k, g in groupby(h) if k == 0] or [0])
        duration = max_len / self.fps
        self.logger.debug(f"[Rule 5] drop_event: f={f:.2f}, b={b:.2f}, max_dur={duration:.2f}s")
        triggered = f < 0.2 and b >= 0.8 and duration >= 1.0
        return RuleResult(RuleID.DROP_EVENT, RULE_DESC_MAP[RuleID.DROP_EVENT], (f, b, duration), triggered)

    def rule_6_low_variance(self) -> RuleResult:
        pos = self.df[['x', 'y']].values
        centroid = np.mean(pos, axis=0)
        drift = np.mean(np.linalg.norm(pos - centroid, axis=1) ** 2)
        self.logger.debug(f"[Rule 6] variance = {drift:.4f}")
        return RuleResult(RuleID.LOW_POSITION_VARIANCE, RULE_DESC_MAP[RuleID.LOW_POSITION_VARIANCE], drift,
                          drift <= 0.1)

    def rule_7_repeat_location(self) -> RuleResult:
        pos = self.df[['x', 'y']].values
        visited = {}
        for x, y in pos:
            key = (int(x // 0.1), int(y // 0.1))
            visited[key] = visited.get(key, 0) + 1
        repeat = sum(1 for v in visited.values() if v >= 2)
        self.logger.debug(f"[Rule 7] repeat_location = {repeat}")
        return RuleResult(RuleID.REPEATED_LOCATIONS, RULE_DESC_MAP[RuleID.REPEATED_LOCATIONS], repeat, repeat >= 3)

    def rule_8_direction_change(self) -> RuleResult:
        pos = self.df[['x', 'y']].values
        delta = pos[-1] - pos[0]
        if np.linalg.norm(delta) < 1e-5:
            return RuleResult(RuleID.FREQUENT_DIR_CHANGE, RULE_DESC_MAP[RuleID.FREQUENT_DIR_CHANGE], np.pi, True)
        angles = []
        for i in range(1, len(pos)):
            vi = pos[i] - pos[i - 1]
            if np.linalg.norm(vi) < 1e-5:
                continue
            cos_theta = np.dot(vi, delta) / (np.linalg.norm(vi) * np.linalg.norm(delta))
            angles.append(np.arccos(np.clip(cos_theta, -1, 1)))
        avg_angle = np.mean(angles) if angles else 0
        self.logger.debug(f"[Rule 8] avg_dir_change = {avg_angle:.3f}")
        return RuleResult(RuleID.FREQUENT_DIR_CHANGE, RULE_DESC_MAP[RuleID.FREQUENT_DIR_CHANGE], avg_angle,
                          avg_angle > 1.0)

    def rule_9_path_ratio(self) -> RuleResult:
        pos = self.df[['x', 'y']].values
        path_len = np.sum(np.linalg.norm(np.diff(pos, axis=0), axis=1))
        disp = np.linalg.norm(pos[-1] - pos[0])
        ratio = path_len / disp if disp > 1e-5 else float('inf')
        self.logger.debug(f"[Rule 9] path_ratio = {ratio:.3f}")
        return RuleResult(RuleID.PATH_RATIO, RULE_DESC_MAP[RuleID.PATH_RATIO], ratio, ratio > 2.5)
