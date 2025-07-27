import time
import json
import threading
import weakref
from queue import Queue
from collections import deque
from typing import Dict, Deque, List, Optional, Callable
from pathlib import Path

from tracking.detection_entry import DetectionEntry


class TrackObservation:
    """
    单帧观测数据，封装了 DetectionEntry + 时间戳 + 帧号
    """

    def __init__(self, entry: DetectionEntry, frame_index: Optional[int] = None):
        self.entry = entry
        self.timestamp = time.time()
        self.frame_index = frame_index

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "frame_index": self.frame_index,
            **self.entry.as_dict()
        }


class TrackWindow:
    """
    每个 track_id 的滑动窗口轨迹结构
    """

    def __init__(self, track_id: str, max_length: int):
        self.track_id = track_id
        self.history: Deque[TrackObservation] = deque(maxlen=max_length)
        self.last_updated: float = time.time()

    def add(self, obs: TrackObservation, frame_window: Optional[int] = None):
        self.history.append(obs)
        self.last_updated = time.time()
        if frame_window is not None:
            self._filter_by_frame_window(frame_window)

    def to_serializable(self) -> dict:
        return {
            "track_id": self.track_id,
            "last_updated": self.last_updated,
            "history": [obs.to_dict() for obs in self.history]
        }

    def _filter_by_frame_window(self, window_span: int):
        """
        滑动窗口裁剪：只保留 `frame_index >= (max_frame_index - window_span)`
        """
        if not self.history:
            return
        max_frame_index = max(obs.frame_index for obs in self.history if obs.frame_index is not None)
        threshold = max_frame_index - window_span
        # 裁剪过旧的帧
        while self.history and self.history[0].frame_index is not None and self.history[0].frame_index < threshold:
            self.history.popleft()


class DualChannelLogger(threading.Thread):
    """
    异步双通道日志记录器（控制台 + 文件）
    """

    def __init__(self, save_dir: Path):
        super().__init__(daemon=True)
        self.log_queue: Queue = Queue()
        self.running = True
        self.save_dir = save_dir
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.save_dir / "track_window_log.jsonl"

    def run(self):
        with open(self.log_file, "a", encoding="utf-8") as f:
            while self.running:
                try:
                    log_item = self.log_queue.get(timeout=1)
                    json.dump(log_item, f, ensure_ascii=False)
                    f.write("\n")
                    f.flush()
                    self._print_to_console(log_item)
                except Exception:
                    continue

    def push(self, item: dict):
        self.log_queue.put(item)

    def _print_to_console(self, item: dict):
        event = item.get("event", "unknown")
        tid = item.get("track_id", "?")
        frame = item.get("frame_index", "-")
        if event == "update":
            print(f"[Track] [frame={frame}] track_id={tid} updated.")
        elif event == "expired":
            print(f"[Expire] track_id={tid} removed due to inactivity.")
        elif event == "snapshot":
            print(f"[Snapshot] saved to {item.get('filename')}")
        else:
            print(f"[Log] {item}")

    def stop(self):
        self.running = False


class SnapshotScheduler(threading.Thread):
    """
    使用 weakref 的定时快照线程，避免对 TrackWindowManager 的强引用闭环
    """

    def __init__(self, manager: 'TrackWindowManager', interval: int = 60):
        super().__init__(daemon=True)
        self.manager_ref = weakref.ref(manager)  # 弱引用
        self.interval = interval
        self.running = True

    def run(self):
        while self.running:
            time.sleep(self.interval)
            manager = self.manager_ref()
            if manager is None:
                print("[SnapshotScheduler] Manager reference lost. Stopping scheduler.")
                break
            try:
                filename = manager.save_snapshot()
                manager.logger.push({
                    "event": "snapshot",
                    "filename": str(filename),
                    "time": time.time()
                })
            except Exception as e:
                print(f"[SnapshotScheduler] Save failed: {e}")

    def stop(self):
        self.running = False


class TrackWindowManager:
    """
    全局轨迹窗口管理器：追踪 track_id -> 轨迹滑动窗口，含 TTL 管理与异步日志
    """

    def __init__(self, max_window_size: int = 30, ttl: int = 10, save_dir: str = "track_logs",
                 snapshot_interval: int = 60, frame_window_span: Optional[int] = None):
        self.windows: Dict[str, TrackWindow] = {}
        self.max_window_size = max_window_size
        self.ttl = ttl  # 存活时长，秒
        self.save_dir = Path(save_dir)
        self.logger = DualChannelLogger(self.save_dir)
        self.logger.start()

        self.snapshot_scheduler = SnapshotScheduler(self, interval=snapshot_interval)
        self.snapshot_scheduler.start()
        self.frame_window_span = frame_window_span  # e.g. 30 帧以内

    def update(self, track_id: str, entry: DetectionEntry, frame_index: Optional[int] = None):
        """
        更新某个轨迹窗口；若不存在自动创建；自动处理清理逻辑
        """
        self._cleanup_lazy()

        obs = TrackObservation(entry, frame_index)
        if track_id not in self.windows:
            self.windows[track_id] = TrackWindow(track_id, self.max_window_size)
        self.windows[track_id].add(obs, frame_window=self.frame_window_span)

        self.logger.push({
            "event": "update",
            "track_id": track_id,
            "time": obs.timestamp,
            "frame_index": frame_index,
            "data": entry.as_dict()
        })

    def _cleanup_lazy(self):
        """
        懒清理：每次 update 时检查所有窗口是否超时
        """
        now = time.time()
        expired_ids = [tid for tid, w in self.windows.items() if now - w.last_updated > self.ttl]
        for tid in expired_ids:
            del self.windows[tid]
            self.logger.push({
                "event": "expired",
                "track_id": tid,
                "time": now
            })

    def get_history(self, track_id: str) -> List[TrackObservation]:
        return list(self.windows.get(track_id, TrackWindow(track_id, self.max_window_size)).history)

    def get_all_track_ids(self) -> List[str]:
        return list(self.windows.keys())

    def save_snapshot(self, filename: Optional[str] = None) -> Path:
        """
        保存所有轨迹窗口的快照为 JSON 文件
        """
        now = int(time.time())
        filename = filename or f"track_snapshot_{now}.json"
        path = self.save_dir / filename
        data = {tid: w.to_serializable() for tid, w in self.windows.items()}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        return path

    def shutdown(self, timeout: int = 5):
        """
        停止后台所有线程，确保日志与快照均已完成，安全退出。
        """
        print("[TrackWindowManager] Initiating graceful shutdown...")

        # 1. 停止异步日志线程
        if hasattr(self, "logger") and self.logger is not None:
            self.logger.stop()
            self.logger.join(timeout=timeout)
            print("[TrackWindowManager] Logger thread stopped.")

        # 2. 停止定时快照线程（weakref 处理 manager 丢失情形）
        if hasattr(self, "snapshot_scheduler") and self.snapshot_scheduler is not None:
            self.snapshot_scheduler.stop()
            self.snapshot_scheduler.join(timeout=timeout)
            print("[TrackWindowManager] Snapshot scheduler stopped.")

        # 3. 退出前再保存一次完整快照（保险起见）
        final_path = self.save_snapshot()
        print(f"[TrackWindowManager] Final snapshot saved to {final_path}")

        print("[TrackWindowManager] Shutdown complete.")

    def get_all_tracks_and_windows(self) -> Dict[str, List[TrackObservation]]:
        """
        获取所有活跃轨迹窗口内容
        :return: dict，格式为 {track_id: [TrackObservation, ...]}
        """
        result = {}
        for track_id, window in self.windows.items():
            # 拷贝一份，防止外部修改原始 deque
            result[track_id] = list(window.history)
        return result

    def get_serialized_track_windows(self) -> Dict[str, List[Dict]]:
        """
        获取序列化后的窗口内容（可写入日志）
        :return: dict，格式为 {track_id: [dict]}
        """
        result = {}
        for track_id, window in self.windows.items():
            result[track_id] = [
                {
                    "frame_index": obs.frame_index,
                    "timestamp": obs.timestamp,
                    "conf": obs.entry.conf,
                    "bbox": [obs.entry.x1, obs.entry.y1, obs.entry.x2, obs.entry.y2],
                    "cls_id": obs.entry.cls_id,
                    "view_name": obs.entry.view_name,
                }
                for obs in window.history
            ]
        return result

    def detect_abnormal_tracks(
            self,
            hook_fn: Callable[[str, List[TrackObservation]], Optional[Dict]]
    ) -> Dict[str, Dict]:
        """
        对所有轨迹窗口执行自定义 hook 函数，收集异常轨迹
        :param hook_fn: 回调函数，输入为 (track_id, observations)，返回值为异常信息 dict（若无异常则返回 None）
        :return: 所有异常轨迹结果：{track_id: 异常信息dict}
        """
        abnormal_results = {}
        for track_id, window in self.windows.items():
            obs_list = list(window.history)
            result = hook_fn(track_id, obs_list)
            if result is not None:
                abnormal_results[track_id] = result
        return abnormal_results
