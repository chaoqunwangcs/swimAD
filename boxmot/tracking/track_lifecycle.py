import threading
from typing import Optional

from tracking.track_window_manager import TrackWindowManager


class TrackManagerController:
    """
    TrackWindowManager 生命周期托管器（带线程管理、异常安全、上下文封装）
    """

    def __init__(
            self,
            max_window_size: int = 30,
            ttl: int = 10,
            save_dir: str = "track_logs",
            snapshot_interval: int = 60,
            auto_shutdown: bool = True,
            frame_index: Optional[int] = None
    ):
        self._manager = TrackWindowManager(
            max_window_size=max_window_size,
            ttl=ttl,
            save_dir=save_dir,
            snapshot_interval=snapshot_interval,
            frame_window_span=frame_index,
        )
        self._closed = False
        self._auto_shutdown = auto_shutdown
        self._lock = threading.Lock()

    @property
    def manager(self) -> TrackWindowManager:
        if self._closed:
            raise RuntimeError("TrackWindowManager is already closed.")
        return self._manager

    @property
    def closed(self) -> bool:
        return self._closed

    def shutdown(self):
        with self._lock:
            if self._closed:
                return
            try:
                self._manager.shutdown()
            except Exception as e:
                print(f"[TrackManagerController] Shutdown error: {e}")
            finally:
                self._closed = True

    def __enter__(self):
        return self.manager

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._auto_shutdown:
            self.shutdown()
