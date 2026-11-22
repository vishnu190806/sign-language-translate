from __future__ import annotations

import logging
import threading
import time
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class ThreadedCamera:
    def __init__(self, source: int = 0, backend: str = "dshow"):
        self.source = source
        self.backend = backend
        self._frame: Optional[np.ndarray] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._cap = None

    def start(self) -> ThreadedCamera:
        if self._running:
            return self

        if self.backend == "dshow":
            self._cap = cv2.VideoCapture(self.source, cv2.CAP_DSHOW)
        elif self.backend == "msmf":
            self._cap = cv2.VideoCapture(self.source, cv2.CAP_MSMF)
        else:
            self._cap = cv2.VideoCapture(self.source)

        if not self._cap.isOpened():
            logger.error(f"Failed to open camera {self.source}")
            raise RuntimeError(f"Cannot open camera {self.source}")

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self._cap.set(cv2.CAP_PROP_FPS, 30)

        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        logger.info(f"Camera {self.source} started")
        return self

    def _capture_loop(self) -> None:
        while self._running:
            ret, frame = self._cap.read()
            if ret:
                with self._lock:
                    self._frame = frame
            else:
                time.sleep(0.01)

    def read(self) -> tuple[bool, Optional[np.ndarray]]:
        with self._lock:
            if self._frame is not None:
                return True, self._frame.copy()
        return False, None

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
            if self._thread.is_alive():
                logger.warning("Camera thread did not exit; releasing capture anyway")
            self._thread = None
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception as e:
                logger.warning(f"Camera release failed: {e}")
            self._cap = None
        logger.info(f"Camera {self.source} stopped")

    def __enter__(self):
        return self.start()

    def __exit__(self, *args):
        self.stop()
