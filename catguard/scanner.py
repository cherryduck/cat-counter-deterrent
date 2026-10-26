"""Background counter scan: runs the (slow) full-ROI cat scan on the latest frame
in its own thread so the depth loop keeps its frame rate."""
from __future__ import annotations

import logging
import threading
import time

import numpy as np

log = logging.getLogger(__name__)


class ScanWorker:
    def __init__(self, detector, confirm_scans: int = 2, strong_conf: float = 0.6):
        self.detector = detector
        self.confirm_scans = max(1, confirm_scans)
        self.strong_conf = strong_conf
        self._frame: tuple | None = None
        self._cv = threading.Condition()
        self._streak = 0
        self.result: tuple = (False, 0.0, None, 0.0, None)   # hit, score, box, ts, rgb
        self.scan_ms = 0.0
        self._stop = False
        self._t = threading.Thread(target=self._loop, daemon=True, name="scan")
        self._t.start()

    def submit(self, rgb: np.ndarray, region, roi: np.ndarray):
        with self._cv:
            self._frame = (rgb, region, roi)
            self._cv.notify()

    def latest(self, max_age_s: float = 1.0):
        """(hit, score, box, rgb_of_scanned_frame) - empty if the result is stale."""
        hit, score, box, ts, rgb = self.result
        if time.monotonic() - ts > max_age_s:
            return False, 0.0, None, None
        return hit, score, box, rgb

    def stop(self):
        with self._cv:
            self._stop = True
            self._cv.notify()

    def _loop(self):
        while True:
            with self._cv:
                while self._frame is None and not self._stop:
                    self._cv.wait()
                if self._stop:
                    return
                rgb, region, roi = self._frame
                self._frame = None
            t = time.perf_counter()
            t_frame = time.monotonic()
            try:
                on, score, box = self.detector.scan(rgb, region, roi)
            except Exception:
                log.exception("scan failed")
                on, score, box = False, 0.0, None
            self.scan_ms = (time.perf_counter() - t) * 1e3
            self._streak = self._streak + 1 if on else 0
            hit = on and (self._streak >= self.confirm_scans or score >= self.strong_conf)
            # age is measured from when the frame was captured, not when the scan finished
            self.result = (hit, score, box, t_frame, rgb)
