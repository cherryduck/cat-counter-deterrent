"""Background counter scan: runs the (slow) full-ROI cat scan on the latest frame
in its own thread so the depth loop keeps its frame rate."""
from __future__ import annotations

import logging
import threading
import time

import numpy as np

log = logging.getLogger(__name__)


class ScanWorker:
    def __init__(self, detector, person_detector=None, person_conf: float = 0.4):
        self.detector = detector
        self.person_detector = person_detector
        self.person_conf = person_conf
        self.person: tuple = (-1e9, 0.0, [])     # (frame time last seen, score, boxes)
        self.person_ms = 0.0
        self._frame: tuple | None = None
        self._cv = threading.Condition()
        self._streak = 0
        self.result: tuple = (False, 0.0, None, 0.0, None, 0)   # on, score, box, ts, rgb, streak
        self.scan_ms = 0.0
        self._stop = False
        self._t = threading.Thread(target=self._loop, daemon=True, name="scan")
        self._t.start()

    def submit(self, rgb: np.ndarray, region, roi: np.ndarray):
        with self._cv:
            self._frame = (rgb, region, roi)
            self._cv.notify()

    def latest(self, max_age_s: float = 1.0):
        """(on, score, box, rgb_of_scanned_frame, streak) - empty if the result is stale.

        `on` is the raw scan result (cat on the counter); whether it counts as a hit
        (streak / strong score, see app.scan_confirmed) is decided by the pipeline,
        which has the depth to judge a lone strong score."""
        on, score, box, ts, rgb, streak = self.result
        if time.monotonic() - ts > max_age_s:
            return False, 0.0, None, None, 0
        return on, score, box, rgb, streak

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
            if self.person_detector is not None:
                t2 = time.perf_counter()
                try:
                    ps, boxes = self.person_detector.persons(rgb, self.person_conf)
                    if boxes:
                        self.person = (t_frame, ps, boxes)
                except Exception:
                    log.exception("person check failed")
                self.person_ms = (time.perf_counter() - t2) * 1e3
            self._streak = self._streak + 1 if on else 0
            # age is measured from when the frame was captured, not when the scan finished
            self.result = (on, score, box, t_frame, rgb, self._streak)
