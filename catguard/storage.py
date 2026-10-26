"""Event evidence: snapshot + short clip, NAS share with local fallback, retention."""
from __future__ import annotations

import logging
import os
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)


def events_root(primary: Path, fallback: Path) -> Path:
    # only use the NAS path if it's actually a mount (not an empty dir on the SD card)
    if primary.exists() and os.path.ismount(primary) and os.access(primary, os.W_OK):
        return primary
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback


def annotate(rgb: np.ndarray, blobs, label: str, cat_box=None) -> np.ndarray:
    img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if cat_box is not None:
        x, y, w, h = (int(v) for v in cat_box)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    for b in blobs:
        x, y, w, h = b.bbox
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(img, f"{b.area_m2:.3f}m2 {b.height_mm:.0f}mm", (x, max(12, y - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
    cv2.putText(img, label, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    return img


class EventStore:
    def __init__(self, primary: Path, fallback: Path, fps: float, pre_s: float, post_s: float,
                 retention_days: int):
        self.primary, self.fallback = primary, fallback
        self.fps = fps
        self.ring: deque[np.ndarray] = deque(maxlen=max(1, int(pre_s * fps)))
        self.post_frames = int(post_s * fps)
        self.retention_days = retention_days
        self._pending: list[tuple[Path, list[np.ndarray], int]] = []
        self._last_cleanup = 0.0

    def push(self, bgr: np.ndarray):
        """Feed every (annotated, BGR) frame; completes clips that are waiting on post frames."""
        self.ring.append(bgr)
        still = []
        for path, frames, remaining in self._pending:
            frames.append(bgr)
            if remaining - 1 <= 0:
                threading.Thread(target=self._write_clip, args=(path, frames), daemon=True).start()
            else:
                still.append((path, frames, remaining - 1))
        self._pending = still
        if time.time() - self._last_cleanup > 3600:
            self._last_cleanup = time.time()
            threading.Thread(target=self.cleanup, daemon=True).start()

    def event(self, snapshot_bgr: np.ndarray, tag: str) -> tuple[Path, bytes]:
        root = events_root(self.primary, self.fallback)
        day = root / datetime.now().strftime("%Y-%m-%d")
        day.mkdir(parents=True, exist_ok=True)
        stem = datetime.now().strftime("%H%M%S") + f"_{tag}"
        ok, jpg = cv2.imencode(".jpg", snapshot_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
        jpg_b = jpg.tobytes()
        snap = day / f"{stem}.jpg"
        try:
            snap.write_bytes(jpg_b)
        except OSError:
            log.exception("could not write snapshot")
        self._pending.append((day / f"{stem}.mp4", list(self.ring), self.post_frames))
        return snap, jpg_b

    def _write_clip(self, path: Path, frames: list[np.ndarray]):
        try:
            h, w = frames[0].shape[:2]
            vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (w, h))
            for f in frames:
                vw.write(f)
            vw.release()
        except Exception:
            log.exception("could not write clip %s", path)

    def cleanup(self):
        cutoff = time.time() - self.retention_days * 86400
        for root in {self.primary, self.fallback}:
            if not root.exists():
                continue
            for p in root.glob("*/*"):
                try:
                    if p.stat().st_mtime < cutoff:
                        p.unlink()
                except OSError:
                    pass
            for d in root.glob("*"):
                if d.is_dir():
                    try:
                        d.rmdir()  # only succeeds when empty
                    except OSError:
                        pass
