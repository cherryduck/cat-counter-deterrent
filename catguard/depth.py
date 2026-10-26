"""Depth-based 'something is on the counter' detection.

Depth frames are uint16 millimetres (0 = invalid), registered to the RGB image.
The empty counter is captured once as a per-pixel baseline; anything that sits
between min_height_mm and max_height_mm *closer* to the camera than the baseline
inside the counter ROI is a candidate blob.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .config import DepthCfg


@dataclass
class Blob:
    bbox: tuple[int, int, int, int]  # x, y, w, h (pixels)
    pixels: int
    area_m2: float
    height_mm: float                 # median height above baseline
    inside_roi: float                # fraction of pixels inside ROI
    depth_mm: float                  # median distance from camera
    mask: np.ndarray                 # bool mask, full frame size

    @property
    def center(self) -> tuple[int, int]:
        x, y, w, h = self.bbox
        return x + w // 2, y + h // 2


def roi_mask(shape: tuple[int, int], polygon: list[list[int]]) -> np.ndarray:
    m = np.zeros(shape, np.uint8)
    if polygon:
        cv2.fillPoly(m, [np.asarray(polygon, np.int32)], 1)
    else:
        m[:] = 1
    return m.astype(bool)


def build_baseline(frames: list[np.ndarray]) -> np.ndarray:
    """Per-pixel median of valid samples; 0 where never valid."""
    stack = np.stack(frames).astype(np.float32)
    stack[stack == 0] = np.nan
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            med = np.nanmedian(stack, axis=0)
    return np.nan_to_num(med, nan=0.0).astype(np.float32)


class Calibration:
    """Baseline depth + ROI polygon, persisted in calibration_dir."""

    def __init__(self, baseline: np.ndarray, polygon: list[list[int]]):
        self.baseline = baseline.astype(np.float32)
        self.polygon = polygon
        self.roi = roi_mask(baseline.shape, polygon)

    def save(self, d: Path):
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / "baseline.npy", self.baseline)
        (d / "roi.json").write_text(json.dumps({"polygon": self.polygon}))

    @classmethod
    def load(cls, d: Path) -> "Calibration":
        baseline = np.load(d / "baseline.npy")
        poly = []
        if (d / "roi.json").exists():
            poly = json.loads((d / "roi.json").read_text())["polygon"]
        return cls(baseline, poly)

    def set_polygon(self, polygon):
        self.polygon = polygon
        self.roi = roi_mask(self.baseline.shape, polygon)


class DepthDetector:
    def __init__(self, cfg: DepthCfg, calib: Calibration):
        self.cfg = cfg
        self.calib = calib
        k = max(1, cfg.open_kernel)
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        # static-blob tracking for baseline absorption
        self._static_since: float | None = None
        self._static_ref: np.ndarray | None = None

    def height_map(self, depth: np.ndarray) -> np.ndarray:
        d = depth.astype(np.float32)
        b = self.calib.baseline
        h = b - d
        h[(d == 0) | (b == 0)] = 0.0
        return h

    def detect(self, depth: np.ndarray) -> list[Blob]:
        c = self.cfg
        h = self.height_map(depth)
        # Whole-frame foreground with NO upper height bound: a person leaning in
        # forms one component reaching outside the ROI (low inside fraction), so
        # it is rejected. Height limits are applied per-blob on the in-ROI part.
        fg = (h >= c.min_height_mm).astype(np.uint8)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, self._kernel)
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, self._kernel)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
        blobs: list[Blob] = []
        roi = self.calib.roi
        for i in range(1, n):
            px = int(stats[i, cv2.CC_STAT_AREA])
            if px < 30:
                continue
            m = labels == i
            m_in = m & roi
            n_in = int(np.count_nonzero(m_in))
            if n_in == 0:
                continue
            inside = n_in / px
            dv = depth[m]
            dv = dv[dv > 0]
            z = float(np.median(dv)) if dv.size else 0.0
            area = px * (z / c.focal_px) ** 2 / 1e6 if z else 0.0  # mm^2 -> m^2
            x, y, w, hh = (int(stats[i, k]) for k in
                           (cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP, cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT))
            blobs.append(Blob((x, y, w, hh), px, area, float(np.median(h[m_in])), inside, z, m))
        return blobs

    def is_cat_sized(self, b: Blob) -> bool:
        c = self.cfg
        return (c.min_area_m2 <= b.area_m2 <= c.max_area_m2
                and b.height_mm <= c.max_height_mm
                and b.inside_roi >= c.min_inside_roi)

    def maybe_absorb(self, depth: np.ndarray, blobs: list[Blob], is_cat: bool,
                     now: float | None = None) -> bool:
        """Absorb a long-static, non-cat object (e.g. a kettle) into the baseline.

        Returns True if the baseline was updated.
        """
        now = time.monotonic() if now is None else now
        cand = [b for b in blobs if b.inside_roi > 0]
        if not cand or is_cat:
            self._static_since, self._static_ref = None, None
            return False
        union = np.zeros(depth.shape, bool)
        for b in cand:
            union |= b.mask
        cur = depth.astype(np.float32)
        if (self._static_ref is None or self._static_since is None
                or self._static_ref.shape != cur.shape):
            self._static_since, self._static_ref = now, cur
            return False
        valid = union & (cur > 0) & (self._static_ref > 0)
        diff = np.abs(cur[valid] - self._static_ref[valid])
        if diff.size == 0 or np.median(diff) > self.cfg.static_tolerance_mm:
            self._static_since, self._static_ref = now, cur
            return False
        if now - self._static_since >= self.cfg.absorb_after_s:
            grown = cv2.dilate(union.astype(np.uint8), self._kernel, iterations=2).astype(bool)
            upd = grown & (cur > 0)
            self.calib.baseline[upd] = cur[upd]
            self._static_since, self._static_ref = None, None
            return True
        return False


def colorize_depth(depth: np.ndarray, max_mm: int = 4000) -> np.ndarray:
    d = np.clip(depth.astype(np.float32) / max_mm * 255, 0, 255).astype(np.uint8)
    img = cv2.applyColorMap(255 - d, cv2.COLORMAP_JET)
    img[depth == 0] = 0
    return img
