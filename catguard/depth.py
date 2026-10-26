"""Depth-based 'something is on the counter' detection.

Depth frames are uint16 millimetres (0 = invalid), registered to the RGB image.
The empty counter is captured once as a per-pixel baseline; anything that sits
between min_height_mm and max_height_mm *closer* to the camera than the baseline
inside the counter ROI is a candidate blob.
"""
from __future__ import annotations

import json
import time
from collections import deque
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


def build_noise(frames: list[np.ndarray]) -> np.ndarray:
    """Per-pixel depth noise (robust sigma, mm) of the temporally-filtered signal.

    Pixels that drop out (0) in more than 20% of frames are unreliable: they get
    a large sigma so only a big height change there counts.
    """
    filt = [temporal_median(frames[max(0, i - 2):i + 1]) for i in range(2, len(frames))] or frames
    stack = np.stack(filt).astype(np.float32)
    valid = stack > 0
    rate = valid.mean(axis=0)
    stack[~valid] = np.nan
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        med = np.nanmedian(stack, axis=0)
        mad = np.nanmedian(np.abs(stack - med), axis=0)
    sigma = np.nan_to_num(mad * 1.4826, nan=100.0).astype(np.float32)
    sigma[rate < 0.8] = np.maximum(sigma[rate < 0.8], 50.0)
    return sigma


def temporal_median(frames: list[np.ndarray]) -> np.ndarray:
    """Median of up to 3 uint16 depth frames, treating 0 as a dropout.

    3 valid samples -> median; 1 dropout -> the farther of the two valid samples
    (errs towards "nothing there"); 2+ dropouts -> invalid (0), i.e. flicker.
    Uses an OpenCV min/max network on uint16 (~5 ms on a Pi 5 vs ~23 ms sorting).
    """
    if len(frames) == 1:
        return frames[0]
    if len(frames) == 2:
        a, b = frames[-2:]
        out = cv2.max(a, b)
        out[(a == 0) | (b == 0)] = 0
        return out
    a, b, c = frames[-3:]
    lo, hi = cv2.min(a, b), cv2.max(a, b)
    out = cv2.max(lo, cv2.min(hi, c))                 # median-of-3 (zeros sort lowest)
    zeros = (a == 0).view(np.uint8) + (b == 0).view(np.uint8) + (c == 0).view(np.uint8)
    one = zeros == 1
    out[one] = cv2.max(hi, c)[one]
    out[zeros >= 2] = 0
    return out


class Calibration:
    """Baseline depth + per-pixel noise + ROI polygon, persisted in calibration_dir."""

    def __init__(self, baseline: np.ndarray, polygon: list[list[int]], noise: np.ndarray | None = None):
        self.baseline = baseline.astype(np.float32)
        self.noise = noise.astype(np.float32) if noise is not None else None
        self.polygon = polygon
        self.roi = roi_mask(baseline.shape, polygon)

    def save(self, d: Path):
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / "baseline.npy", self.baseline)
        if self.noise is not None:
            np.save(d / "noise.npy", self.noise)
        (d / "roi.json").write_text(json.dumps({"polygon": self.polygon}))

    @classmethod
    def load(cls, d: Path) -> "Calibration":
        baseline = np.load(d / "baseline.npy")
        noise = np.load(d / "noise.npy") if (d / "noise.npy").exists() else None
        poly = []
        if (d / "roi.json").exists():
            poly = json.loads((d / "roi.json").read_text())["polygon"]
        return cls(baseline, poly, noise)

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
        self._recent: deque[np.ndarray] = deque(maxlen=3)
        self.filtered: np.ndarray | None = None   # last temporally-filtered depth

    def reset_static(self):
        self._static_since, self._static_ref = None, None

    def height_map(self, depth: np.ndarray) -> np.ndarray:
        d = depth.astype(np.float32)
        b = self.calib.baseline
        h = b - d
        h[(d == 0) | (b == 0)] = 0.0
        return h

    def threshold_map(self) -> np.ndarray | float:
        c = self.cfg
        noise = self.calib.noise
        if noise is None:
            return c.min_height_mm
        key = (id(noise), c.min_height_mm, c.noise_k)
        if getattr(self, "_thr_key", None) != key:
            self._thr = np.maximum(c.min_height_mm, c.noise_k * noise).astype(np.float32)
            self._thr_key = key
        return self._thr

    def detect(self, depth: np.ndarray) -> list[Blob]:
        c = self.cfg
        if c.temporal_frames > 1:
            self._recent.append(depth)
            depth = temporal_median(list(self._recent)[-c.temporal_frames:])
        self.filtered = depth
        h = self.height_map(depth)
        # Whole-frame foreground with NO upper height bound: a person leaning in
        # forms one component reaching outside the ROI (low inside fraction), so
        # it is rejected. Height limits are applied per-blob on the in-ROI part.
        fg = (h >= self.threshold_map()).astype(np.uint8)
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

    def is_person(self, b: Blob) -> bool:
        """Person-like: bigger than any cat, or a sizeable blob reaching the bottom
        of the image (someone standing at the counter always does)."""
        if self.is_cat_sized(b):
            return False
        H = b.mask.shape[0]
        x, y, w, h = b.bbox
        return b.area_m2 > self.cfg.max_area_m2 or (y + h >= H - 2 and b.area_m2 >= self.cfg.person_min_area_m2)

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


class MotionGate:
    """Is a cat-sized blob *animate*? A cat moves or arrives by itself; a bag
    that a person has just put down does neither.

    A candidate is animate if, compared with the same blob `lag` frames ago, it
    has moved (centroid shift), changed shape (low mask IoU), or its *interior*
    depth changed. Edge pixels are excluded: depth noise at object edges is
    large and made a static bag look like it was moving.
    Measured on a static bag: shift <= 2.2 px, IoU >= 0.86, interior-change <= 5%.

    It is also animate if it *arrived* (new blob) with no person-sized blob in
    the scene in the previous `person_window_s` - i.e. a cat jumping up. Things
    placed by a person are born with a person present, so they don't count.
    """

    def __init__(self, lag: int = 5, motion_mm: float = 40.0, interior_frac: float = 0.2,
                 shift_px: float = 5.0, iou_min: float = 0.7, hold_s: float = 2.0,
                 settle_s: float = 1.5, person_window_s: float = 2.0):
        self.lag, self.motion_mm, self.interior_frac = lag, motion_mm, interior_frac
        self.shift_px, self.iou_min, self.hold_s = shift_px, iou_min, hold_s
        self.settle_s, self.person_window_s = settle_s, person_window_s
        self.hist: deque[tuple[np.ndarray, list[Blob]]] = deque(maxlen=lag + 1)
        self._moving_until: list[tuple[np.ndarray, float]] = []   # (mask, until)
        self._person: list[tuple[np.ndarray, float]] = []         # (mask, until): touched by a person
        self._last_person = -1e9
        self._tracks: list[dict] = []                              # {mask, last, arrival}
        self._kernel = np.ones((5, 5), np.uint8)
        self.last_score: float = 0.0
        self.last_why: str = ""

    def _match(self, b: Blob, others: list[Blob]):
        prev = [o for o in others if np.count_nonzero(o.mask & b.mask) >= 0.3 * min(o.pixels, b.pixels)]
        return max(prev, key=lambda o: np.count_nonzero(o.mask & b.mask)) if prev else None

    def update(self, depth: np.ndarray, cands: list[Blob], now: float,
               person_masks: list[np.ndarray] = ()) -> list[bool]:
        """Return, for each candidate, whether it's animate."""
        self._person = [(m, t) for m, t in self._person if t > now]
        for m in person_masks:
            self._person.append((cv2.dilate(m.astype(np.uint8), np.ones((15, 15), np.uint8)).astype(bool),
                                 now + self.settle_s))
        if person_masks:
            self._last_person = now
        self.hist.append((depth, cands))
        self._moving_until = [(m, t) for m, t in self._moving_until if t > now]
        self._tracks = [t for t in self._tracks if now - t["last"] < 0.5]
        old = self.hist[0] if len(self.hist) > self.lag else None
        out = []
        self.last_score, self.last_why = 0.0, ""
        for b in cands:
            why = ""
            # --- arrival: a new blob with no person around = it got there by itself
            tr = next((t for t in self._tracks
                       if np.count_nonzero(t["mask"] & b.mask) >= 0.3 * b.pixels), None)
            if tr is None:
                tr = {"arrival": now - self._last_person > self.person_window_s, "born": now}
                self._tracks.append(tr)
                if tr["arrival"]:
                    why = "arrived"
            tr["mask"], tr["last"] = b.mask, now
            # --- movement vs `lag` frames ago
            if old is not None and not why:
                od, oc = old
                o = self._match(b, oc)
                if o is not None:
                    inter = np.count_nonzero(o.mask & b.mask)
                    iou = inter / max(1, np.count_nonzero(o.mask | b.mask))
                    (cx, cy), (ox, oy) = b.center, o.center
                    shift = float(np.hypot(cx - ox, cy - oy))
                    core = cv2.erode((b.mask & o.mask).astype(np.uint8), self._kernel).astype(bool)
                    core &= (depth > 0) & (od > 0)
                    frac = (float(np.mean(np.abs(depth[core].astype(np.float32) - od[core]) > self.motion_mm))
                            if core.any() else 0.0)
                    self.last_score = max(self.last_score, frac, 1 - iou)
                    if shift >= self.shift_px:
                        why = f"shift {shift:.0f}px"
                    elif iou < self.iou_min:
                        why = f"shape iou {iou:.2f}"
                    elif frac >= self.interior_frac:
                        why = f"interior {frac:.2f}"
            if why and why != "arrived" and any(np.count_nonzero(m & b.mask) > 0.05 * b.pixels
                                                 for m, _ in self._person):
                why = ""   # a person was just here: it's their hand, not the object
            if why:
                self._moving_until.append((b.mask, now + self.hold_s))
                self.last_why = why
            recent = any(np.count_nonzero(m & b.mask) >= 0.3 * b.pixels for m, _ in self._moving_until)
            out.append(bool(why) or recent)
        return out
