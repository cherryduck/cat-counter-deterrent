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
        # adaptive background: per-pixel occupancy (EMA of "above"/"below" baseline)
        self._occ_up: np.ndarray | None = None
        self._occ_dn: np.ndarray | None = None
        self.scenery: np.ndarray | None = None    # pixels currently treated as background
        self._last_now: float | None = None
        self.adapted_px = 0
        self._ref_id = None    # id of the calibrated baseline the drift limit refers to
        self._ref = None
        self._adapt_dt = 0.0
        self._adapt_n = 0

    def reset_static(self):
        self._static_since, self._static_ref = None, None

    def height_map(self, depth: np.ndarray) -> np.ndarray:
        d = depth.astype(np.float32)
        b = self.calib.baseline
        h = b - d
        h[(d == 0) | (b == 0)] = 0.0
        return h

    def threshold_map(self) -> np.ndarray:
        """Per-pixel height threshold (mm).

        max(min_height, noise_k * measured noise, quant_k * Kinect depth step).
        The Kinect measures disparity, so depth comes in steps of ~2.85e-6*z^2 mm
        (~40 mm at 3.7 m): a fixed 60 mm threshold is barely one step far away.
        """
        c = self.cfg
        b = self.calib.baseline
        noise = self.calib.noise
        key = (id(b), id(noise), c.min_height_mm, c.noise_k, c.quant_k)
        if getattr(self, "_thr_key", None) != key:
            thr = np.full(b.shape, c.min_height_mm, np.float32)
            thr = np.maximum(thr, c.quant_k * 2.85e-6 * b.astype(np.float32) ** 2)
            if noise is not None:
                thr = np.maximum(thr, c.noise_k * noise)
            self._thr = thr
            self._thr_key = key
        return self._thr

    def detect(self, depth: np.ndarray, now: float | None = None) -> list[Blob]:
        c = self.cfg
        if c.temporal_frames > 1:
            self._recent.append(depth)
            depth = temporal_median(list(self._recent)[-c.temporal_frames:])
        self.filtered = depth
        h = self.height_map(depth)
        thr = self.threshold_map()
        # Whole-frame foreground with NO upper height bound: a person leaning in
        # forms one component reaching outside the ROI (low inside fraction), so
        # it is rejected. Height limits are applied per-blob on the in-ROI part.
        up = h >= thr
        fg = up.astype(np.uint8)
        if self.scenery is not None:
            fg[self.scenery] = 0
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
        self._adapt(depth, h, thr, up, blobs, now)
        return blobs

    def _adapt(self, depth, h, thr, up, blobs, now):
        """Slowly learn the background so lighting drift / objects left out heal.

        Each pixel keeps an occupancy EMA of being above (or below) the baseline.
        - occupancy > scenery_occ: masked out of detection (persistent, not new)
        - occupancy > absorb_occ:  baseline set to the current depth
        Pixels under a person are frozen (people standing still mustn't vanish).
        Time constant adapt_tau_s (~8 s): a cat that just arrived is detected long
        before it could become scenery; one that sits still for ~20 s fades out.
        """
        c = self.cfg
        if c.adapt_tau_s <= 0:
            return
        dt = 1 / 15 if now is None or self._last_now is None else max(0.0, min(1.0, now - self._last_now))
        self._last_now = now
        # Runs every adapt_every frames with the accumulated dt (tau is seconds, so
        # this is equivalent to per-frame for learning; saves ~8 ms/frame on a Pi 5).
        self._adapt_dt += dt
        self._adapt_n += 1
        if self._adapt_n % c.adapt_every and self.scenery is not None:
            return
        a = float(min(1.0, self._adapt_dt / c.adapt_tau_s))
        self._adapt_dt = 0.0
        if self._occ_up is None or self._occ_up.shape != h.shape:
            self._occ_up = np.zeros(h.shape, np.float32)
            self._occ_dn = np.zeros(h.shape, np.float32)
        # Only *drift* may be learned: depth within adapt_max_steps Kinect depth
        # steps (min adapt_max_mm) of the calibrated empty counter. Lighting bias is
        # 1-4 steps; a person (>= ~0.5 m closer) never qualifies, so someone sitting
        # still can't be absorbed and then show up as cat-sized pieces when they move.
        if self._ref_id != id(self.calib.baseline):        # new calibration
            ref = self.calib.baseline.copy()
            self._ref, self._ref_id = ref, id(self.calib.baseline)
            lim = np.maximum(c.adapt_max_mm, c.adapt_max_steps * 2.85e-6 * ref ** 2)
            u16 = lambda x: np.clip(x, 1, 65535).astype(np.uint16)   # lower bound >= 1: excludes dropouts
            self._drift_lo, self._drift_hi = u16(ref - lim), u16(ref + lim)
            self._scen_lo = u16(ref - c.max_height_mm)    # scenery: up to cat height above the counter
            bad = ref <= 0
            for arr in (self._drift_lo, self._scen_lo):
                arr[bad] = 65535                         # no reference -> never in range
        drift_ok = cv2.inRange(depth, self._drift_lo, self._drift_hi)       # uint8 0/255
        scen_ok = cv2.inRange(depth, self._scen_lo, self._drift_hi)
        dn = ((h <= -thr) & (depth > 0)).view(np.uint8)
        upd = None
        person = [b.mask for b in blobs if self.is_person(b)]
        if person:
            frozen = np.zeros(h.shape, bool)
            for m in person:
                frozen |= m
            upd = (~frozen).view(np.uint8)
        cv2.accumulateWeighted(up.view(np.uint8), self._occ_up, a, mask=upd)
        cv2.accumulateWeighted(dn, self._occ_dn, a, mask=upd)
        # Scenery (masked, not learnt): anything persistently there up to cat height,
        # e.g. a kettle left out. Not people: their pixels are far above that.
        self.scenery = (self._occ_up > c.scenery_occ) & (scen_ok > 0)
        absorb = ((self._occ_up > c.absorb_occ) | (self._occ_dn > c.absorb_occ)) & (drift_ok > 0)
        if upd is not None:
            absorb &= upd.view(bool)
        n = int(np.count_nonzero(absorb))
        if n:
            self.calib.baseline[absorb] = depth[absorb]
            self._occ_up[absorb] = 0
            self._occ_dn[absorb] = 0
            self.adapted_px += n
            self._thr_key = None   # baseline changed -> threshold map depends on it

    @property
    def reference(self) -> np.ndarray:
        """Calibrated empty-counter depth (not the adapted one)."""
        if self._ref_id == id(self.calib.baseline) and self._ref is not None:
            return self._ref
        return self.calib.baseline

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
        self._tracks: list[dict] = []                              # {mask, last, arrival, n}
        self._seen: np.ndarray | None = None                       # last time each pixel was in a candidate
        self.footprint_s = 30.0
        self.arrive_frames = 4
        self.min_moves = 3          # movement must be seen in >= this many checks...
        self.moves_window_s = 1.0   # ...within this window (flicker gives isolated hits)
        self._moves: list[tuple[np.ndarray, float]] = []
        # territory memory: where candidates have been in the last territory_mem_s,
        # applied with a delay so the cat's own last few frames don't count
        self.territory_frac = 0.12
        self.territory_mem_s = 3.0
        self.territory_delay_s = 0.4
        self._terr_pending: deque[tuple[float, np.ndarray]] = deque()
        self._terr_seen: np.ndarray | None = None
        self._terr_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
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
        if self._seen is None or self._seen.shape != depth.shape:
            self._seen = np.full(depth.shape, -1e9, np.float64)
            self._terr_seen = np.full(depth.shape, -1e9, np.float64)
        while self._terr_pending and now - self._terr_pending[0][0] >= self.territory_delay_s:
            t, m = self._terr_pending.popleft()
            self._terr_seen[m] = t
        recent_area = (now - self._terr_seen) <= self.territory_mem_s
        known = cv2.dilate(recent_area.view(np.uint8), self._terr_kernel).view(bool) if recent_area.any() else None
        for b in cands:
            why = ""
            # --- arrival: a new blob with no person around = it got there by itself.
            # Not if something was at that spot recently (flickering phantoms keep
            # "re-arriving"), and only once it has persisted for a few frames.
            tr = next((t for t in self._tracks
                       if np.count_nonzero(t["mask"] & b.mask) >= 0.3 * b.pixels), None)
            if tr is None:
                fresh = np.mean(now - self._seen[b.mask] > self.footprint_s) > 0.7
                tr = {"arrival": fresh and now - self._last_person > self.person_window_s,
                      "born": now, "n": 0}
                self._tracks.append(tr)
            tr["n"] += 1
            if tr["arrival"] and tr["n"] == self.arrive_frames:
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
                    # "new ground": share of the blob outside everywhere candidates have
                    # been in the last few seconds. Flickering phantoms split/merge
                    # inside their own footprint (centroid jumps ~20 px, IoU drops)
                    # but rarely gain ground; a walking cat gains ground every frame.
                    terr = (np.count_nonzero(b.mask & ~known) / b.pixels) if known is not None else 0.0
                    self.last_score = max(self.last_score, terr)
                    if terr >= self.territory_frac:
                        why = f"new ground {terr:.2f}"
                    elif frac >= self.interior_frac:
                        why = f"interior {frac:.2f}"
            if why and why != "arrived" and any(np.count_nonzero(m & b.mask) > 0.05 * b.pixels
                                                 for m, _ in self._person):
                why = ""   # a person was just here: it's their hand, not the object
            if why and why != "arrived":
                self._moves = [(m, t) for m, t in self._moves if now - t <= self.moves_window_s]
                n_prev = sum(np.count_nonzero(m & b.mask) >= 0.3 * b.pixels for m, _ in self._moves)
                self._moves.append((b.mask, now))
                if n_prev + 1 < self.min_moves:
                    why = ""          # isolated movement reading: wait for confirmation
            if why:
                self._moving_until.append((b.mask, now + self.hold_s))
                self.last_why = why
            recent = any(np.count_nonzero(m & b.mask) >= 0.3 * b.pixels for m, _ in self._moving_until)
            out.append(bool(why) or recent)
        u = np.zeros(depth.shape, bool)
        for b in cands:
            self._seen[b.mask] = now
            u |= b.mask
        self._terr_pending.append((now, u))
        return out
