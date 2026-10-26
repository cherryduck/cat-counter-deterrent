"""The detection pipeline (shared by live run and offline replay) and the live service."""
from __future__ import annotations

import logging
import signal
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from .capture import open_source
from .config import Config
from .decision import Decider
from .depth import Blob, Calibration, DepthDetector, build_baseline
from .detector import brightness

log = logging.getLogger("catguard")


@dataclass
class Verdict:
    ts: float
    blobs: list[Blob]
    candidates: list[Blob]
    dark: bool
    cat_score: float | None
    hit: bool
    present: bool
    trigger: bool
    absorbed: bool = False
    timings: dict = field(default_factory=dict)
    cat_box: tuple | None = None

    def label(self) -> str:
        s = "DARK" if self.dark else "LIT"
        if self.cat_score is not None:
            s += f" cat={self.cat_score:.2f}"
        if self.candidates:
            s += f" cand={len(self.candidates)}"
        if self.trigger:
            s += " TRIGGER"
        return s


class Pipeline:
    def __init__(self, cfg: Config, calib: Calibration, detector=None, scanner=None):
        self.cfg = cfg
        self.scanner = scanner          # async ScanWorker (live); None = synchronous (replay/tests)
        self._streak = 0
        self.calib = calib
        self.depth = DepthDetector(cfg.depth, calib)
        self.detector = detector
        self.decider = Decider(cfg.decision)
        self.night_mode = cfg.detector.night_mode
        self._n = 0
        self._last_scan = (False, 0.0, None)
        self._roi_cache = None

    def _roi(self):
        """(bbox of ROI, ROI grown by the feet margin) - recomputed if the polygon changes."""
        key = id(self.calib.roi)
        if self._roi_cache is None or self._roi_cache[0] != key:
            m = self.calib.roi.astype(np.uint8)
            x, y, w, h = cv2.boundingRect(m) if m.any() else (0, 0, m.shape[1], m.shape[0])
            k = max(1, self.cfg.detector.roi_feet_margin_px)
            grown = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1)))
            self._roi_cache = (key, (x, y, w, h), grown.astype(bool))
        return self._roi_cache[1], self._roi_cache[2]

    def process(self, ts: float, rgb: np.ndarray, depth: np.ndarray, now: float | None = None) -> Verdict:
        now = ts if now is None else now
        self._n += 1
        t0 = time.perf_counter()
        blobs = self.depth.detect(depth)
        cands = sorted((b for b in blobs if self.depth.is_cat_sized(b)), key=lambda b: -b.pixels)
        t1 = time.perf_counter()
        (rx, ry, rw, rh), feet_roi = self._roi()
        dark = brightness(rgb[ry:ry + rh, rx:rx + rw]) < self.cfg.detector.min_brightness
        score = None
        hit = False
        cat_box = None
        if not dark and self.detector is not None:
            if cands:
                hit, score = self.detector.confirm(rgb, cands[0].bbox)
            every = self.cfg.detector.scan_every
            if not hit and every > 0:
                # depth can be blind (sunlight on the counter): scan RGB directly
                if self.scanner is not None:
                    self.scanner.submit(rgb, (rx, ry, rw, rh), feet_roi)
                    self._last_scan = self.scanner.latest()
                elif self._n % every == 0:
                    on, sc, bx = self.detector.scan(rgb, (rx, ry, rw, rh), feet_roi)
                    self._streak = self._streak + 1 if on else 0
                    d = self.cfg.detector
                    self._last_scan = (on and (self._streak >= d.scan_confirm or sc >= d.scan_strong_conf),
                                       sc, bx)
                s_hit, s_score, s_box = self._last_scan
                score = max(score or 0.0, s_score)
                if s_hit:
                    hit, cat_box = True, s_box
            elif not hit:
                self._last_scan = (False, 0.0, None)
        elif cands:
            if dark:
                hit = self.night_mode == "depth_only"
            else:
                hit = True  # no model available: depth-only
        t2 = time.perf_counter()
        trig = self.decider.update(hit, now=now)
        absorbed = self.depth.maybe_absorb(depth, blobs, is_cat=hit or self.decider.present, now=now)
        return Verdict(ts, blobs, cands, dark, score, hit, self.decider.present, trig, absorbed,
                       {"depth_ms": (t1 - t0) * 1e3, "detect_ms": (t2 - t1) * 1e3}, cat_box)


def load_detector(cfg: Config, live: bool = False):
    """Return a detector exposing confirm() (fast crop model) and scan() (accurate model)."""
    from .detector import CatDetector
    d = cfg.detector
    scan_p, conf_p = cfg.path(d.model), cfg.path(d.confirm_model)
    if not scan_p.exists():
        log.warning("model %s missing: running depth-only", scan_p)
        return None
    scan = CatDetector(d, scan_p, d.imgsz, threads=d.scan_threads if live else 4,
                       full_frame_fallback=False)
    if not conf_p.exists() or conf_p == scan_p:
        return scan
    conf = CatDetector(d, conf_p, d.confirm_imgsz, threads=2 if live else 4,
                       full_frame_fallback=False)
    return _Combined(conf, scan)


class _Combined:
    def __init__(self, conf, scan):
        self._conf, self._scan = conf, scan

    def confirm(self, rgb, bbox):
        return self._conf.confirm(rgb, bbox)

    def scan(self, rgb, region, roi):
        return self._scan.scan(rgb, region, roi)


def calibrate_frames(frames_iter, n: int):
    depths, last_rgb = [], None
    for _, rgb, depth in frames_iter:
        depths.append(depth)
        last_rgb = rgb
        if len(depths) >= n:
            break
    if not depths:
        raise RuntimeError("no frames for calibration")
    return build_baseline(depths), last_rgb


class Service:
    """Live loop: Kinect -> pipeline -> sound / evidence / Home Assistant."""

    def __init__(self, cfg: Config, dry_run: bool = False):
        from .audio import Player
        from .storage import EventStore
        self.cfg = cfg
        self.calib_dir = cfg.path(cfg.storage.calibration_dir)
        self.calib = Calibration.load(self.calib_dir)
        det = load_detector(cfg, live=True)
        self.scanner = None
        if det is not None and cfg.detector.scan_every > 0:
            from .scanner import ScanWorker
            self.scanner = ScanWorker(det, cfg.detector.scan_confirm, cfg.detector.scan_strong_conf)
        self.pipe = Pipeline(cfg, self.calib, det, self.scanner)
        self.player = Player(cfg.path(cfg.audio.sounds_dir), cfg.audio.device, cfg.audio.volume)
        self.store = EventStore(Path(cfg.storage.events_dir), Path(cfg.storage.fallback_dir).expanduser(),
                                cfg.capture.fps, cfg.storage.pre_seconds, cfg.storage.post_seconds,
                                cfg.storage.retention_days)
        self.enabled = True
        self.dry_run = dry_run
        self._recalibrate = False
        self._test_sound = False
        self._running = True
        self.ha = None
        if cfg.mqtt.enabled:
            from .mqtt_ha import HaBridge
            self.ha = HaBridge(cfg.mqtt, self.command)

    # Commands arrive on the MQTT thread; keep handlers trivial.
    def command(self, name: str, payload: str):
        restore = name.startswith("restore:")
        name = name.removeprefix("restore:")
        if name == "enabled":
            self.enabled = payload.upper() == "ON"
        elif name == "dry_run":
            self.dry_run = payload.upper() == "ON"
        elif name == "cooldown":
            self.pipe.decider.cfg.cooldown_s = float(payload)
        elif name == "volume":
            self.player.volume = max(0.0, min(1.0, float(payload) / 100))
        elif name == "night_mode" and payload in ("depth_only", "off"):
            self.pipe.night_mode = payload
        elif name == "test_sound" and not restore:
            self._test_sound = True
        elif name == "recalibrate" and not restore:
            self._recalibrate = True
        else:
            return
        if not restore:
            log.info("command %s=%s", name, payload)
            self._publish_settings()

    def _publish_settings(self):
        if not self.ha:
            return
        self.ha.state("enabled", "ON" if self.enabled else "OFF")
        self.ha.state("dry_run", "ON" if self.dry_run else "OFF")
        self.ha.state("cooldown", int(self.pipe.decider.cfg.cooldown_s))
        self.ha.state("volume", int(round(self.player.volume * 100)))
        self.ha.state("night_mode", self.pipe.night_mode)

    def stop(self, *_):
        self._running = False

    def run(self):
        from .storage import annotate
        signal.signal(signal.SIGTERM, self.stop)
        signal.signal(signal.SIGINT, self.stop)
        if self.ha:
            self.ha.start()
            time.sleep(1.0)  # let retained settings restore before publishing ours
            self._publish_settings()
            self.ha.state("cat", "OFF")
        src = open_source(self.cfg.capture.source, realtime=True)
        period = 1.0 / self.cfg.capture.fps
        last_present = False
        frames = src.frames()
        log.info("running (enabled=%s dry_run=%s)", self.enabled, self.dry_run)
        stats_t, stats_n, proc_ms = time.monotonic(), 0, 0.0
        try:
            while self._running:
                t_start = time.monotonic()
                ts, rgb, depth = next(frames)
                if self._recalibrate:
                    self._recalibrate = False
                    log.info("recalibrating baseline")
                    base, _ = calibrate_frames(frames, self.cfg.depth.baseline_frames)
                    self.calib.baseline = base
                    self.calib.save(self.calib_dir)
                    continue
                if self._test_sound:
                    self._test_sound = False
                    log.info("test sound: %s", self.player.play())
                v = self.pipe.process(ts, rgb, depth, now=time.monotonic())
                if v.absorbed:
                    log.info("static object absorbed into baseline")
                    self.calib.save(self.calib_dir)
                frame = annotate(rgb, v.candidates, datetime.now().strftime("%H:%M:%S ") + v.label(), v.cat_box)
                self.store.push(frame)
                if v.candidates:
                    log.debug("cand %s", [(round(b.area_m2, 3), round(b.height_mm)) for b in v.candidates])
                if v.present != last_present:
                    last_present = v.present
                    log.info("cat present=%s (%s)", v.present, v.label())
                    if self.ha:
                        self.ha.state("cat", "ON" if v.present else "OFF")
                if v.trigger and self.enabled:
                    name = "(dry run)" if self.dry_run else self.player.play()
                    snap, jpg = self.store.event(frame, "dry" if self.dry_run else "cat")
                    log.info("TRIGGER %s sound=%s snapshot=%s", v.label(), name, snap)
                    if self.ha:
                        self.ha.trigger(jpg, datetime.now(timezone.utc).isoformat())
                dt = time.monotonic() - t_start
                stats_n += 1
                proc_ms += dt * 1e3
                if time.monotonic() - stats_t >= 30:
                    el = time.monotonic() - stats_t
                    log.info("stats: %.1f fps, %.0f ms/frame, scan %.0f ms, %s, blobs=%d",
                             stats_n / el, proc_ms / stats_n,
                             self.scanner.scan_ms if self.scanner else 0, v.label(), len(v.blobs))
                    stats_t, stats_n, proc_ms = time.monotonic(), 0, 0.0
                if dt < period:
                    time.sleep(period - dt)
        finally:
            if self.scanner:
                self.scanner.stop()
            src.close()
            if self.ha:
                self.ha.stop()
