"""The detection pipeline (shared by live run and offline replay) and the live service."""
from __future__ import annotations

import json
import logging
import queue
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
from .depth import Blob, Calibration, DepthDetector, MotionGate, build_baseline, build_noise
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
    reason: str = ""
    snap_rgb: np.ndarray | None = None     # frame the camera detection was made on
    motion: float = 0.0

    def label(self) -> str:
        s = "DARK" if self.dark else "LIT"
        if self.cat_score is not None:
            s += f" cat={self.cat_score:.2f}"
        if self.candidates:
            s += f" cand={len(self.candidates)} mov={self.motion:.2f}"
        if self.reason:
            s += f" [{self.reason}]"
        if self.trigger:
            s += " TRIGGER"
        return s


class Pipeline:
    def __init__(self, cfg: Config, calib: Calibration, detector=None, scanner=None,
                 warmup_s: float = 0.0):
        self.cfg = cfg
        self.warmup_s = warmup_s        # live service only; replays/tests start warm
        self.scanner = scanner          # async ScanWorker (live); None = synchronous (replay/tests)
        self._streak = 0
        self.calib = calib
        self.depth = DepthDetector(cfg.depth, calib)
        dc = cfg.depth
        self.motion = MotionGate(dc.motion_lag, dc.motion_mm, dc.motion_interior_frac, dc.motion_shift_px,
                                 dc.motion_iou_min, dc.motion_hold_s, person_window_s=dc.person_window_s)
        self.detector = detector
        self.decider = Decider(cfg.decision)
        self.night_mode = cfg.detector.night_mode
        self._n = 0
        self._t_start: float | None = None
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
        c = self.cfg
        t0 = time.perf_counter()
        blobs = self.depth.detect(depth, now)
        fd = self.depth.filtered if self.depth.filtered is not None else depth
        cands = sorted((b for b in blobs if self.depth.is_cat_sized(b)), key=lambda b: -b.pixels)
        persons = [b.mask for b in blobs if self.depth.is_person(b)]
        animate = self.motion.update(fd, cands, now, persons)
        t1 = time.perf_counter()
        (rx, ry, rw, rh), feet_roi = self._roi()
        dark = brightness(rgb[ry:ry + rh, rx:rx + rw]) < c.detector.min_brightness
        score = None
        reason = ""
        cat_box = None
        snap_rgb = None
        # 1) depth: a cat-sized thing on the counter that is moving by itself
        moving = [b for b, a in zip(cands, animate) if a]
        if moving and (not dark or self.night_mode == "depth_only"):
            reason = "depth:" + (self.motion.last_why or "moving")
        # 2) camera: YOLO scan of the counter (covers depth blind spots, e.g. sunlight)
        if not dark and self.detector is not None and c.detector.scan_every > 0:
            if self.scanner is not None:
                self.scanner.submit(rgb, (rx, ry, rw, rh), feet_roi)
                s_hit, score, s_box, s_rgb = self.scanner.latest()
            else:
                if self._n % c.detector.scan_every == 0:
                    on, sc, bx = self.detector.scan(rgb, (rx, ry, rw, rh), feet_roi)
                    self._streak = self._streak + 1 if on else 0
                    d = c.detector
                    self._last_scan = (on and (self._streak >= d.scan_confirm or sc >= d.scan_strong_conf),
                                       sc, bx)
                s_hit, score, s_box = self._last_scan
                s_rgb = None
            if s_hit:
                cat_box, snap_rgb = s_box, s_rgb
                reason = reason + "+camera" if reason else "camera"
        if self._t_start is None:
            self._t_start = now
        if now - self._t_start < self.warmup_s and reason.startswith("depth"):
            reason = ""   # warm-up: motion/territory memory and background still priming
        hit = bool(reason)
        t2 = time.perf_counter()
        trig = self.decider.update(hit, now=now)
        absorbed = False   # background adaptation (DepthDetector._adapt) replaces the old absorb
        return Verdict(ts, blobs, cands, dark, score, hit, self.decider.present, trig, absorbed,
                       {"depth_ms": (t1 - t0) * 1e3, "detect_ms": (t2 - t1) * 1e3}, cat_box,
                       reason, snap_rgb, self.motion.last_score)


def load_detector(cfg: Config, live: bool = False):
    """Return the camera cat detector (used for the counter scan)."""
    from .detector import CatDetector
    d = cfg.detector
    p = cfg.path(d.model)
    if not p.exists():
        log.warning("model %s missing: running depth-only", p)
        return None
    return CatDetector(d, p, d.imgsz, threads=d.scan_threads if live else 4, full_frame_fallback=False)


def calibrate_frames(frames_iter, n: int):
    depths, last_rgb = [], None
    for _, rgb, depth in frames_iter:
        depths.append(depth)
        last_rgb = rgb
        if len(depths) >= n:
            break
    if not depths:
        raise RuntimeError("no frames for calibration")
    return build_baseline(depths), last_rgb, build_noise(depths)


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
        self.pipe = Pipeline(cfg, self.calib, det, self.scanner, warmup_s=cfg.depth.warmup_s)
        self.player = Player(cfg.path(cfg.audio.sounds_dir), cfg.audio.device, cfg.audio.volume)
        self.store = EventStore(Path(cfg.storage.events_dir), Path(cfg.storage.fallback_dir).expanduser(),
                                cfg.capture.fps, cfg.storage.pre_seconds, cfg.storage.post_seconds,
                                cfg.storage.retention_days, cfg.storage.save_raw,
                                cfg.storage.raw_retention_days)
        self.enabled = True
        self.dry_run = dry_run
        self._recalibrate = False
        self._test_sound = False
        self._running = True
        self.ha = None
        if cfg.mqtt.enabled:
            from .mqtt_ha import HaBridge
            self.ha = HaBridge(cfg.mqtt, self.command, counter_file=self.calib_dir / "triggers_today.json")
        # aiming state: Kinect tilt + "view changed since calibration" flag
        self._aim_file = self.calib_dir / "aim.json"
        aim = json.loads(self._aim_file.read_text()) if self._aim_file.exists() else {}
        self.tilt = int(aim.get("tilt", 0))
        self.calib_stale = bool(aim.get("stale", False))
        self._requests: queue.Queue = queue.Queue()
        self._fps = 0.0
        self._last_v: Verdict | None = None
        self._roi_valid = 0
        self.web = None
        if cfg.web.enabled:
            from .web import WebUI
            try:
                self.web = WebUI(self, cfg.web.port)
            except OSError:
                log.exception("live view disabled: could not bind port %d", cfg.web.port)

    # ---- live view / aiming -------------------------------------------------
    def request(self, name: str, arg):
        """Called from web threads; executed on the main loop."""
        self._requests.put((name, arg))

    def _save_aim(self):
        self._aim_file.write_text(json.dumps({"tilt": self.tilt, "stale": self.calib_stale}))

    def web_state(self) -> dict:
        v = self._last_v
        return {"tilt": self.tilt, "calib_stale": self.calib_stale, "polygon": self.calib.polygon,
                "fps": round(self._fps, 1), "label": v.label() if v else "", "present": bool(v and v.present),
                "enabled": self.enabled, "dry_run": self.dry_run,
                "triggers_today": self.ha.triggers_today if self.ha else None,
                "roi_depth_valid": self._roi_valid, "min_height_mm": self.cfg.depth.min_height_mm}

    def _handle_requests(self, src) -> bool:
        """Process queued web requests. Returns True if a recalibration was requested."""
        recal = False
        while not self._requests.empty():
            name, arg = self._requests.get_nowait()
            log.info("live view request %s %s", name, arg if name != "roi" else f"{len(arg)} points")
            if name == "tilt":
                if hasattr(src, "set_tilt"):
                    src.set_tilt(arg)
                    self.tilt = arg
                    self.calib_stale = True
                    self._save_aim()
            elif name == "roi":
                self.calib.set_polygon(arg)
                self.calib.save(self.calib_dir)
            elif name == "recalibrate":
                recal = True
            elif name == "test_sound":
                self._test_sound = True
        return recal

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
            self.ha.state("triggers_today", self.ha.triggers_today)
        src = open_source(self.cfg.capture.source, realtime=True)
        # Never move the motor at startup (it would invalidate the baseline);
        # just read where it currently is.
        if hasattr(src, "get_tilt"):
            try:
                cur = src.get_tilt()
                if cur is not None:
                    self.tilt = int(round(cur))
                    log.info("Kinect tilt is %.1f deg", cur)
            except Exception:
                log.exception("could not read tilt")
        period = 1.0 / self.cfg.capture.fps
        last_present = False
        frames = src.frames()
        log.info("running (enabled=%s dry_run=%s)", self.enabled, self.dry_run)
        stats_t, stats_n, proc_ms = time.monotonic(), 0, 0.0
        prev_start = None
        cand_since = None          # near-miss recorder: cat-sized blob present without a trigger
        cand_last_seen = -1e9
        cand_saved = False
        try:
            while self._running:
                t_start = time.monotonic()
                try:
                    ts, rgb, depth = next(frames)
                except StopIteration:
                    log.info("source ended")
                    break
                if self._handle_requests(src) or self._recalibrate:
                    self._recalibrate = False
                    log.info("recalibrating baseline")
                    base, snap, noise = calibrate_frames(frames, self.cfg.depth.baseline_frames)
                    self.calib.baseline = base
                    self.calib.noise = noise
                    self.calib.save(self.calib_dir)
                    cv2.imwrite(str(self.calib_dir / "snapshot.jpg"), cv2.cvtColor(snap, cv2.COLOR_RGB2BGR))
                    self.pipe.depth.reset_static()
                    self.calib_stale = False
                    self._save_aim()
                    continue
                if self._test_sound:
                    self._test_sound = False
                    log.info("test sound: %s", self.player.play())
                v = self.pipe.process(ts, rgb, depth, now=time.monotonic())
                if self.calib_stale:
                    v.trigger = False   # view moved since calibration: don't act on it
                self._last_v = v
                if self.web and self.web.hub.wanted:
                    roi = self.calib.roi
                    self._roi_valid = int(100 * np.count_nonzero((depth > 0) & roi) / max(1, roi.sum()))
                    self.web.hub.publish(rgb, depth, v, self.pipe.depth.height_map(depth))
                if v.absorbed:
                    log.info("static object absorbed into baseline")
                    self.calib.save(self.calib_dir)
                frame = annotate(rgb, v.candidates, datetime.now().strftime("%H:%M:%S ") + v.label(), v.cat_box)
                self.store.push_raw(rgb, depth)
                self.store.push(frame)
                if v.candidates:
                    log.debug("cand %s", [(round(b.area_m2, 3), round(b.height_mm)) for b in v.candidates])
                if v.present != last_present:
                    last_present = v.present
                    log.info("cat present=%s (%s)", v.present, v.label())
                    if self.ha:
                        self.ha.state("cat", "ON" if v.present else "OFF")
                # near-miss recorder: save raw data when something cat-sized sits on the
                # counter for 1 s without triggering (no notification / sound)
                mono = time.monotonic()
                if v.candidates:
                    if cand_since is None or mono - cand_last_seen > 10:
                        cand_since, cand_saved = mono, False
                    cand_last_seen = mono
                    if (not cand_saved and not v.present and mono - cand_since >= 1.0
                            and self.cfg.storage.save_near_misses):
                        cand_saved = True
                        snap, _ = self.store.event(frame, "cand")
                        log.info("near miss recorded (%s) %s", v.label(), snap)
                if v.trigger:
                    cand_saved = True
                if v.trigger and self.enabled:
                    name = "(dry run)" if self.dry_run else self.player.play()
                    # camera detections: snapshot the frame the cat was actually seen on
                    snap_img = (annotate(v.snap_rgb, [], datetime.now().strftime("%H:%M:%S ") + v.label(), v.cat_box)
                                if v.snap_rgb is not None else frame)
                    snap, jpg = self.store.event(snap_img, "dry" if self.dry_run else "cat")
                    log.info("TRIGGER %s sound=%s snapshot=%s", v.label(), name, snap)
                    if self.ha:
                        try:
                            self.ha.trigger(jpg, datetime.now(timezone.utc).isoformat())
                        except Exception:
                            log.exception("Home Assistant update failed (trigger still handled)")
                dt = time.monotonic() - t_start
                if prev_start is not None:
                    self._fps = 0.9 * self._fps + 0.1 / max(1e-3, t_start - prev_start)
                prev_start = t_start
                stats_n += 1
                proc_ms += dt * 1e3
                if time.monotonic() - stats_t >= 30:
                    if self.ha:
                        self.ha.roll_day()
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
            if self.web:
                self.web.stop()
            src.close()
            if self.ha:
                self.ha.stop()
