"""Configuration loading (TOML -> nested dataclasses)."""
from __future__ import annotations

import tomllib
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path


@dataclass
class CaptureCfg:
    source: str = "kinect"          # "kinect" or path to a recorded session dir
    fps: float = 15.0               # processing rate cap


@dataclass
class DepthCfg:
    min_height_mm: float = 60.0     # blob must be at least this far above the counter
    max_height_mm: float = 600.0    # ...and no more (taller = person / cupboard noise)
    min_area_m2: float = 0.015      # real-world top-down area range for a cat
    max_area_m2: float = 0.30
    min_inside_roi: float = 0.85    # fraction of blob pixels inside the counter ROI
    focal_px: float = 525.0         # Kinect 640x480 focal length
    open_kernel: int = 5            # morphological clean-up kernel size
    baseline_frames: int = 60       # frames averaged when calibrating
    absorb_after_s: float = 180.0   # absorb static non-cat blobs into baseline after this
    static_tolerance_mm: float = 25.0
    temporal_frames: int = 3        # per-pixel median over this many frames (kills flicker)
    noise_k: float = 4.0            # per-pixel threshold = max(min_height, noise_k * sigma)
    quant_k: float = 3.0            # ...and >= quant_k Kinect depth steps (2.85e-6*z^2 mm)
    warmup_s: float = 5.0           # no depth triggers for this long after start
    adapt_tau_s: float = 8.0        # background learning time constant (0 = off)
    scenery_occ: float = 0.5        # occupancy above which a pixel is treated as background
    adapt_every: int = 3            # update the background every N frames (cost)
    adapt_max_mm: float = 150.0     # background may drift at most this far from calibration...
    adapt_max_steps: float = 5.0    # ...or this many Kinect depth steps, whichever is larger
    absorb_occ: float = 0.9         # occupancy above which the baseline is updated (~20 s)
    motion_lag: int = 5             # frames between motion comparisons (~0.33 s)
    motion_mm: float = 40.0         # per-pixel depth change counted as movement (interior only)
    motion_interior_frac: float = 0.2   # fraction of interior pixels that must change
    motion_shift_px: float = 5.0    # ...or centroid moves this far in `motion_lag` frames
    motion_iou_min: float = 0.7     # ...or shape overlap drops below this
    motion_hold_s: float = 2.0      # stays "animate" this long after last movement
    person_window_s: float = 2.0
    person_min_area_m2: float = 0.08    # bottom-edge blob at least this big counts as a person    # new blob with no person seen for this long = arrived by itself


@dataclass
class DetectorCfg:
    model: str = "models/yolo11n_640.onnx"
    imgsz: int = 640                # model input size (must match the exported model)
    scan_tile: int = 320            # counter scan tile size in camera pixels (upscaled to imgsz)
    cat_conf: float = 0.35
    person_conf: float = 0.35       # cat boxes mostly inside a person this confident are ignored
    min_brightness: float = 40.0    # mean RGB luma below this = "dark" -> depth-only
    night_mode: str = "depth_only"  # "depth_only" or "off"
    confirm_model: str = "models/yolov8n.onnx"   # fast model for tight crops around depth blobs
    confirm_imgsz: int = 320
    scan_threads: int = 3           # CPU threads for the background scan (Pi 5 has 4)
    crop_margin: float = 0.6        # crop = blob bbox grown by this fraction
    scan_every: int = 2             # when lit, scan the whole counter for cats every N frames (0 = off)
    scan_confirm: int = 2           # consecutive positive scans needed...
    scan_strong_conf: float = 0.6   # ...unless a single scan is at least this confident
    roi_feet_margin_px: int = 25    # ROI grown by this much for the "feet on counter" test
    # people (lit only): full-frame person check with the fast model
    person_model: str = "models/yolov8n.onnx"
    person_imgsz: int = 320
    person_gate_conf: float = 0.4   # measured: people 0.76-0.93, cats <= 0.20
    person_recent_s: float = 5.0    # no depth triggers while a person was seen this recently
    # camera hits must agree with depth where depth is valid
    camera_depth_recent_s: float = 10.0   # a depth candidate overlapped the box this recently...
    camera_new_frac: float = 0.35         # ...or this share of the box is above the calibrated counter
    light_jump: float = 25.0        # counter brightness change (0-255) within 1 s = light switched
    light_settle_s: float = 3.0     # no depth triggers for this long after a light switch


@dataclass
class DecisionCfg:
    persist_hits: int = 3           # candidate in >= hits of last window frames
    persist_window: int = 5
    cooldown_s: float = 45.0
    max_per_hour: int = 20


@dataclass
class AudioCfg:
    device: str = "default"         # ALSA device, e.g. "plughw:CARD=Device,DEV=0"
    sounds_dir: str = "sounds"
    volume: float = 1.0             # 0..1 software gain


@dataclass
class MqttCfg:
    enabled: bool = True
    host: str = "192.168.1.119"
    port: int = 1883
    username: str = ""
    password: str = ""
    base_topic: str = "catguard"
    discovery_prefix: str = "homeassistant"


@dataclass
class StorageCfg:
    events_dir: str = "/mnt/catcam"           # NAS share mount
    fallback_dir: str = "~/catguard-events"   # used if NAS mount missing
    calibration_dir: str = "calibration"
    retention_days: int = 30
    pre_seconds: float = 3.0
    post_seconds: float = 3.0
    save_raw: bool = True             # raw RGB+depth per event for replay tuning (~10-20 MB each)
    raw_retention_days: int = 7
    save_near_misses: bool = True     # record cat-sized blobs that didn't trigger (for tuning)


@dataclass
class WebCfg:
    enabled: bool = True
    port: int = 8080


@dataclass
class Config:
    capture: CaptureCfg = field(default_factory=CaptureCfg)
    depth: DepthCfg = field(default_factory=DepthCfg)
    detector: DetectorCfg = field(default_factory=DetectorCfg)
    decision: DecisionCfg = field(default_factory=DecisionCfg)
    audio: AudioCfg = field(default_factory=AudioCfg)
    mqtt: MqttCfg = field(default_factory=MqttCfg)
    storage: StorageCfg = field(default_factory=StorageCfg)
    web: WebCfg = field(default_factory=WebCfg)
    base_dir: Path = Path(".")

    def path(self, p: str) -> Path:
        """Resolve a config path relative to the config file's directory."""
        pp = Path(p).expanduser()
        return pp if pp.is_absolute() else (self.base_dir / pp)


def _fill(dc, data: dict):
    for f in fields(dc):
        if f.name not in data:
            continue
        cur = getattr(dc, f.name)
        if is_dataclass(cur):
            _fill(cur, data[f.name])
        else:
            setattr(dc, f.name, type(cur)(data[f.name]) if cur is not None else data[f.name])
    unknown = set(data) - {f.name for f in fields(dc)}
    if unknown:
        raise ValueError(f"unknown config keys in [{type(dc).__name__}]: {sorted(unknown)}")
    return dc


def load(path: str | Path | None) -> Config:
    cfg = Config()
    if path is None:
        return cfg
    path = Path(path)
    with open(path, "rb") as fh:
        data = tomllib.load(fh)
    _fill(cfg, data)
    cfg.base_dir = path.resolve().parent
    return cfg
