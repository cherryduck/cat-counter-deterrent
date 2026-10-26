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


@dataclass
class DetectorCfg:
    model: str = "models/yolo11n_640.onnx"
    imgsz: int = 640                # model input size (must match the exported model)
    scan_tile: int = 320            # counter scan tile size in camera pixels (upscaled to imgsz)
    cat_conf: float = 0.35
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
