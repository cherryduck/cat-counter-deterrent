"""Unit tests on synthetic depth frames (no Kinect needed)."""
import numpy as np
import pytest

from catguard.config import Config
from catguard.decision import Decider
from catguard.depth import Calibration, DepthDetector, build_baseline
from catguard.app import Pipeline

H, W = 480, 640
COUNTER_MM = 1500  # camera-to-counter distance
ROI = [[100, 200], [540, 200], [540, 460], [100, 460]]


def empty():
    d = np.full((H, W), 2500, np.uint16)        # floor / far wall
    d[200:460, 100:540] = COUNTER_MM            # counter surface
    return d


def with_box(d, x, y, w, h, height_mm):
    d = d.copy()
    d[y:y + h, x:x + w] = d[y:y + h, x:x + w] - height_mm
    return d


@pytest.fixture
def det():
    cfg = Config()
    base = build_baseline([empty() for _ in range(5)])
    return DepthDetector(cfg.depth, Calibration(base, ROI)), cfg


def test_baseline_ignores_invalid():
    a, b = empty(), empty()
    a[0, 0] = 0
    base = build_baseline([a, b, empty()])
    assert base[0, 0] == 2500
    z = np.zeros((H, W), np.uint16)
    assert build_baseline([z, z])[5, 5] == 0


def test_empty_counter_no_blobs(det):
    d, _ = det
    noisy = (empty().astype(np.int32) + np.random.default_rng(0).integers(-8, 9, (H, W))).astype(np.uint16)
    assert d.detect(noisy) == []


def test_cat_sized_blob_detected(det):
    d, _ = det
    # 157x87 px blob whose top is 250 mm above a 1.5 m counter -> measured at 1.25 m ~ 0.077 m2
    blobs = d.detect(with_box(empty(), 250, 280, 157, 87, 250))
    assert len(blobs) == 1
    b = blobs[0]
    assert 0.06 < b.area_m2 < 0.10
    assert 230 < b.height_mm < 270
    assert d.is_cat_sized(b)


def test_small_object_rejected(det):
    d, _ = det
    blobs = d.detect(with_box(empty(), 300, 300, 25, 25, 150))  # mug
    assert all(not d.is_cat_sized(b) for b in blobs)


def test_person_spilling_out_of_roi_rejected(det):
    d, _ = det
    # torso leaning over the front edge: extends from below the ROI into it
    frame = empty()
    frame[380:480, 200:360] = COUNTER_MM - 300
    blobs = d.detect(frame)
    assert blobs and all(not d.is_cat_sized(b) for b in blobs)


def test_too_tall_ignored(det):
    d, _ = det
    blobs = d.detect(with_box(empty(), 250, 280, 150, 80, 900))
    assert blobs and not any(d.is_cat_sized(b) for b in blobs)


def test_static_object_absorbed(det):
    d, cfg = det
    frame = with_box(empty(), 250, 280, 150, 90, 200)  # kettle-ish, placed and left
    t = 0.0
    absorbed = False
    while t < cfg.depth.absorb_after_s + 2:
        absorbed |= d.maybe_absorb(frame, d.detect(frame), is_cat=False, now=t)
        t += 1
    assert absorbed
    assert d.detect(frame) == []


def test_decider_persistence_and_cooldown():
    cfg = Config().decision
    dec = Decider(cfg)
    assert [dec.update(h, now=i) for i, h in enumerate([True, False, True])] == [False, False, False]
    assert dec.update(True, now=3)          # 3 hits in window
    assert not dec.update(True, now=4)      # cooldown
    assert dec.update(True, now=4 + cfg.cooldown_s)


def test_decider_rate_limit():
    cfg = Config().decision
    cfg.cooldown_s = 0
    cfg.max_per_hour = 2
    dec = Decider(cfg)
    fired = [dec.update(True, now=i) for i in range(10)]
    assert sum(fired) == 2


def test_pipeline_dark_depth_only(det):
    _, cfg = det
    base = build_baseline([empty()])
    pipe = Pipeline(cfg, Calibration(base, ROI), detector=None)
    rgb = np.zeros((H, W, 3), np.uint8)  # dark
    cat = with_box(empty(), 250, 280, 157, 87, 250)
    trig = [pipe.process(i * 0.1, rgb, cat).trigger for i in range(5)]
    assert trig.count(True) == 1
    pipe2 = Pipeline(cfg, Calibration(base, ROI), detector=None)
    pipe2.night_mode = "off"
    assert not any(pipe2.process(i * 0.1, rgb, cat).trigger for i in range(5))


class FakeDetector:
    """Scan-only detector stub: reports a cat whose feet are at `feet` (or none)."""
    def __init__(self, feet=None, score=0.9):
        self.feet, self.score, self.scans = feet, score, 0

    def confirm(self, rgb, bbox):
        return False, 0.0

    def scan(self, rgb, region, roi):
        self.scans += 1
        if self.feet is None:
            return False, 0.0, None
        fx, fy = self.feet
        return bool(roi[fy, fx]), self.score, (fx - 40, fy - 60, 80, 60)


def _lit():
    return np.full((H, W, 3), 160, np.uint8)


def test_lit_scan_finds_cat_without_depth():
    cfg = Config()
    base = build_baseline([empty()])
    det = FakeDetector(feet=(300, 300))          # on the counter
    pipe = Pipeline(cfg, Calibration(base, ROI), detector=det)
    blind = np.zeros((H, W), np.uint16)         # sunlight: no depth at all
    trig = [pipe.process(i * 0.1, _lit(), blind).trigger for i in range(8)]
    assert trig.count(True) == 1
    assert det.scans == 4                        # scan_every=2


def test_lit_scan_ignores_cat_off_counter():
    cfg = Config()
    base = build_baseline([empty()])
    det = FakeDetector(feet=(300, 120))          # on top of the wall cupboards, outside ROI
    pipe = Pipeline(cfg, Calibration(base, ROI), detector=det)
    assert not any(pipe.process(i * 0.1, _lit(), empty()).trigger for i in range(10))
