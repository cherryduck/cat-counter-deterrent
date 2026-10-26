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


def test_static_object_fades_into_background(det):
    d, cfg = det
    frame = with_box(empty(), 250, 280, 150, 90, 200)  # kettle-ish, placed and left
    first = d.detect(frame, now=0.0)
    assert any(d.is_cat_sized(b) for b in first)
    t = 0.0
    while t < 30:
        t += 1 / 15
        blobs = d.detect(frame, now=t)
    assert blobs == []                               # gone after ~30 s
    # ...and something new elsewhere is still detected straight away
    new = with_box(frame, 400, 380, 120, 70, 250)
    d.detect(new, now=t + 0.067)                     # temporal median needs it in 2 of 3 frames
    assert any(d.is_cat_sized(b) for b in d.detect(new, now=t + 0.133))


def test_night_drift_heals_without_triggering():
    """Tonight's bug: after dark the counter edge reads 2-4 depth steps closer
    than the daytime calibration, in flickering strips. Must not trigger."""
    cfg = Config()
    day = front_scene()
    pipe = Pipeline(cfg, Calibration(build_baseline([day]), FRONT), detector=None)
    rgb = np.zeros((H, W, 3), np.uint8)
    rng = np.random.default_rng(3)
    trig = []
    # Dusk, as recorded on the real Pi: the phantom strip builds up over ~a minute.
    # (Whether switching the kitchen light causes an *abrupt* shift is untested.)
    for i in range(15 * 180):
        night = day.copy()
        p_on = min(0.8, max(0.0, (i / 15 - 10) / 60 * 0.8))
        strip = np.zeros((H, W), bool)
        strip[300:328, 150:420] = rng.random((28, 270)) < p_on   # flickery strip at the front edge
        night[strip] = COUNTER_MM - 150                          # 3-4 steps "above" the counter
        trig.append(pipe.process(i / 15, rgb, night).trigger)
    assert sum(trig) == 0, f"{sum(trig)} triggers"


def test_cat_still_detected_after_long_quiet_period():
    cfg = Config()
    pipe = Pipeline(cfg, Calibration(build_baseline([front_scene()]), FRONT), detector=None)
    rgb = np.zeros((H, W, 3), np.uint8)
    for i in range(15 * 60):
        pipe.process(i / 15, rgb, front_scene())
    t0 = 60.0
    fired = [pipe.process(t0 + i / 15, rgb, with_box(front_scene(), 240, 235, 130, 75, 250)).trigger
             for i in range(15)]
    assert any(fired[:8])                            # within ~0.5 s


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


def _cat_at(x, y, base=None):
    return with_box(base if base is not None else empty(), x, y, 157, 87, 250)


def test_pipeline_dark_moving_cat_triggers():
    cfg = Config()
    base = build_baseline([empty()])
    pipe = Pipeline(cfg, Calibration(base, ROI), detector=None)
    rgb = np.zeros((H, W, 3), np.uint8)  # dark
    trig = [pipe.process(i * 0.067, rgb, _cat_at(200 + 6 * i, 280)).trigger for i in range(20)]
    assert trig.count(True) == 1
    pipe2 = Pipeline(cfg, Calibration(base, ROI), detector=None)
    pipe2.night_mode = "off"
    assert not any(pipe2.process(i * 0.067, rgb, _cat_at(200 + 6 * i, 280)).trigger for i in range(20))


def test_static_object_does_not_trigger():
    """Something cat-sized put down by a person that then just sits there is not a cat."""
    cfg = Config()
    pipe = Pipeline(cfg, Calibration(build_baseline([front_scene()]), FRONT), detector=None)
    rgb = np.zeros((H, W, 3), np.uint8)
    noise = np.random.default_rng(1)
    for i in range(80):
        d = with_box(front_scene(), 250, 230, 120, 70, 200)
        if i < 10:
            d[300:480, 150:500] = 900                                  # person standing at the counter
        d = (d.astype(np.int32) + noise.integers(-10, 11, (H, W))).astype(np.uint16)
        assert not pipe.process(i * 0.067, rgb, d).trigger, i


# Realistic scene for arrival tests: counter top occupies rows 200-330, below it
# the cupboard fronts / floor where a person stands (outside the ROI).
FRONT = [[100, 200], [540, 200], [540, 330], [100, 330]]


def front_scene():
    d = np.full((H, W), 2500, np.uint16)
    d[200:330, 100:540] = COUNTER_MM
    d[330:480, 100:540] = 1700        # cupboard fronts below the counter edge
    return d


def test_object_placed_by_hand_does_not_trigger():
    """A person puts a kettle down and walks away: the arm moving over it must not count."""
    cfg = Config()
    pipe = Pipeline(cfg, Calibration(build_baseline([front_scene()]), FRONT), detector=None)
    rgb = np.zeros((H, W, 3), np.uint8)
    trig = []
    for i in range(60):
        d = with_box(front_scene(), 250, 230, 120, 70, 200)        # the kettle
        if i < 20:
            sway = (i % 4) * 6
            d[300:480, 150 + sway:500 + sway] = 900                   # torso in front of the counter
            d[255:300, 300 + sway:340 + sway] = COUNTER_MM - 350      # forearm/hand over the kettle
        trig.append(pipe.process(i * 0.067, rgb, d).trigger)
    assert not any(trig)


def test_cat_jumping_up_triggers_quickly():
    """Cat crosses the counter edge (outside ROI) mid-jump, lands, then sits still."""
    cfg = Config()
    pipe = Pipeline(cfg, Calibration(build_baseline([front_scene()]), FRONT), detector=None)
    rgb = np.zeros((H, W, 3), np.uint8)
    fired_at = None
    for i in range(40):
        d = front_scene()
        if i < 3:
            d[290:380, 260:340] = 1300                                # mid-jump across the front edge
        else:
            d = with_box(d, 240, 235, 130, 75, 250)                   # landed, sitting still
        if pipe.process(i * 0.067, rgb, d).trigger and fired_at is None:
            fired_at = i
    assert fired_at is not None and (fired_at - 3) * 0.067 < 0.5     # arrival rule: fast


def test_cat_walking_after_being_placed_triggers():
    """Placed by hand (no arrival credit) but then walks along the counter."""
    cfg = Config()
    pipe = Pipeline(cfg, Calibration(build_baseline([front_scene()]), FRONT), detector=None)
    rgb = np.zeros((H, W, 3), np.uint8)
    fired_at = None
    for i in range(60):
        x = 200 if i < 25 else 200 + 4 * (i - 25)                     # sits 1.7 s, then walks
        d = with_box(front_scene(), x, 235, 130, 75, 250)
        if i < 10:
            d[300:480, 150:500] = 900
        if pipe.process(i * 0.067, rgb, d).trigger and fired_at is None:
            fired_at = i
    assert fired_at is not None and fired_at <= 25 + 15                # within 1 s of starting to walk


def test_flicker_suppressed_by_temporal_median(det):
    d, _ = det
    rng = np.random.default_rng(2)
    found = 0
    for i in range(30):
        f = empty()
        speck = rng.random((H, W)) < 0.15          # 15% of pixels jump 90 mm each frame, randomly
        f[speck & (np.arange(H)[:, None] > 250) & (np.arange(H)[:, None] < 330)] -= 90
        found += sum(d.is_cat_sized(b) for b in d.detect(f))
    assert found == 0


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


def test_trigger_count_persists_and_survives_offline_broker(tmp_path):
    import json
    from datetime import date, timedelta
    from catguard.config import MqttCfg
    from catguard.mqtt_ha import HaBridge
    f = tmp_path / "c.json"
    f.write_text(json.dumps({"date": date.today().isoformat(), "count": 3}))
    b = HaBridge(MqttCfg(host="127.0.0.1", port=1), lambda n, p: None, counter_file=f)
    assert b.triggers_today == 3
    b.trigger(b"jpg", "t")                       # broker unreachable: must not raise
    assert b.triggers_today == 4 and json.loads(f.read_text())["count"] == 4
    b._today = date.today() - timedelta(days=1)
    b.roll_day()
    assert b.triggers_today == 0 and json.loads(f.read_text())["count"] == 0
    f.write_text(json.dumps({"date": "2020-01-01", "count": 9}))
    assert HaBridge(MqttCfg(), lambda n, p: None, counter_file=f).triggers_today == 0
