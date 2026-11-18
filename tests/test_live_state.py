"""Learnt background survives a restart (false trigger 2026-11-18 19:45:02, 7 s after a restart)."""
import numpy as np

from catguard.app import Pipeline
from catguard.config import Config
from catguard.depth import Calibration, build_baseline, load_live_state, save_live_state

from test_depth_decision import FRONT, H, W, front_scene, with_box

DARK = np.zeros((H, W, 3), np.uint8)


def _calib():
    return Calibration(build_baseline([front_scene()]), FRONT)


def _clutter():
    return with_box(front_scene(), 300, 240, 65, 50, 300)        # bowl + plate left out (cat-sized)


def _learnt_pipeline():
    """Service that has been running while someone put clutter down and left."""
    p = Pipeline(Config(), _calib(), detector=None)
    for i in range(15 * 90):
        d = _clutter()
        if i < 15 * 3:
            d[300:480, 150:500] = 900
        p.process(i / 15, DARK, d)
    return p


def test_restored_state_masks_clutter_at_once_cold_start_does_not():
    st = _learnt_pipeline().depth.export_state()
    warm = Pipeline(Config(), _calib(), detector=None)
    warm.depth.import_state(st)
    cold = Pipeline(Config(), _calib(), detector=None)
    w = warm.process(0.0, DARK, _clutter())
    c = cold.process(0.0, DARK, _clutter())
    assert not w.candidates                     # known background from the first frame
    assert c.candidates                         # cold: the clutter looks new


def test_restored_state_keeps_reference_and_still_sees_a_cat():
    p = _learnt_pipeline()
    st = p.depth.export_state()
    q = Pipeline(Config(), _calib(), detector=None)
    q.depth.import_state(st)
    assert np.array_equal(q.depth.reference, _calib().baseline)        # calibration, not learnt
    fired = []
    for i in range(30):
        d = with_box(_clutter(), 150 + 6 * i, 260, 110, 60, 250)        # cat walking in
        fired.append(q.process(100 + i / 15, DARK, d).trigger)
    assert any(fired)


def test_save_load_roundtrip_and_rejections(tmp_path):
    st = _learnt_pipeline().depth.export_state()
    fp = _calib().fingerprint()
    f = tmp_path / "live_state.npz"
    save_live_state(f, st, fp, now=1000.0)
    got, why = load_live_state(f, fp, (H, W), 3600, now=1100.0)
    assert got is not None, why
    for k in st:
        assert np.array_equal(got[k], st[k]), k
    assert load_live_state(f, fp, (H, W), 3600, now=1000.0 + 3601)[0] is None      # stale
    assert load_live_state(f, fp, (H, W), 3600, now=900.0)[0] is None              # clock went back
    assert load_live_state(f, "other", (H, W), 3600, now=1100.0)[0] is None        # recalibrated
    assert load_live_state(f, fp, (240, 320), 3600, now=1100.0)[0] is None         # wrong size
    assert load_live_state(tmp_path / "missing.npz", fp, (H, W), 3600)[0] is None
    f.write_bytes(f.read_bytes()[:1000])                                           # power cut mid-write
    assert load_live_state(f, fp, (H, W), 3600, now=1100.0)[0] is None
    assert not (tmp_path / "live_state.npz.tmp").exists()


def test_fingerprint_changes_with_calibration_and_outline():
    a = _calib()
    b = _calib(); b.baseline[300, 300] += 1
    c = _calib(); c.set_polygon([[0, 0], [10, 0], [10, 10]])
    assert a.fingerprint() == _calib().fingerprint()
    assert len({a.fingerprint(), b.fingerprint(), c.fingerprint()}) == 3


def test_cold_start_warmup_then_cat_still_triggers():
    """Cold start: depth triggers held off for cold_warmup_s (20 s); a cat at 25 s still fires."""
    cfg = Config()
    p = Pipeline(cfg, _calib(), detector=None, warmup_s=cfg.depth.cold_warmup_s)
    fired = []
    for i in range(15 * 40):
        t = i / 15
        d = _clutter()
        if 10 <= t < 13 or t >= 25:
            k = int((t - (10 if t < 13 else 25)) * 15)
            d = with_box(d, 150 + 5 * k, 260, 110, 60, 250)            # cat walks in at 10 s and at 25 s
        if p.process(t, DARK, d).trigger:
            fired.append(t)
    assert fired and all(t >= 25 for t in fired), fired             # 10 s: in warm-up (the cost)
