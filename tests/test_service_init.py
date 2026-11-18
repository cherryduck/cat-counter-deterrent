"""Smoke test: the live Service object can be constructed (catches init-order bugs)."""
import numpy as np
from catguard.config import Config


def test_service_constructs(tmp_path, monkeypatch):
    import catguard.app as A
    from catguard.depth import Calibration
    cfg = Config()
    cfg.storage.calibration_dir = str(tmp_path / "cal")
    cfg.storage.events_dir = str(tmp_path / "ev")
    cfg.storage.fallback_dir = str(tmp_path / "fb")
    cfg.mqtt.enabled = False
    Calibration(np.full((480, 640), 2000, np.float32), [[0, 0], [10, 0], [10, 10]]).save(tmp_path / "cal")
    monkeypatch.setattr(A, "load_detector", lambda *a, **k: None)
    svc = A.Service(cfg, dry_run=True)
    assert svc.store.baseline_fn() is svc.calib.baseline


def test_service_restores_learnt_background_and_picks_warmup(tmp_path, monkeypatch):
    import catguard.app as A
    from catguard.depth import Calibration
    cfg = Config()
    cfg.storage.calibration_dir = str(tmp_path / "cal")
    cfg.storage.events_dir = str(tmp_path / "ev")
    cfg.storage.fallback_dir = str(tmp_path / "fb")
    cfg.mqtt.enabled = False
    cfg.web.enabled = False
    Calibration(np.full((480, 640), 2000, np.float32), [[0, 0], [600, 0], [600, 400]]).save(tmp_path / "cal")
    monkeypatch.setattr(A, "load_detector", lambda *a, **k: None)
    s1 = A.Service(cfg, dry_run=True)
    assert s1.pipe.warmup_s == cfg.depth.cold_warmup_s                     # nothing saved yet
    d = np.full((480, 640), 2000, np.uint16); d[100:150, 100:160] = 1700  # object left out
    for i in range(15 * 60):
        s1.pipe.process(i / 15, np.zeros((480, 640, 3), np.uint8), d)
    assert s1.save_state(wait=True)
    s2 = A.Service(cfg, dry_run=True)
    assert s2.pipe.warmup_s == cfg.depth.warmup_s                          # restored
    assert np.array_equal(s2.pipe.depth.scenery, s1.pipe.depth.scenery)
    assert np.array_equal(s2.calib.baseline, s1.calib.baseline)
    assert np.array_equal(s2.pipe.depth.reference, np.full((480, 640), 2000, np.float32))
