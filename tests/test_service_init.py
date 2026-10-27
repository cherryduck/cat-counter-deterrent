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
