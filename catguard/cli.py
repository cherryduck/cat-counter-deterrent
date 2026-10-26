"""catguard command line.

  catguard run [--dry-run]           live service
  catguard check [--seconds N]       phase-1 check: fps + sample images
  catguard calibrate                 capture empty-counter baseline + snapshot
  catguard roi X,Y X,Y X,Y ...       set counter polygon (pixels on snapshot.jpg)
  catguard record DIR [--seconds N]  record raw RGB+depth session
  catguard replay DIR [--video OUT]  run detector offline on a recorded session
  catguard test-sound                play one sound
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from . import config as config_mod
from .capture import Recorder, open_source


def _cfg(args):
    p = args.config
    if p is None and Path("config.toml").exists():
        p = "config.toml"
    return config_mod.load(p)


def cmd_check(cfg, args):
    from .depth import colorize_depth
    src = open_source(cfg.capture.source)
    out = cfg.path(cfg.storage.calibration_dir)
    out.mkdir(parents=True, exist_ok=True)
    n, t0 = 0, time.monotonic()
    rgb = depth = None
    for _, rgb, depth in src.frames():
        n += 1
        if time.monotonic() - t0 >= args.seconds:
            break
    src.close()
    dt = time.monotonic() - t0
    cv2.imwrite(str(out / "check_rgb.jpg"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(out / "check_depth.png"), colorize_depth(depth))
    valid = np.count_nonzero(depth) / depth.size
    print(f"frames={n} fps={n / dt:.1f} rgb={rgb.shape} depth={depth.shape} "
          f"depth_valid={valid:.0%} depth_range={depth[depth > 0].min()}-{depth.max()}mm")
    print(f"saved {out}/check_rgb.jpg, check_depth.png")


def cmd_calibrate(cfg, args):
    from .app import calibrate_frames
    from .depth import Calibration, colorize_depth
    d = cfg.path(cfg.storage.calibration_dir)
    poly = []
    if (d / "roi.json").exists():
        poly = Calibration.load(d).polygon
    src = open_source(cfg.capture.source)
    print(f"capturing {cfg.depth.baseline_frames} frames - keep the counter empty and still")
    base, rgb = calibrate_frames(src.frames(), cfg.depth.baseline_frames)
    src.close()
    cal = Calibration(base, poly)
    cal.save(d)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(d / "snapshot.jpg"), bgr)
    cv2.imwrite(str(d / "baseline_depth.png"), colorize_depth(base.astype(np.uint16)))
    _grid(bgr, d / "snapshot_grid.jpg")
    print(f"baseline saved; valid={np.count_nonzero(base) / base.size:.0%}; see {d}/snapshot_grid.jpg")


def _grid(bgr, path):
    g = bgr.copy()
    for x in range(0, g.shape[1], 40):
        cv2.line(g, (x, 0), (x, g.shape[0]), (0, 255, 255), 1)
        cv2.putText(g, str(x), (x + 2, 10), cv2.FONT_HERSHEY_PLAIN, 0.7, (0, 255, 255), 1)
    for y in range(0, g.shape[0], 40):
        cv2.line(g, (0, y), (g.shape[1], y), (0, 255, 255), 1)
        cv2.putText(g, str(y), (2, y + 12), cv2.FONT_HERSHEY_PLAIN, 0.7, (0, 255, 255), 1)
    cv2.imwrite(str(path), g)


def cmd_roi(cfg, args):
    from .depth import Calibration
    d = cfg.path(cfg.storage.calibration_dir)
    cal = Calibration.load(d)
    poly = [[int(v) for v in p.split(",")] for p in args.points]
    if len(poly) < 3:
        sys.exit("need at least 3 points")
    cal.set_polygon(poly)
    cal.save(d)
    snap = cv2.imread(str(d / "snapshot.jpg"))
    if snap is not None:
        ov = snap.copy()
        cv2.fillPoly(ov, [np.asarray(poly, np.int32)], (0, 255, 0))
        cv2.imwrite(str(d / "roi_preview.jpg"), cv2.addWeighted(ov, 0.35, snap, 0.65, 0))
    print(f"ROI saved ({len(poly)} points); preview {d}/roi_preview.jpg")


def cmd_record(cfg, args):
    src = open_source(cfg.capture.source)
    rec = Recorder(Path(args.dir))
    period = 1.0 / cfg.capture.fps
    t0 = time.monotonic()
    n = 0
    print(f"recording {args.seconds}s to {args.dir} (Ctrl-C to stop early)")
    try:
        for fr in src.frames():
            ts = time.monotonic()
            rec.add(fr)
            n += 1
            if ts - t0 >= args.seconds:
                break
            time.sleep(max(0.0, period - (time.monotonic() - ts)))
    except KeyboardInterrupt:
        pass
    rec.flush()
    src.close()
    print(f"recorded {n} frames")


def cmd_replay(cfg, args):
    from .app import Pipeline, load_detector
    from .depth import Calibration
    from .storage import annotate
    calib_dir = Path(args.calibration) if args.calibration else cfg.path(cfg.storage.calibration_dir)
    calib = Calibration.load(calib_dir)
    pipe = Pipeline(cfg, calib, None if args.no_model else load_detector(cfg))
    src = open_source(args.dir)
    vw = None
    n = trig = cand_frames = hit_frames = 0
    det_ms = []
    for ts, rgb, depth in src.frames():
        v = pipe.process(ts, rgb, depth)
        n += 1
        cand_frames += bool(v.candidates)
        hit_frames += v.hit
        if v.cat_score is not None:
            det_ms.append(v.timings["detect_ms"])
        if v.trigger:
            trig += 1
            print(f"TRIGGER at frame {n} t={ts:.2f} {v.label()} "
                  f"blobs={[(round(b.area_m2, 3), round(b.height_mm)) for b in v.candidates]}")
        if args.verbose and v.blobs:
            print(n, v.label(), [(round(b.area_m2, 3), round(b.height_mm), round(b.inside_roi, 2))
                                 for b in v.blobs])
        if args.video:
            img = annotate(rgb, v.candidates, f"#{n} {v.label()}", v.cat_box)
            if vw is None:
                vw = cv2.VideoWriter(args.video, cv2.VideoWriter_fourcc(*"mp4v"), cfg.capture.fps,
                                     (img.shape[1], img.shape[0]))
            vw.write(img)
    if vw:
        vw.release()
    print(f"frames={n} candidate_frames={cand_frames} hit_frames={hit_frames} triggers={trig}"
          + (f" detect_ms_avg={np.mean(det_ms):.0f}" if det_ms else ""))


def cmd_test_sound(cfg, args):
    from .audio import Player
    p = Player(cfg.path(cfg.audio.sounds_dir), cfg.audio.device, cfg.audio.volume)
    print("played", p.play(block=True))


def cmd_run(cfg, args):
    from .app import Service
    Service(cfg, dry_run=args.dry_run).run()


def main(argv=None):
    ap = argparse.ArgumentParser(prog="catguard")
    ap.add_argument("-c", "--config")
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("run"); s.add_argument("--dry-run", action="store_true")
    s = sub.add_parser("check"); s.add_argument("--seconds", type=float, default=5)
    sub.add_parser("calibrate")
    s = sub.add_parser("roi"); s.add_argument("points", nargs="+", help="X,Y pairs")
    s = sub.add_parser("record"); s.add_argument("dir"); s.add_argument("--seconds", type=float, default=30)
    s = sub.add_parser("replay"); s.add_argument("dir"); s.add_argument("--video")
    s.add_argument("--calibration"); s.add_argument("--no-model", action="store_true")
    sub.add_parser("test-sound")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = _cfg(args)
    {"run": cmd_run, "check": cmd_check, "calibrate": cmd_calibrate, "roi": cmd_roi,
     "record": cmd_record, "replay": cmd_replay, "test-sound": cmd_test_sound}[args.cmd](cfg, args)


if __name__ == "__main__":
    main()
