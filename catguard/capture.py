"""Frame sources: live Kinect (libfreenect sync API) and recorded sessions.

A frame is (timestamp, rgb uint8 HxWx3, depth uint16 mm HxW registered to rgb).
Recorded sessions are directories of compressed .npz chunks.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Iterator

import numpy as np

Frame = tuple[float, np.ndarray, np.ndarray]


class KinectSource:
    def __init__(self, index: int = 0):
        import freenect  # type: ignore
        self.fn = freenect
        self.index = index
        self.depth_fmt = getattr(freenect, "DEPTH_REGISTERED", None)
        if self.depth_fmt is None:
            raise RuntimeError("freenect wrapper lacks DEPTH_REGISTERED")

    def frames(self) -> Iterator[Frame]:
        fn = self.fn
        fails = 0
        while True:
            d = fn.sync_get_depth(self.index, self.depth_fmt)
            v = fn.sync_get_video(self.index, fn.VIDEO_RGB)
            if d is None or v is None:
                fails += 1
                if fails > 50:
                    raise RuntimeError("Kinect not responding")
                time.sleep(0.1)
                continue
            fails = 0
            yield time.time(), v[0], d[0].astype(np.uint16)

    def close(self):
        try:
            self.fn.sync_stop()
        except Exception:
            pass


class ReplaySource:
    def __init__(self, path: Path, realtime: bool = False):
        self.files = sorted(Path(path).glob("chunk_*.npz"))
        if not self.files:
            raise FileNotFoundError(f"no chunk_*.npz in {path}")
        self.realtime = realtime

    def frames(self) -> Iterator[Frame]:
        prev = None
        for f in self.files:
            z = np.load(f)
            for ts, rgb, depth in zip(z["ts"], z["rgb"], z["depth"]):
                if self.realtime and prev is not None:
                    time.sleep(max(0.0, min(1.0, ts - prev)))
                prev = ts
                yield float(ts), rgb, depth

    def close(self):
        pass


class Recorder:
    def __init__(self, out: Path, chunk: int = 50):
        self.out = Path(out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.chunk = chunk
        self.buf: list[Frame] = []
        self.n = 0

    def add(self, fr: Frame):
        self.buf.append(fr)
        if len(self.buf) >= self.chunk:
            self.flush()

    def flush(self):
        if not self.buf:
            return
        ts, rgb, depth = zip(*self.buf)
        np.savez_compressed(self.out / f"chunk_{self.n:05d}.npz",
                            ts=np.array(ts), rgb=np.stack(rgb), depth=np.stack(depth))
        self.n += 1
        self.buf = []


def open_source(spec: str, realtime: bool = False):
    if spec == "kinect":
        return KinectSource()
    return ReplaySource(Path(spec).expanduser(), realtime=realtime)
