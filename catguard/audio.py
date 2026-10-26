"""Pre-loaded WAV playback through ALSA (aplay), random rotation, software gain."""
from __future__ import annotations

import logging
import random
import subprocess
import threading
import wave
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


class Player:
    def __init__(self, sounds_dir: Path, device: str = "default", volume: float = 1.0):
        self.device = device
        self.volume = volume
        self.sounds: list[tuple[str, np.ndarray, int, int]] = []
        for p in sorted(Path(sounds_dir).glob("*.wav")):
            with wave.open(str(p)) as w:
                if w.getsampwidth() != 2:
                    log.warning("skipping %s: not 16-bit", p.name)
                    continue
                data = np.frombuffer(w.readframes(w.getnframes()), np.int16)
                self.sounds.append((p.name, data, w.getframerate(), w.getnchannels()))
        if not self.sounds:
            raise RuntimeError(f"no 16-bit WAV files in {sounds_dir}")
        self._last: str | None = None
        self._lock = threading.Lock()
        log.info("loaded sounds: %s", [s[0] for s in self.sounds])

    def pick(self):
        choices = [s for s in self.sounds if s[0] != self._last] or self.sounds
        s = random.choice(choices)
        self._last = s[0]
        return s

    def play(self, block: bool = False) -> str:
        name, data, rate, ch = self.pick()
        pcm = np.clip(data.astype(np.float32) * self.volume, -32768, 32767).astype(np.int16).tobytes()

        def run():
            if not self._lock.acquire(blocking=False):
                return  # already playing
            try:
                subprocess.run(["aplay", "-q", "-D", self.device, "-t", "raw", "-f", "S16_LE",
                                "-r", str(rate), "-c", str(ch)], input=pcm, check=False, timeout=30)
            except Exception:
                log.exception("playback failed")
            finally:
                self._lock.release()

        if block:
            run()
        else:
            threading.Thread(target=run, daemon=True).start()
        return name
