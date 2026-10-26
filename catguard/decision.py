"""Turn per-frame verdicts into triggers: persistence, cooldown, rate limit."""
from __future__ import annotations

import time
from collections import deque

from .config import DecisionCfg


class Decider:
    def __init__(self, cfg: DecisionCfg):
        self.cfg = cfg
        self.window: deque[bool] = deque(maxlen=cfg.persist_window)
        self.last_trigger = -1e18
        self.history: deque[float] = deque()

    @property
    def present(self) -> bool:
        return sum(self.window) >= self.cfg.persist_hits

    def update(self, hit: bool, now: float | None = None) -> bool:
        """Feed one frame verdict; return True if the deterrent should fire now."""
        now = time.monotonic() if now is None else now
        self.window.append(hit)
        while self.history and now - self.history[0] > 3600:
            self.history.popleft()
        if not self.present:
            return False
        if now - self.last_trigger < self.cfg.cooldown_s:
            return False
        if len(self.history) >= self.cfg.max_per_hour:
            return False
        self.last_trigger = now
        self.history.append(now)
        return True
