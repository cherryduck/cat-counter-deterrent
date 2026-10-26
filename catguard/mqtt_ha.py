"""Home Assistant integration over MQTT discovery.

Entities: enabled switch, cat-on-counter binary sensor, triggers-today sensor,
last-trigger camera, test-sound & recalibrate buttons, cooldown & volume numbers,
night-mode select. Availability via MQTT last-will.
"""
from __future__ import annotations

import json
import logging
import threading
from datetime import date
from typing import Callable

from .config import MqttCfg

log = logging.getLogger(__name__)

DEVICE = {"identifiers": ["catguard"], "name": "Cat Guard", "manufacturer": "DIY",
          "model": "Pi 5 + Kinect"}


class HaBridge:
    def __init__(self, cfg: MqttCfg, on_command: Callable[[str, str], None]):
        import paho.mqtt.client as mqtt
        self.cfg = cfg
        self.t = cfg.base_topic
        self.on_command = on_command
        self._today = date.today()
        self.triggers_today = 0
        self.c = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="catguard")
        if cfg.username:
            self.c.username_pw_set(cfg.username, cfg.password)
        self.c.will_set(f"{self.t}/availability", "offline", retain=True)
        self.c.on_connect = self._on_connect
        self.c.on_message = self._on_message
        self.c.reconnect_delay_set(1, 60)
        self._state: dict[str, str] = {}
        self._lock = threading.Lock()

    def start(self):
        self.c.connect_async(self.cfg.host, self.cfg.port, keepalive=30)
        self.c.loop_start()

    def stop(self):
        self.c.publish(f"{self.t}/availability", "offline", retain=True)
        self.c.loop_stop()
        self.c.disconnect()

    # ---- discovery -------------------------------------------------------
    def _disc(self, comp: str, oid: str, payload: dict):
        base = {"unique_id": f"catguard_{oid}", "object_id": f"catguard_{oid}",
                "device": DEVICE, "availability_topic": f"{self.t}/availability"}
        base.update(payload)
        self.c.publish(f"{self.cfg.discovery_prefix}/{comp}/catguard/{oid}/config",
                       json.dumps(base), retain=True)

    def _on_connect(self, client, userdata, flags, rc, props=None):
        if rc != 0:
            log.warning("MQTT connect failed: %s", rc)
            return
        log.info("MQTT connected")
        t = self.t
        self._disc("switch", "enabled", {"name": "Enabled", "command_topic": f"{t}/enabled/set",
                                         "state_topic": f"{t}/enabled", "icon": "mdi:cat"})
        self._disc("binary_sensor", "cat", {"name": "Cat on counter", "state_topic": f"{t}/cat",
                                            "device_class": "occupancy"})
        self._disc("sensor", "triggers_today", {"name": "Triggers today",
                                                "state_topic": f"{t}/triggers_today",
                                                "state_class": "total_increasing",
                                                "icon": "mdi:counter"})
        self._disc("sensor", "last_trigger", {"name": "Last trigger",
                                              "state_topic": f"{t}/last_trigger",
                                              "device_class": "timestamp"})
        self._disc("camera", "snapshot", {"name": "Last trigger", "topic": f"{t}/snapshot"})
        self._disc("button", "test_sound", {"name": "Test sound",
                                            "command_topic": f"{t}/test_sound/press",
                                            "icon": "mdi:volume-high"})
        self._disc("button", "recalibrate", {"name": "Recalibrate (counter must be empty)",
                                             "command_topic": f"{t}/recalibrate/press",
                                             "icon": "mdi:target"})
        self._disc("number", "cooldown", {"name": "Cooldown", "command_topic": f"{t}/cooldown/set",
                                          "state_topic": f"{t}/cooldown", "min": 5, "max": 600,
                                          "step": 5, "unit_of_measurement": "s",
                                          "entity_category": "config"})
        self._disc("number", "volume", {"name": "Volume", "command_topic": f"{t}/volume/set",
                                        "state_topic": f"{t}/volume", "min": 0, "max": 100,
                                        "step": 5, "unit_of_measurement": "%",
                                        "entity_category": "config"})
        self._disc("select", "night_mode", {"name": "Night mode",
                                            "command_topic": f"{t}/night_mode/set",
                                            "state_topic": f"{t}/night_mode",
                                            "options": ["depth_only", "off"],
                                            "entity_category": "config"})
        self._disc("switch", "dry_run", {"name": "Dry run (no sound)",
                                         "command_topic": f"{t}/dry_run/set",
                                         "state_topic": f"{t}/dry_run",
                                         "entity_category": "config"})
        for cmd in ("enabled/set", "test_sound/press", "recalibrate/press", "cooldown/set",
                    "volume/set", "night_mode/set", "dry_run/set"):
            client.subscribe(f"{t}/{cmd}")
        # retained 'enabled' etc. come back on subscribe of the state topics -> restore
        for st in ("enabled", "cooldown", "volume", "night_mode", "dry_run"):
            client.subscribe(f"{t}/{st}")
        client.publish(f"{t}/availability", "online", retain=True)
        with self._lock:
            for k, v in self._state.items():
                client.publish(f"{t}/{k}", v, retain=True)

    def _on_message(self, client, userdata, msg):
        sub = msg.topic[len(self.t) + 1:]
        payload = msg.payload.decode(errors="ignore")
        if sub.endswith("/set") or sub.endswith("/press"):
            name = sub.rsplit("/", 1)[0]
        else:
            name = "restore:" + sub   # retained state from previous run
        try:
            self.on_command(name, payload)
        except Exception:
            log.exception("command %s failed", name)

    # ---- state -----------------------------------------------------------
    def state(self, key: str, value, retain: bool = True):
        v = str(value)
        with self._lock:
            if retain:
                self._state[key] = v
        self.c.publish(f"{self.t}/{key}", v, retain=retain)

    def trigger(self, jpg: bytes, iso_ts: str):
        if date.today() != self._today:
            self._today, self.triggers_today = date.today(), 0
        self.triggers_today += 1
        self.state("triggers_today", self.triggers_today)
        self.state("last_trigger", iso_ts)
        self.c.publish(f"{self.t}/snapshot", jpg, retain=True)
