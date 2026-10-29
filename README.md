# Cat Guard

**Cat Guard watches the kitchen counter with an Xbox 360 Kinect and plays a sound when the cat jumps up.**

What it does:

- Spots a cat on the counter in daylight, under kitchen lights, and in complete darkness.
- Plays one of your own recordings through the kitchen speaker within about half a second of him landing. It rotates the recordings so he doesn't get used to one.
- Ignores people: you cooking, washing up, putting shopping down, or switching the light off on the way out.
- Sends a notification to your phone with a snapshot, and keeps a short video of every event on the NAS.
- Lets you switch it on and off, mute it, and change the volume and cooldown from Home Assistant.

What it does **not** do:

- Tell one cat from another. Any cat on the counter triggers it.
- Ignore people in the dark. With the light off, the camera can't see people, so someone standing at the counter in the dark could trigger it.
- Trigger while someone is in the kitchen with the light on, unless the camera is confident the cat is there. You can shoo him yourself.
- Work without the Pi. The NAS only stores files and runs Home Assistant.

**Contents:** [Words you'll see](#words-youll-see) · [How it works](#how-it-works) · [Hardware](#hardware) · [Install](#install) · [First run](#first-run-step-by-step) · [Everyday use](#everyday-use) · [Tuning](#tuning-what-to-change-when) · [Troubleshooting](#troubleshooting) · [Command reference](#command-reference) · [Technical reference](#technical-reference)

---

## Words you'll see

| Word | Meaning |
|---|---|
| **Counter outline** | The shape of the worktop, drawn on the camera picture. Only things on the worktop count. |
| **Calibration** | A depth picture of the *empty* counter. Anything sticking up from it is "something on the counter". |
| **Depth** | The Kinect's infrared distance reading for every pixel. Works in the dark; washed out by direct sunlight. |
| **Camera check** | An image-recognition model ([YOLO](https://docs.ultralytics.com/)) that looks for a cat in the colour picture. Lit kitchen only. |
| **Person check** | The same kind of model, looking for people. Lit kitchen only. |
| **Trigger** | Cat Guard has decided the cat is on the counter: it plays a sound, saves a snapshot, and notifies you. |
| **Dry run** | Everything happens except the sound. Use it while testing. |
| **Cooldown** | The minimum time between two sounds (default 45 s). |
| **Near miss** | Something cat-sized sat on the counter for a second without triggering. Saved quietly for tuning, with no notification. |
| **Event** | One trigger or near miss: a snapshot, a 6-second video and a raw recording, saved on the NAS. |
| **Replay** | Running a saved raw recording back through the detector on a PC, to see exactly why it did or didn't trigger. |
| **MQTT / broker** | The messaging system Cat Guard uses to talk to Home Assistant. The broker (Mosquitto) runs on the NAS. |

---

## How it works

```
Kinect ──USB──► Raspberry Pi 5 ──USB audio adapter──► audio combiner ──► kitchen speaker
                    │                                      ▲
                    │                               Amazon Echo (also in)
                    ├─ NFS ──► NAS  catcam/  (snapshots, video, raw recordings)
                    └─ MQTT ─► NAS  Mosquitto ─► Home Assistant ─► phone notification
```

Everything time-critical runs on the Pi, so a slow network can't delay the sound. Every frame (15 per second), it decides whether the cat is on the counter in one of two ways:

1. **Depth (main path, day and night).** Something cat-sized is above the empty counter *and moving by itself*: it walks onto new ground, its shape changes, or it appears out of nowhere when nobody has been near. A bag you put down doesn't move, so it doesn't count. It's ignored after about 4 s and becomes part of the background after about 20 s.
2. **Camera (lit kitchen only).** The camera check sees a cat with its feet on the counter. This covers the area by the window, where sunlight blinds the depth sensor. Where depth *does* have a reading, it must agree that something new is there. This stops shadows and appliances from counting.

Both paths are blocked:

- while the person check sees someone in the kitchen, and for 5 s afterwards (depth path; the camera path needs a confident cat that isn't on the person);
- for 3 s after the light is switched on or off;
- for 5 s after the service starts.

The cat must be seen in 3 of the last 5 frames before it triggers. After a trigger it waits for the cooldown, with at most 20 triggers an hour.

**Speed:** about 0.25–0.5 s from landing to sound when depth sees him, and 0.6–1.3 s when only the camera does (by the window in sunlight).

---

## Hardware

| Part | Notes |
|---|---|
| Raspberry Pi 5 | Raspberry Pi OS (Debian 13 "trixie"), 64-bit. Hostname `CatDetector`, 192.168.1.235. |
| Kinect for Xbox 360 (model 1414) | Needs its **mains power adapter with the USB/power splitter**; Pi USB can't power it. Works from 0.8 m; best at 1.5–3.5 m. Mount it high, looking down at the worktop. |
| USB audio adapter | The Pi 5 has no headphone socket. Plug it **straight into the Pi**: it was silent through a USB hub. Use the green/headphone socket, not the pink/mic one. |
| Audio combiner | Echo and Pi in, one output to the speaker, since the speaker has only one input. |
| Synology DS220+ | Runs Mosquitto and Home Assistant in Container Manager, and shares the `catcam` folder over NFS. 192.168.1.119. |

---

## Install

### 1. NAS: broker, Home Assistant and the `catcam` folder

1. Copy `deploy/nas/` to `/volume1/docker/catguard/` on the NAS.
2. Create the broker login (pick your own password):
   ```sh
   docker run --rm -v /volume1/docker/catguard/mosquitto/config:/mosquitto/config \
     eclipse-mosquitto:2 mosquitto_passwd -c -b /mosquitto/config/passwd catguard 'YOUR_PASSWORD'
   ```
3. In Container Manager → **Project** → Create, point it at that folder and start it. Home Assistant is then at `http://192.168.1.119:8123`.
4. Create a shared folder `catcam`. Under **NFS Permissions**, allow the Pi's IP with read/write.

### 2. Pi: system packages and the Kinect

```sh
sudo apt install libfreenect-dev freenect libusb-1.0-0-dev nfs-common alsa-utils
sudo cp deploy/pi/51-kinect.rules /etc/udev/rules.d/
sudo cp deploy/pi/blacklist-gspca-kinect.conf /etc/modprobe.d/
sudo modprobe -r gspca_kinect; sudo udevadm control --reload; sudo udevadm trigger
```

The blacklist matters: the built-in `gspca_kinect` driver grabs the camera and stops everything else from using it.

Mount the NAS folder (add to `/etc/fstab`, then `sudo mkdir /mnt/catcam && sudo mount -a`):

```
192.168.1.119:/volume1/catcam /mnt/catcam nfs defaults,_netdev,nofail,soft,timeo=50,x-systemd.automount,x-systemd.mount-timeout=10 0 0
```

If the NAS is down, events are saved to `~/catguard-events` instead.

### 3. Pi: Cat Guard itself

```sh
rsync -a --exclude .venv --exclude .secrets ./ catdetector:catguard/      # from this repo
ssh catdetector
cd ~/catguard
python3 -m venv .venv && .venv/bin/pip install -e .
```

**Python Kinect bindings.** No Debian package exists, and neither the 0.5.3 release nor upstream's `setup.py` builds against NumPy 2 / Cython 3. Build them from libfreenect master with a minimal setup script:

```sh
git clone https://github.com/OpenKinect/libfreenect ~/libfreenect-master
cd ~/libfreenect-master/wrappers/python
cat > setup_min.py <<'EOF'
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np
setup(ext_modules=cythonize([Extension("freenect", ["freenect.pyx"],
      libraries=["usb-1.0", "freenect", "freenect_sync"],
      include_dirs=["/usr/include/libusb-1.0", np.get_include()])], language_level=3))
EOF
~/catguard/.venv/bin/pip install "cython>=3"
~/catguard/.venv/bin/python setup_min.py build_ext --inplace
cp freenect*.so ~/catguard/.venv/lib/python3*/site-packages/
```

**Models** (not in git). Export on a PC, not the Pi, then copy them to `~/catguard/models/`:

```sh
uv venv -p 3.12 .venv-export
uv pip install -p .venv-export --index-strategy unsafe-best-match \
  --extra-index-url https://download.pytorch.org/whl/cpu torch torchvision ultralytics onnx onnxslim
.venv-export/bin/yolo export model=yolo11n.pt format=onnx imgsz=640 opset=12 simplify=True   # camera check
.venv-export/bin/yolo export model=yolov8n.pt format=onnx imgsz=320 opset=12 simplify=True   # person check
mv yolo11n.onnx models/yolo11n_640.onnx; mv yolov8n.onnx models/yolov8n.onnx
```

**Sounds** (not in git): put **16-bit WAV** files in `~/catguard/sounds/`. Other formats are skipped with a warning.

**Configuration:**

```sh
cp config.example.toml config.toml
```

At minimum, set `[audio] device` (find the adapter with `aplay -l`; here it's `plughw:CARD=Device,DEV=0`) and `[mqtt] password`.

**Audio adapter volume.** Cheap USB adapters start at about 29%. Set the volume to maximum and save it, so it's restored after a reboot or replug:

```sh
amixer -c Device sset Speaker 151     # 151 = 100% on this adapter (check "Limits" in amixer -c Device sget Speaker)
sudo alsactl store
```

**Service:**

```sh
sudo cp deploy/pi/catguard.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now catguard
```

### Updating

Edit the code on the PC, run the tests, then copy it across and restart:

```sh
.venv/bin/pytest -q          # check for "failed" BEFORE deploying
rsync -a --exclude .venv --exclude .secrets --exclude __pycache__ --exclude .git \
  --exclude calibration --exclude config.toml --exclude deploy/nas ./ catdetector:catguard/
ssh catdetector 'sudo systemctl restart catguard'
```

Don't deploy straight after `pytest | tail`: a pipeline reports `tail`'s exit code, so a failing test run still "succeeds".

---

## First run, step by step

1. **Check the Kinect works.** On the Pi: `.venv/bin/catguard -c config.toml check`. Expect about 27 fps and a sensible depth range. Stop the service first (`sudo systemctl stop catguard`), because only one program can use the Kinect at a time.
2. **Aim it.** With the service running, open **http://192.168.1.235:8080** on a phone or PC on your home Wi-Fi. Pick **Camera** view. Use **▼ Down / ▲ Up**, or move the Kinect by hand, until the whole worktop is in view and as large as possible.
3. **Check sunlight.** Switch to **Above counter**. Dimmed areas have no depth reading. Big holes by the window in daytime are expected; that's what the camera check covers.
4. **Calibrate.** Clear the counter, step out of view and press **Recalibrate (counter empty!)**. The red "detection paused" banner disappears.
5. **Draw the outline.** Press **Draw counter outline**, tap the worktop's corners on the picture, then **Save outline**. Use **Undo** and **Clear** to fix mistakes.
6. **Test the sound.** Press **Test sound** and check it reaches the speaker.
7. **Connect Home Assistant.** Settings → Devices & services → Add integration → **MQTT**. Enter broker `192.168.1.119`, port `1883`, user `catguard` and the password. A **Cat Guard** device appears on its own.
8. **Run silently for a few days.** Leave **Dry run** on. Check each notification: was it the cat? Then turn **Dry run** off.

> **Moved the Kinect?** Recalibrate and redraw the outline. Using the tilt buttons marks the calibration as out of date and pauses detection until you recalibrate. The tilt motor only moves when you press a button, never on its own.

---

## Everyday use

### Home Assistant

The **Cat Guard** device (Settings → Devices & services → MQTT → Cat Guard) has:

| Entity | What it's for |
|---|---|
| **Enabled** (switch) | Turns detection off completely. |
| **Dry run (no sound)** (switch) | Detects, saves and notifies, but stays silent. |
| **Cat on counter** | On while the cat is detected. |
| **Triggers today** | Count since midnight. It survives restarts. |
| **Last trigger** | Time of the last trigger. |
| **Last snapshot** (camera) | Picture from the last trigger, with a box round the cat. |
| **Volume** | 0–100%, Cat Guard's own volume. The adapter's hardware volume is separate; see [Install](#3-pi-cat-guard-itself). |
| **Cooldown** | 5–600 s between sounds. |
| **Night mode** | `depth_only` keeps detecting in the dark. `off` stops detecting while the kitchen is dark. |
| **Test sound** / **Recalibrate (counter must be empty)** | Same as the buttons on the aiming page. |

Settings changed here are remembered by the broker and come back after the Pi restarts.

There's also a **Cat Guard** dashboard in the sidebar, and three automations live in Home Assistant itself, not in this repo:

- **Cat Guard – cat on counter:** sends a phone notification with the snapshot. Tapping it opens the snapshot. It has a **Pause 1 hour** button.
- **Cat Guard – pause from notification:** turns **Enabled** off for an hour when you press that button.
- **Cat Guard – offline alert:** notifies you if the Pi has been unreachable for 10 minutes.

### The `catcam` folder on the NAS

One folder per day, with one set of files per event named by time:

| File | What it is | Kept for |
|---|---|---|
| `HHMMSS_dry.jpg` / `_cat.jpg` | Snapshot of a trigger, in dry run / with sound | 30 days |
| `HHMMSS_*.mp4` | 3 s before and 3 s after, with boxes drawn on | 30 days |
| `HHMMSS_cand.*` | A near miss, with no notification | 30 days |
| `HHMMSS_*.raw.npz` | Raw colour + depth + the background at the time, for replay (10–30 MB) | 7 days |

### The aiming page (http://192.168.1.235:8080)

It shows a live view (**Camera**, **Depth** or **Above counter**), plus tilt, outline drawing, recalibration and a test sound, with a status line (fps, cat present, depth coverage). It has **no login**, so anyone on your home network can use it. It's only encoded while someone has it open, so it costs nothing otherwise.

---

## Tuning: what to change when

| Symptom | What to do |
|---|---|
| The cat ignores the sound | Add more, different recordings to `sounds/`. Raise the speaker volume. Lengthen the cooldown so it's less routine. |
| Too many plays while he sits there | Raise **Cooldown** in Home Assistant. |
| False triggers at night | Set **Night mode** to `off`, or send me the event time to replay. |
| A trigger shows a person or object | Note the time and replay the event (see [Troubleshooting](#troubleshooting)) before changing anything. |
| Missed cat | Look for a `_cand` near miss at that time and replay it. |
| Detection paused banner | The view moved or tilt was used: recalibrate with the counter empty. |
| Dusk lands on the wrong side of "dark" | Change `detector.min_brightness` (default 40 out of 255). |

All settings are in `config.toml`; see the [technical reference](#all-settings) for the full list. Restart the service after editing it.

---

## Troubleshooting

### It triggered on the wrong thing, or missed the cat

Don't guess: replay the event. Copy the raw recording to the PC and run it through the current code, starting from the background the Pi had at the time:

```sh
scp catdetector:/mnt/catcam/2026-10-29/124321_dry.raw.npz .
.venv/bin/catguard -c pi-config/config.toml replay 124321_dry.raw.npz --live-baseline
```

`pi-config/` needs a copy of the Pi's `config.toml`, `calibration/` and `models/`. The output lists every trigger with the reason (`depth:new ground 0.17`, `depth:arrived`, `camera`, …) and blob sizes. Change a setting or the code, then replay again until it does the right thing. Re-check all the other saved events before deploying.

On the Pi, a replay takes about 80 s per event and competes with the live service; on a PC it takes about 2 s.

Things that make old recordings misleading:

- **A recalibration since the recording.** Check with `journalctl -u catguard | grep -iE "recalib|tilt"`. Recordings from before it don't match the new calibration.
- **Recordings without `--live-baseline`** start from the calibration, not the background the Pi had learnt, so they can't reproduce background-learning problems.

### Other problems

| Problem | Check |
|---|---|
| No sound | `catguard test-sound`; adapter plugged straight into the Pi (not a hub); green socket; `amixer -c Device sget Speaker` at 100%; **Dry run** off; combiner wired Echo + Pi **in**, speaker **out**. |
| Service keeps restarting | `journalctl -u catguard -n 50`; `systemctl show catguard -p NRestarts`. |
| Kinect not found | Power adapter connected; `lsusb` shows `045e:02ae`; `gspca_kinect` not loaded (`lsmod`); nothing else using it (stop the service before `check`/`record`). |
| Home Assistant shows "Unavailable" | Pi off, or broker down: `docker ps` on the NAS. |
| No files in `catcam` | `mount | grep catcam` on the Pi; otherwise look in `~/catguard-events`. A test notification sent from Home Assistant doesn't create files; only the Pi does. |
| Frame rate drops below 15 | The `stats:` line in the journal every 30 s shows fps, ms per frame and scan time. |

Useful log lines:

```sh
journalctl -u catguard -f | grep -vE "data bytes"         # live
journalctl -u catguard --since today | grep -E "TRIGGER|near miss"
```

---

## Command reference

Run as `catguard -c config.toml <command>` (`-v` for more logging). Stop the service first for the commands that use the Kinect: `check`, `calibrate` and `record`.

| Command | What it does |
|---|---|
| `run [--dry-run]` | The live service (what systemd runs). |
| `check [--seconds N]` | Measures fps, saves `calibration/check_rgb.jpg` and `check_depth.png`. |
| `calibrate` | Captures the empty-counter calibration (keeps the existing outline). |
| `roi X,Y X,Y X,Y …` | Sets the counter outline from the command line; writes `roi_preview.jpg`. |
| `record DIR [--seconds N]` | Records raw colour + depth (default 30 s) for later replay. |
| `replay PATH [options]` | Runs a recording (`DIR` or `*.raw.npz`) through the detector and lists triggers. |
| `test-sound` | Plays one recording. |

`replay` options:

| Option | Meaning |
|---|---|
| `--live-baseline` | For `*.raw.npz` events: start from the background the service had at the time. |
| `--no-model` | Depth only: skip the camera and person checks (fast). |
| `--warmup S` | Simulate the 5 s start-up pause. |
| `--calibration DIR` | Use a different calibration. |
| `--video FILE` | Write an annotated video. |

Tests: `.venv/bin/pytest -q` (no Kinect needed).

---

## Technical reference

### Code layout

| File | Contents |
|---|---|
| `catguard/capture.py` | Kinect via libfreenect (frames copied out of its buffers), tilt motor via ctypes, `ReplaySource` / `Recorder`. |
| `catguard/depth.py` | Calibration, temporal median, height map, blobs, adaptive background, `MotionGate`. |
| `catguard/detector.py` | YOLO ONNX wrapper: tiled counter scan, person check. |
| `catguard/scanner.py` | Background thread running the camera scan + person check on the latest frame. |
| `catguard/app.py` | `Pipeline` (per-frame decision), `Service` (live loop), depth vetoes for camera hits. |
| `catguard/decision.py` | Persistence, cooldown, hourly cap. |
| `catguard/storage.py` | Snapshots, clips, raw events, retention. |
| `catguard/mqtt_ha.py` | MQTT discovery, state, commands, persisted daily count. |
| `catguard/web.py` | Aiming page (MJPEG stream + API). |
| `catguard/audio.py` | WAV playback through `aplay`, random rotation (never the same twice in a row), software gain. |

### Depth path in detail

- **Calibration:** per-pixel median of 60 empty frames (0 = no reading is ignored), plus a per-pixel noise estimate.
- **Filtering:** median of the last 3 frames per pixel; a pixel needs readings in 2 of 3. This removes the flicker shiny or dark objects cause (frying pan, knife block: 93 false blobs in 150 empty frames → 1 in 322).
- **"Above the counter" threshold**, per pixel: the largest of 60 mm, 4 × measured noise, and 3 Kinect depth steps. The Kinect measures disparity, so depth comes in steps of about 2.85×10⁻⁶ · z² mm: about 24 mm at 2.9 m and 40 mm at 3.7 m.
- **Blobs:** connected regions above the threshold, with no upper height limit at this stage. That way a person leaning in forms one region that spills outside the outline and is rejected.
- **Cat-sized:** 0.015–0.30 m² top-down (pixels × (z / 525)²), at most 600 mm tall, at least 85% inside the outline.
- **Person-like (depth):** bigger than a cat, or ≥ 0.08 m² and touching the bottom of the picture. In practice, people near the Kinect give no depth, so this rarely fires; the person check does the real work.
- **Adaptive background:** each pixel tracks how often it's above or below the background (time constant 8 s, updated every 3rd frame).
  - Over 50% of the time, up to cat height: treated as background ("scenery"). A kettle left out is ignored after about 4 s.
  - Over 90%: the background itself is updated (about 20 s).
  - It may only drift up to max(150 mm, 5 depth steps) from the calibration. Lighting drift at dusk is 1–4 steps; a person is at least 0.5 m nearer and can never be absorbed.
  - Pixels under a person-like blob are frozen.
- **Moving by itself** (any one, compared with 5 frames ago, ~0.33 s):
  - **New ground:** ≥ 12% of the blob lies outside everywhere a candidate has been in the last 3 s (0.4 s delay so the blob's own last frames don't count).
  - **Interior change:** ≥ 20% of the blob's inner pixels (edges eroded) changed by > 40 mm.
  - Needs 3 such readings within 1 s; then counts as moving for 2 s.
  - **Arrived:** a new blob that persists for 4 frames, at a spot with no candidate in the last 30 s, with no person-like blob in the last 2 s. This is the cat jumping up.
  - Measured margins: a still bag reaches ≤ 5% interior change and ≤ 2.2 px shift; the cat moving reached 0.52–0.84.

### Camera path in detail

- Model `yolo11n` at 640, on 320-px tiles across the outline's bounding box (each upscaled 2×), in a background thread (~570 ms per scan). The cat's box must have its bottom centre (its feet) inside the outline, grown by 25 px.
- Needs 2 positive scans in a row, or one ≥ 0.6. Results older than 1 s are discarded, and the snapshot is the exact frame the scan ran on.
- **Vetoes** where depth has a reading:
  - ≥ 30% of the box is more than 600 mm above the counter, or a person-like blob overlaps it: it's a person (hair scored cat = 0.62).
  - Neither a depth candidate overlapping the box in the last 10 s, nor ≥ 35% of the box > 80 mm above the calibrated counter: it's a static object (coffee machine in shadow: 0.18–0.28; the cat: 0.46–0.59).
  - Where depth has no reading (sunlight), the camera alone decides.
- **Dark:** mean brightness inside the outline < 40, judged on the counter area, not the whole frame, because the window skews it. In the dark the camera and person checks are skipped.

### Person check

- `yolov8n` at 320 on the whole frame, after each camera scan (~65 ms).
- Measured scores: people 0.76–0.93; the cat ≤ 0.20. Gate at 0.4.
- When it sees a person: no depth triggers for 5 s, and camera triggers need ≥ 0.6 and < 30% overlap with a person box.

### Measured performance (Pi 5)

- Kinect: 27 fps colour + depth; the service processes 15 fps at about 23–28 ms per frame.
- Depth noise about 20 mm at 2–3 m; distance to the counter 2.9–3.7 m.
- From this position the cat is only about 80 px long. The small 320 model missed him; tiling with the 640 model fixed it. Mounting the Kinect closer would help more than a bigger model.
- Sunlight blinds the depth sensor near the window in daytime, and biases it everywhere: after dusk the counter reads 4–17 cm nearer than a sunny-afternoon calibration until the background adapts.
- Validation, 36 saved events (27–29 Oct): all 7 false alarms (people, coffee machine) → 0 triggers; all 12 cat triggers and the 05:33 night event still trigger.

### MQTT topics

Base topic `catguard/`. Home Assistant discovery under `homeassistant/`; device availability on `catguard/availability` (with a last-will of `offline`).

| State topic | Command topic |
|---|---|
| `enabled`, `dry_run` (`ON`/`OFF`) | `enabled/set`, `dry_run/set` |
| `cooldown` (s), `volume` (%) | `cooldown/set`, `volume/set` |
| `night_mode` (`depth_only`/`off`) | `night_mode/set` |
| `cat`, `triggers_today`, `last_trigger` (ISO time), `snapshot` (JPEG, sent before `last_trigger`) | — |
| — | `test_sound/press`, `recalibrate/press` |

State is retained, and the service restores `enabled`, `dry_run`, `cooldown`, `volume` and `night_mode` from it at start-up. The daily count is kept in `calibration/triggers_today.json`.

### Aiming page API

`GET /` page · `GET /stream?view=rgb|depth|height` MJPEG · `GET /snapshot.jpg?view=…` · `GET /api/state` · `POST /api/tilt`, `/api/roi`, `/api/recalibrate`, `/api/test_sound`.

### All settings

Defaults from `catguard/config.py`. `config.toml` only needs the ones you change; unknown keys are an error.

| Section | Key | Default | Meaning |
|---|---|---|---|
| `capture` | `source` | `kinect` | Or a recorded session dir. |
| | `fps` | 15 | Processing rate. |
| `depth` | `min_height_mm` / `max_height_mm` | 60 / 600 | Height range of a cat above the counter. |
| | `min_area_m2` / `max_area_m2` | 0.015 / 0.30 | Top-down size of a cat. |
| | `min_inside_roi` | 0.85 | Share of a blob that must be inside the outline. |
| | `noise_k`, `quant_k` | 4, 3 | Threshold: × measured noise, × depth steps. |
| | `temporal_frames` | 3 | Frames in the per-pixel median. |
| | `warmup_s` | 5 | No depth triggers after start-up. |
| | `adapt_tau_s`, `adapt_every` | 8, 3 | Background learning speed; 0 turns it off. |
| | `scenery_occ`, `absorb_occ` | 0.5, 0.9 | Ignore / absorb thresholds. |
| | `adapt_max_mm`, `adapt_max_steps` | 150, 5 | How far the background may drift from calibration. |
| | `motion_lag`, `motion_mm`, `motion_interior_frac` | 5, 40, 0.2 | Movement test. |
| | `motion_shift_px`, `motion_iou_min` | 5, 0.7 | Present in code; the live gate uses new ground + interior change. |
| | `motion_hold_s` | 2 | Counts as moving this long after the last movement. |
| | `person_window_s`, `person_min_area_m2` | 2, 0.08 | Arrival rule; depth person size. |
| | `baseline_frames` | 60 | Frames used to calibrate. |
| | `absorb_after_s`, `static_tolerance_mm` | 180, 25 | Old absorb mechanism, **no longer used** (replaced by the adaptive background). |
| `detector` | `model`, `imgsz`, `scan_tile` | `models/yolo11n_640.onnx`, 640, 320 | Camera check model and tiling. |
| | `cat_conf`, `scan_confirm`, `scan_strong_conf` | 0.35, 2, 0.6 | Camera thresholds. |
| | `scan_every`, `scan_threads` | 2, 3 | Scan frequency (0 = off), CPU threads. |
| | `roi_feet_margin_px` | 25 | Outline growth for the feet test. |
| | `min_brightness`, `night_mode` | 40, `depth_only` | Dark threshold; behaviour in the dark. |
| | `person_model`, `person_imgsz`, `person_gate_conf`, `person_recent_s` | `models/yolov8n.onnx`, 320, 0.4, 5 | Person check. |
| | `person_conf` | 0.35 | Ignore cat boxes mostly inside a person box. |
| | `camera_depth_recent_s`, `camera_new_frac` | 10, 0.35 | Depth agreement for camera hits. |
| | `light_jump`, `light_settle_s` | 25, 3 | Light-switch detection. |
| | `confirm_model`, `confirm_imgsz`, `crop_margin` | — | Crop confirmation, **not used** by the live pipeline. |
| `decision` | `persist_hits` / `persist_window` | 3 / 5 | Frames needed. |
| | `cooldown_s`, `max_per_hour` | 45, 20 | Rate limits. |
| `audio` | `device`, `sounds_dir`, `volume` | `default`, `sounds`, 1.0 | Output and software gain. |
| `mqtt` | `enabled`, `host`, `port`, `username`, `password`, `base_topic`, `discovery_prefix` | | Broker. |
| `storage` | `events_dir`, `fallback_dir`, `calibration_dir` | `/mnt/catcam`, `~/catguard-events`, `calibration` | Where files go. |
| | `retention_days`, `raw_retention_days` | 30, 7 | Clean-up. |
| | `pre_seconds`, `post_seconds` | 3, 3 | Clip length. |
| | `save_raw`, `save_near_misses` | true, true | Raw recordings; near misses. |
| `web` | `enabled`, `port` | true, 8080 | Aiming page. |

### Kinect quirks

- `sync_get_depth/video` return views onto libfreenect's internal buffers: copy them before keeping frames.
- Never set the tilt at start-up. It silently changes the view and invalidates the calibration. The service only reads the angle; moving it marks the calibration stale.
- One program at a time: stop the service before `check`, `calibrate` or `record`.
