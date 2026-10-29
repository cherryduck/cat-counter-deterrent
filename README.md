# Cat Guard

**Cat Guard watches a kitchen counter (or any worktop, table or shelf) with an Xbox 360 Kinect, and plays a sound when your cat jumps up.**

What it does:

- Spots a cat on the counter in daylight, under room lights, and in complete darkness.
- Plays one of your own recordings within about half a second of the cat landing. It rotates between recordings so the cat doesn't get used to one.
- Ignores people: cooking, washing up, putting shopping down, switching the light off on the way out.
- Saves a snapshot and a short video of every event and, if you use Home Assistant, sends a phone notification.
- Can be switched on and off, muted and adjusted from Home Assistant.

What it does **not** do:

- Tell one cat from another. Any cat on the counter triggers it.
- Ignore people in the dark. With the light off the camera can't see people, so someone standing at the counter in the dark could trigger it.
- Trigger while someone is in the room with the light on, unless the camera is confident the cat is there.
- Run without the Pi. Everything else (network storage, Home Assistant) is optional.

**Contents:** [Words you'll see](#words-youll-see) · [How it works](#how-it-works) · [What you need](#what-you-need) · [Install](#install) · [First run](#first-run-step-by-step) · [Home Assistant](#home-assistant-optional) · [Everyday use](#everyday-use) · [Tuning](#tuning-what-to-change-when) · [Troubleshooting](#troubleshooting) · [Command reference](#command-reference) · [Technical reference](#technical-reference)

---

## Words you'll see

| Word | Meaning |
|---|---|
| **Counter outline** | The shape of the worktop, drawn on the camera picture. Only things on the worktop count. |
| **Calibration** | A depth picture of the *empty* counter. Anything sticking up from it is "something on the counter". |
| **Depth** | The Kinect's infrared distance reading for every pixel. Works in the dark; washed out by direct sunlight. |
| **Camera check** | An image-recognition model ([YOLO](https://docs.ultralytics.com/)) that looks for a cat in the colour picture. Lit room only. |
| **Person check** | The same kind of model, looking for people. Lit room only. |
| **Trigger** | Cat Guard has decided the cat is on the counter: it plays a sound, saves a snapshot and tells Home Assistant. |
| **Dry run** | Everything happens except the sound. Use it while testing. |
| **Cooldown** | The minimum time between two sounds (default 45 s). |
| **Near miss** | Something cat-sized sat on the counter for a second without triggering. Saved quietly for tuning. |
| **Event** | One trigger or near miss: a snapshot, a 6-second video and a raw recording. |
| **Replay** | Running a saved raw recording back through the detector, to see exactly why it did or didn't trigger. |
| **MQTT / broker** | The messaging system Cat Guard uses to talk to Home Assistant. The broker is usually Mosquitto. |

---

## How it works

```
Kinect ──USB──► Raspberry Pi 5 ──audio out──► speaker
                    │
                    ├─ (optional) network share ──► snapshots, videos, raw recordings
                    └─ (optional) MQTT ──► Home Assistant ──► phone notification
```

Everything time-critical runs on the Pi, so a slow network can't delay the sound. Fifteen times a second, it decides whether the cat is on the counter in one of two ways:

1. **Depth (main path, day and night).** Something cat-sized is above the empty counter *and moving by itself*: it walks onto new ground, its shape changes, or it appears when nobody has been near. A bag you put down doesn't move, so it doesn't count. It's ignored after about 4 s and becomes part of the background after about 20 s.
2. **Camera (lit room only).** The camera check sees a cat with its feet on the counter. This covers places where sunlight blinds the depth sensor, such as by a window. Where depth *does* have a reading, it must agree that something new is there. This stops shadows and appliances from counting.

Both paths are blocked:

- while the person check sees someone, and for 5 s afterwards (the camera path still works for a confident cat that isn't on the person);
- for 3 s after the light is switched on or off;
- for 5 s after the service starts.

The cat must be seen in 3 of the last 5 frames before it triggers. After a trigger it waits for the cooldown, with at most 20 triggers an hour.

**Speed:** about 0.25–0.5 s from landing to sound when depth sees the cat, and 0.6–1.3 s when only the camera does.

---

## What you need

| Part | Notes |
|---|---|
| **Raspberry Pi 5** | 64-bit Raspberry Pi OS (tested on the Debian 13 "trixie" release). A Pi 4 is untested; the image models need the Pi 5's speed. |
| **Kinect for Xbox 360** (model 1414) | Needs the **Kinect power adapter** (mains brick with a USB/power splitter), cheap second-hand; the Pi can't power it. The Xbox One Kinect is **not** supported. |
| **Speaker** | The Pi 5 has no headphone socket, so use a cheap USB audio adapter, HDMI audio or a Bluetooth speaker. If your speaker's only input is already taken (for example by a smart speaker), a small audio combiner lets both share it. |
| **Sound recordings** | A few short WAV files: a clap, a hiss, your voice saying "off!". Several different ones work better than one. |
| *Optional:* network storage | Any NFS share (NAS, another PC) to keep events off the Pi's SD card. Without one, events go to `~/catguard-events` on the Pi. |
| *Optional:* Home Assistant and an MQTT broker | For notifications and remote control. Both can run on a NAS, a spare PC or a Home Assistant box. |

**Placing the Kinect:** mount it high, looking down at the counter from about 1.5–3.5 m away; it can't see closer than 0.8 m. The closer it is, the bigger the cat looks and the more reliable the camera check. At about 3 m a cat is only about 80 pixels long, which is near the limit. Avoid pointing it straight at a sunny window.

---

## Install

In the commands below, replace:

| Placeholder | With |
|---|---|
| `<pi>` | Your Pi's hostname or IP address, e.g. `catguard.local` |
| `<user>` | Your user name on the Pi |
| `<share>` | Your NFS share, e.g. `192.168.1.10:/volume1/catcam` |

### 1. System packages and the Kinect (on the Pi)

```sh
sudo apt install git python3-venv libfreenect-dev freenect libusb-1.0-0-dev alsa-utils
git clone <this repo> ~/catguard && cd ~/catguard
sudo cp deploy/pi/51-kinect.rules /etc/udev/rules.d/
sudo cp deploy/pi/blacklist-gspca-kinect.conf /etc/modprobe.d/
sudo modprobe -r gspca_kinect; sudo udevadm control --reload; sudo udevadm trigger
```

The blacklist matters: Linux's built-in `gspca_kinect` driver grabs the camera and stops anything else from using it.

### 2. Cat Guard and the Kinect Python bindings

```sh
cd ~/catguard
python3 -m venv .venv && .venv/bin/pip install -e .
```

There's no packaged Python module for the Kinect, and the released bindings don't build against current NumPy and Cython. Build them from libfreenect's source with a minimal setup script:

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

### 3. Models

The two image models aren't included. Export them on a PC, because it's slow on a Pi, then copy them across:

```sh
uv venv -p 3.12 .venv-export
uv pip install -p .venv-export --index-strategy unsafe-best-match \
  --extra-index-url https://download.pytorch.org/whl/cpu torch torchvision ultralytics onnx onnxslim
.venv-export/bin/yolo export model=yolo11n.pt format=onnx imgsz=640 opset=12 simplify=True   # camera check
.venv-export/bin/yolo export model=yolov8n.pt format=onnx imgsz=320 opset=12 simplify=True   # person check
mv yolo11n.onnx yolo11n_640.onnx
ssh <pi> mkdir -p catguard/models
scp yolo11n_640.onnx yolov8n.onnx <pi>:catguard/models/
```

Without the models, Cat Guard still runs on depth alone. You lose the camera check and the person check, so expect false alarms when people are around.

### 4. Sounds

Put **16-bit WAV** files in `~/catguard/sounds/` (create the folder). Other formats are skipped with a warning. To convert, run `ffmpeg -i clap.m4a -ac 1 -ar 44100 -sample_fmt s16 clap.wav`.

### 5. Configuration

```sh
cp config.example.toml config.toml
```

Settings to change:

| Setting | What to put |
|---|---|
| `[audio] device` | Your audio output. Run `aplay -l`; for a USB adapter listed as `card 2: Device` use `plughw:CARD=Device,DEV=0`. |
| `[mqtt] host`, `username`, `password` | Your broker. If you don't use Home Assistant, set `enabled = false` instead. |
| `[storage] events_dir` | Where events go: a network share's mount point or any local folder. |

Cheap USB audio adapters often start at about 30% volume. Turn it up and save it so it survives reboots:

```sh
alsamixer        # F6 to pick the adapter, raise the level, Esc
sudo alsactl store
```

### 6. Optional: network share for events

For an NFS share, add a line like this to `/etc/fstab`:

```
<share> /mnt/catcam nfs defaults,_netdev,nofail,soft,timeo=50,x-systemd.automount,x-systemd.mount-timeout=10 0 0
```

Then run `sudo apt install nfs-common && sudo mkdir -p /mnt/catcam && sudo mount -a`. The share must let the Pi write. If it's unavailable, events go to `[storage] fallback_dir` (`~/catguard-events`) instead.

### 7. Run it as a service

`deploy/pi/catguard.service` contains the original author's user name and paths. Edit `User=`, `WorkingDirectory=` and `ExecStart=` to match yours (`/home/<user>/catguard`), then:

```sh
sudo cp deploy/pi/catguard.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now catguard
```

### Updating

```sh
cd ~/catguard && git pull && .venv/bin/pip install -e . && sudo systemctl restart catguard
```

If you change the code, run `.venv/bin/pytest -q` and check nothing failed before restarting. The tests don't need a Kinect.

---

## First run, step by step

1. **Check the Kinect works.** Stop the service (`sudo systemctl stop catguard`), because only one program can use the Kinect at a time. Run `.venv/bin/catguard -c config.toml check`. Expect 25–30 fps and a sensible depth range in millimetres. Then start the service again.
2. **Aim it.** Open **http://\<pi\>:8080** on a phone or PC on the same network. Pick **Camera** view. Use **▼ Down / ▲ Up**, or move the Kinect by hand, until the whole counter is in view and as large as possible.
3. **Check for sunlight.** Switch to **Above counter**. Dimmed areas have no depth reading. Holes near a sunny window are normal; the camera check covers them in daylight.
4. **Calibrate.** Clear the counter, step out of view and press **Recalibrate (counter empty!)**. The red "detection paused" banner disappears.
5. **Draw the outline.** Press **Draw counter outline**, tap the counter's corners on the picture, then **Save outline**. Use **Undo** and **Clear** to fix mistakes. Keep the outline on the worktop surface; leave out walls, cupboard fronts and the floor.
6. **Test the sound.** Press **Test sound**.
7. **Run in dry run for a few days.** Turn on **Dry run** in Home Assistant, or put `--dry-run` on the `ExecStart=` line of the service. Look at the snapshots it saves: was it the cat every time? Then turn dry run off.

> **Moved the Kinect?** Recalibrate and redraw the outline. Using the tilt buttons marks the calibration as out of date and pauses detection until you recalibrate. The tilt motor only moves when you press a button.

---

## Home Assistant (optional)

### Broker

If you don't have an MQTT broker yet, the simplest is the **Mosquitto broker** add-on in Home Assistant OS. For Docker (for example on a NAS), `deploy/nas/` has a `docker-compose.yml` for Mosquitto and Home Assistant. Set `TZ` in it to your time zone, then, from that folder, create a broker login and start it:

```sh
docker run --rm -v "$PWD/mosquitto/config:/mosquitto/config" \
  eclipse-mosquitto:2 mosquitto_passwd -c -b /mosquitto/config/passwd catguard 'YOUR_PASSWORD'
docker compose up -d
```

### Connect

In Home Assistant: Settings → Devices & services → Add integration → **MQTT**. Enter your broker's address, port `1883` and the login. A **Cat Guard** device appears on its own.

### Entities

| Entity | What it's for |
|---|---|
| **Enabled** (switch) | Turns detection off completely. |
| **Dry run (no sound)** (switch) | Detects and saves, but stays silent. |
| **Cat on counter** | On while the cat is detected. |
| **Triggers today** | Count since midnight. It survives restarts. |
| **Last trigger** | Time of the last trigger. |
| **Last snapshot** (camera) | Picture from the last trigger, with a box round the cat. |
| **Volume** | 0–100%, Cat Guard's own volume. The audio device's volume is separate (`alsamixer`). |
| **Cooldown** | 5–600 s between sounds. |
| **Night mode** | `depth_only` keeps detecting in the dark. `off` stops detecting while the room is dark. |
| **Test sound** / **Recalibrate (counter must be empty)** | Same as the buttons on the aiming page. |

Settings changed in Home Assistant are kept by the broker and come back after the Pi restarts.

### Example: phone notification with the snapshot

Settings → Automations & scenes → Create automation → ⋮ → **Edit in YAML**, then paste the YAML below. Replace the entity IDs with yours: find them under Settings → Devices & services → Entities by searching "cat guard". For your phone's notify action, go to Developer tools → Actions and type `notify.mobile_app`.

```yaml
alias: Cat Guard – cat on counter
mode: queued
triggers:
  - trigger: state
    entity_id: sensor.cat_guard_last_trigger
conditions:
  - condition: template   # ignore restarts, only real new triggers
    value_template: >
      {{ trigger.from_state is not none
         and trigger.from_state.state not in ['unknown', 'unavailable']
         and trigger.to_state.state not in ['unknown', 'unavailable'] }}
actions:
  - action: notify.mobile_app_YOUR_PHONE
    data:
      title: "🐱 Cat on the counter"
      message: "{{ states('sensor.cat_guard_triggers_today') }} today"
      data:
        image: /api/camera_proxy/camera.cat_guard_last_snapshot
        clickAction: "entityId:camera.cat_guard_last_snapshot"   # Android: tapping opens the snapshot
        tag: catguard
```

Other useful ideas:

- A "Pause 1 hour" button on the notification that turns **Enabled** off for an hour.
- An alert when the Cat Guard device has been unavailable for 10 minutes.

---

## Everyday use

### Events

Events are saved in one folder per day under `[storage] events_dir`, named by time:

| File | What it is | Kept for |
|---|---|---|
| `HHMMSS_dry.jpg` / `_cat.jpg` | Snapshot of a trigger, in dry run / with sound | 30 days |
| `HHMMSS_*.mp4` | 3 s before and 3 s after, with boxes drawn on | 30 days |
| `HHMMSS_cand.*` | A near miss | 30 days |
| `HHMMSS_*.raw.npz` | Raw colour, depth and the background at the time, for replay (10–30 MB) | 7 days |

### The aiming page (http://\<pi\>:8080)

It shows a live view (**Camera**, **Depth** or **Above counter**) with a status line (fps, cat present, depth coverage). It also has buttons for tilt, drawing the outline, recalibrating and testing the sound. Video is only encoded while someone has the page open.

The page has **no login**, so anyone on your network can use it. Don't expose port 8080 to the internet.

---

## Tuning: what to change when

| Symptom | What to do |
|---|---|
| The cat ignores the sound | Add more, different recordings. Make it louder. Lengthen the cooldown so it's less routine. |
| Too many plays while the cat sits there | Raise **Cooldown** (`[decision] cooldown_s`). |
| False triggers at night | Set **Night mode** to `off`, or replay the events to see why. |
| A trigger shows a person or object | Replay the event (see [Troubleshooting](#troubleshooting)) before changing anything. |
| A cat was missed | Look for a `_cand` near miss at that time and replay it. |
| Your cat is unusually small or large | Change `[depth] min_area_m2` / `max_area_m2` (top-down size in m²). Replay output prints each blob's size. |
| "Dark" switches at the wrong time | Change `[detector] min_brightness` (default 40 out of 255). |
| Pi too slow (fps below 15) | Raise `[detector] scan_every`, or set it to 0 to turn the camera check off. |

All settings are in `config.toml`; the [technical reference](#all-settings) lists them all. Restart the service after editing it.

---

## Troubleshooting

### It triggered on the wrong thing, or missed the cat

Don't guess; replay the event. Replay runs the recording through the current code, starting from the background the Pi had at the time. It lists every trigger with its reason (`depth:new ground 0.17`, `depth:arrived`, `camera` and so on) and the blob sizes:

```sh
.venv/bin/catguard -c config.toml replay /mnt/catcam/2025-01-01/124321_dry.raw.npz --live-baseline
```

Change a setting and replay again. Check the other saved events too before restarting the service.

A replay takes over a minute per event on the Pi while the service is running. It's much faster on a PC: copy the Pi's `config.toml`, `calibration/` and `models/` into a folder there, install Cat Guard (no Kinect needed) and run the same command.

Two things make old recordings misleading:

- **A recalibration since the recording.** Recordings from before it don't match the new calibration. Check with `journalctl -u catguard | grep -iE "recalib|tilt"`.
- **Replaying without `--live-baseline`.** The replay then starts from the calibration, not the background the Pi had learnt.

### Other problems

| Problem | Check |
|---|---|
| No sound | Run `catguard test-sound`. Check `[audio] device` (`aplay -l`), the device volume (`alsamixer`) and that **Dry run** is off. USB audio adapters can be silent through a USB hub, so plug straight into the Pi. On adapters with two sockets, use the headphone one (often green), not the mic one (pink). |
| Service keeps restarting | `journalctl -u catguard -n 50`. |
| Kinect not found | Is the power adapter connected? Does `lsusb` show `045e:02ae`? `gspca_kinect` must not be loaded (`lsmod`), and nothing else may be using the Kinect. |
| Home Assistant shows "Unavailable" | The Pi is off, or the broker is unreachable. |
| No files in the events folder | Check `mount | grep catcam`; otherwise look in `~/catguard-events`. |
| Low frame rate | The `stats:` line in the journal every 30 s shows fps, ms per frame and scan time. |

```sh
journalctl -u catguard -f | grep -vE "data bytes"         # live log
journalctl -u catguard --since today | grep -E "TRIGGER|near miss"
```

---

## Command reference

Run as `catguard -c config.toml <command>` (`-v` for more logging). Stop the service first for the commands that use the Kinect: `check`, `calibrate` and `record`.

| Command | What it does |
|---|---|
| `run [--dry-run]` | The live service (what systemd runs). |
| `check [--seconds N]` | Measures fps and saves `calibration/check_rgb.jpg` and `check_depth.png`. |
| `calibrate` | Captures the empty-counter calibration (keeps the existing outline). |
| `roi X,Y X,Y X,Y …` | Sets the counter outline from the command line and writes `roi_preview.jpg`. |
| `record DIR [--seconds N]` | Records raw colour and depth (default 30 s) for later replay. |
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

---

## Technical reference

### Code layout

| File | Contents |
|---|---|
| `catguard/capture.py` | Kinect via libfreenect (frames copied out of its buffers), tilt motor via ctypes, `ReplaySource` / `Recorder`. |
| `catguard/depth.py` | Calibration, temporal median, height map, blobs, adaptive background, `MotionGate`. |
| `catguard/detector.py` | YOLO ONNX wrapper: tiled counter scan, person check. |
| `catguard/scanner.py` | Background thread running the camera scan and person check on the latest frame. |
| `catguard/app.py` | `Pipeline` (per-frame decision), `Service` (live loop), depth vetoes for camera hits. |
| `catguard/decision.py` | Persistence, cooldown, hourly cap. |
| `catguard/storage.py` | Snapshots, clips, raw events, retention. |
| `catguard/mqtt_ha.py` | MQTT discovery, state, commands, persisted daily count. |
| `catguard/web.py` | Aiming page (MJPEG stream and API). |
| `catguard/audio.py` | WAV playback through `aplay`, random rotation (never the same twice in a row), software gain. |

### Depth path in detail

- **Calibration:** per-pixel median of 60 empty frames (pixels with no reading are ignored), plus a per-pixel noise estimate.
- **Filtering:** median of the last 3 frames per pixel; a pixel needs readings in 2 of 3. This removes the flicker that shiny or dark objects (pans, knife blocks) cause.
- **"Above the counter" threshold**, per pixel: the largest of 60 mm, 4 × measured noise, and 3 Kinect depth steps. The Kinect measures disparity, so depth comes in steps of about 2.85×10⁻⁶ · z² mm: about 24 mm at 2.9 m and 40 mm at 3.7 m.
- **Blobs:** connected regions above the threshold, with no upper height limit at this stage. That way a person leaning in forms one region that spills outside the outline and is rejected.
- **Cat-sized:** 0.015–0.30 m² top-down (pixels × (z / 525)²), at most 600 mm tall, at least 85% inside the outline.
- **Person-like (depth):** bigger than a cat, or ≥ 0.08 m² and touching the bottom of the picture. People close to the Kinect often give no depth at all, so the person check does most of this work.
- **Adaptive background:** each pixel tracks how often it's above or below the background (time constant 8 s, updated every 3rd frame).
  - Over 50% of the time, up to cat height: treated as background ("scenery"). A kettle left out is ignored after about 4 s.
  - Over 90%: the background itself is updated (about 20 s).
  - It may only drift up to max(150 mm, 5 depth steps) from the calibration. Sunlight biases the depth sensor, so after dusk surfaces read 1–4 steps nearer than a daytime calibration. A person is at least 0.5 m nearer and can never be absorbed.
  - Pixels under a person-like blob are frozen.
- **Moving by itself** (compared with 5 frames, about 0.33 s, earlier):
  - **New ground:** ≥ 12% of the blob lies outside everywhere a candidate has been in the last 3 s. There's a 0.4 s delay so the blob's own last few frames don't count.
  - **Interior change:** ≥ 20% of the blob's inner pixels (edges eroded) changed by more than 40 mm.
  - Either needs 3 readings within 1 s; the blob then counts as moving for 2 s.
  - **Arrived:** this also counts. A new blob persists for 4 frames, at a spot with no candidate in the last 30 s, and no person-like blob has been seen in the last 2 s. This is the cat jumping up.

### Camera path in detail

- Model `yolo11n` at 640, on 320-px tiles across the outline's bounding box (each upscaled 2×), in a background thread (about 570 ms per scan on a Pi 5). The bottom centre of the cat's box (its feet) must be inside the outline, grown by 25 px.
- Needs 2 positive scans in a row, or one ≥ 0.6. Results older than 1 s are discarded, and the snapshot is the exact frame the scan ran on.
- **Vetoes**, where depth has a reading:
  - If ≥ 30% of the box is more than 600 mm above the counter, or a person-like blob overlaps it, it's a person (hair can score as a cat).
  - If no depth candidate overlapped the box in the last 10 s and < 35% of the box is more than 80 mm above the calibrated counter, it's a static object, such as an appliance in shadow.
  - Where depth has no reading (sunlight), the camera alone decides.
- **Dark:** mean brightness inside the outline < 40. It's judged on the counter area, not the whole frame, because a bright window skews it. In the dark the camera and person checks are skipped.

### Person check

- `yolov8n` at 320 on the whole frame, after each camera scan (about 65 ms on a Pi 5).
- Gate at 0.4. In testing, people scored 0.76–0.93 and a cat on the counter at most 0.20.
- When it sees a person: no depth triggers for 5 s, and camera triggers need ≥ 0.6 and < 30% overlap with a person box.

### Reference measurements

From the original installation (Pi 5, Kinect 2.9–3.7 m from the counter):

- Kinect: about 27 fps colour and depth; the service processes 15 fps at about 23–28 ms per frame.
- Depth noise about 20 mm at 2–3 m.
- A still bag: ≤ 5% interior change and ≤ 2.2 px shift. A cat moving: 0.52–0.84.
- At that distance the cat was about 80 px long. The 320-px model missed it, and tiling with the 640 model fixed it. Mounting the Kinect closer helps more than a bigger model.
- Over three days of real use (36 saved events), the current version ignored every false alarm (people, an appliance in shadow) and caught every cat event, day and night.

### MQTT topics

The base topic is `catguard/` (`[mqtt] base_topic`). Home Assistant discovery messages go under `homeassistant/`. Device availability is on `catguard/availability`, with a last-will of `offline`.

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
| | `motion_shift_px`, `motion_iou_min` | 5, 0.7 | Present in code; the live gate uses new ground and interior change instead. |
| | `motion_hold_s` | 2 | Counts as moving this long after the last movement. |
| | `person_window_s`, `person_min_area_m2` | 2, 0.08 | Arrival rule; depth person size. |
| | `baseline_frames` | 60 | Frames used to calibrate. |
| | `absorb_after_s`, `static_tolerance_mm` | 180, 25 | Old absorb mechanism, **no longer used**. |
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
| `mqtt` | `enabled`, `host`, `port`, `username`, `password`, `base_topic`, `discovery_prefix` | | Broker. Always set `host`: the built-in default is the original author's address. |
| `storage` | `events_dir`, `fallback_dir`, `calibration_dir` | `/mnt/catcam`, `~/catguard-events`, `calibration` | Where files go. |
| | `retention_days`, `raw_retention_days` | 30, 7 | Clean-up. |
| | `pre_seconds`, `post_seconds` | 3, 3 | Clip length. |
| | `save_raw`, `save_near_misses` | true, true | Raw recordings; near misses. |
| `web` | `enabled`, `port` | true, 8080 | Aiming page. |

### Kinect quirks

- `sync_get_depth/video` return views onto libfreenect's internal buffers; copy them before keeping frames.
- Never set the tilt at start-up. It silently changes the view and invalidates the calibration. The service only reads the angle; moving it marks the calibration stale.
- Only one program can use the Kinect at a time: stop the service before `check`, `calibrate` or `record`.
