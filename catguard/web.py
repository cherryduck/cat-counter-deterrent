"""Live view + aiming page served from the Pi (LAN only, no auth).

  /                 page: live view, tilt buttons, draw counter outline, recalibrate
  /stream?view=     MJPEG (rgb | depth | height)
  /snapshot.jpg     single frame
  /api/state        JSON status
  POST /api/tilt {"deg": n} | /api/roi {"polygon": [[x,y],...]} | /api/recalibrate | /api/test_sound
"""
from __future__ import annotations

import json
import logging
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np

from .depth import colorize_depth

log = logging.getLogger(__name__)


class FrameHub:
    """Latest frame + verdict, shared between the main loop and HTTP threads."""

    def __init__(self):
        self._cv = threading.Condition()
        self._seq = 0
        self._data = None
        self.last_client = 0.0

    @property
    def seq(self) -> int:
        return self._seq

    @property
    def wanted(self) -> bool:
        return time.monotonic() - self.last_client < 5.0

    def publish(self, rgb, depth, verdict, height):
        with self._cv:
            self._data = (rgb, depth, verdict, height)
            self._seq += 1
            self._cv.notify_all()

    def wait(self, seq: int, timeout: float = 2.0):
        with self._cv:
            self._cv.wait_for(lambda: self._seq != seq and self._data is not None, timeout=timeout)
            return self._seq, self._data


def render(data, view: str, polygon, info: dict) -> np.ndarray:
    rgb, depth, v, height = data
    if view == "depth":
        img = colorize_depth(depth)
    else:
        img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        if view == "height" and height is not None:
            above = height >= info.get("min_height_mm", 60)
            red = img.copy()
            red[above] = (0, 0, 255)
            img = cv2.addWeighted(red, 0.55, img, 0.45, 0)
            img[depth == 0] = (img[depth == 0] * 0.4).astype(np.uint8)  # no depth data: dimmed
    if polygon:
        cv2.polylines(img, [np.asarray(polygon, np.int32)], True, (0, 255, 255), 2)
    if v is not None:
        for b in v.candidates:
            x, y, w, h = b.bbox
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        if v.cat_box is not None:
            x, y, w, h = (int(a) for a in v.cat_box)
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return img


class WebUI:
    def __init__(self, service, port: int):
        self.svc = service
        self.hub = FrameHub()
        hub, svc = self.hub, service

        class H(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, format, *args):  # noqa: A002
                pass

            def _send(self, code, body: bytes, ctype="application/json"):
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                u = urlparse(self.path)
                q = parse_qs(u.query)
                view = q.get("view", ["rgb"])[0]
                hub.last_client = time.monotonic()
                if u.path == "/":
                    self._send(200, PAGE.encode(), "text/html; charset=utf-8")
                elif u.path == "/api/state":
                    self._send(200, json.dumps(svc.web_state()).encode())
                elif u.path == "/snapshot.jpg":
                    _, data = hub.wait(hub.seq, 3)   # next fresh frame
                    if data is None:
                        return self._send(503, b"no frame")
                    ok, jpg = cv2.imencode(".jpg", render(data, view, svc.calib.polygon, svc.web_state()))
                    self._send(200, jpg.tobytes(), "image/jpeg")
                elif u.path == "/stream":
                    self.send_response(200)
                    self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    seq, last = 0, 0.0
                    try:
                        while True:
                            hub.last_client = time.monotonic()
                            seq, data = hub.wait(seq)
                            if data is None or time.monotonic() - last < 0.09:  # ~10 fps
                                continue
                            last = time.monotonic()
                            img = render(data, view, svc.calib.polygon, svc.web_state())
                            ok, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 75])
                            b = jpg.tobytes()
                            self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                             + str(len(b)).encode() + b"\r\n\r\n" + b + b"\r\n")
                    except (BrokenPipeError, ConnectionResetError):
                        return
                else:
                    self._send(404, b"not found", "text/plain")

            def do_POST(self):
                n = int(self.headers.get("Content-Length") or 0)
                try:
                    body = json.loads(self.rfile.read(n) or b"{}")
                except ValueError:
                    return self._send(400, b'{"error":"bad json"}')
                p = urlparse(self.path).path
                if p == "/api/tilt":
                    svc.request("tilt", max(-27, min(27, int(body.get("deg", 0)))))
                elif p == "/api/roi":
                    poly = [[int(x), int(y)] for x, y in body.get("polygon", [])]
                    if len(poly) < 3:
                        return self._send(400, b'{"error":"need 3+ points"}')
                    svc.request("roi", poly)
                elif p == "/api/recalibrate":
                    svc.request("recalibrate", None)
                elif p == "/api/test_sound":
                    svc.request("test_sound", None)
                else:
                    return self._send(404, b'{"error":"not found"}')
                self._send(200, b'{"ok":true}')

        self.server = ThreadingHTTPServer(("0.0.0.0", port), H)
        self.server.daemon_threads = True
        threading.Thread(target=self.server.serve_forever, daemon=True, name="web").start()
        log.info("live view on http://0.0.0.0:%d/", port)

    def stop(self):
        self.server.shutdown()


PAGE = r"""<!doctype html><html><head><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Cat Guard</title><style>
body{font-family:system-ui,sans-serif;margin:12px;background:#111;color:#eee}
#wrap{position:relative;width:100%;max-width:640px;aspect-ratio:4/3}
#wrap img,#wrap canvas{position:absolute;inset:0;width:100%;height:100%}
button{font-size:16px;padding:8px 12px;margin:3px;border-radius:6px;border:1px solid #555;background:#2a2a2a;color:#eee}
button.on{background:#2d6cdf}.row{margin:8px 0}#status{font-family:monospace;font-size:13px;white-space:pre-wrap}
#banner{background:#a33;padding:8px;border-radius:6px;display:none;margin:6px 0}
</style></head><body>
<h3 style="margin:4px 0">Cat Guard - live view</h3>
<div id="banner">View changed since calibration - detection paused. Clear the counter, then press <b>Recalibrate</b>.</div>
<div id="wrap"><img id="cam"><canvas id="cv" width="640" height="480"></canvas></div>
<div class="row">View:
 <button data-v="rgb" class="on">Camera</button><button data-v="depth">Depth</button><button data-v="height">Above counter</button></div>
<div class="row">Tilt: <button onclick="tilt(-2)">&#9660; Down</button><button onclick="tilt(0,true)">Level</button>
 <button onclick="tilt(2)">&#9650; Up</button> <span id="tiltv"></span></div>
<div class="row"><button id="edit" onclick="toggleEdit()">Draw counter outline</button>
 <button onclick="pts.pop();draw()">Undo</button><button onclick="pts=[];draw()">Clear</button>
 <button onclick="saveRoi()">Save outline</button></div>
<div class="row"><button onclick="post('/api/recalibrate')">Recalibrate (counter empty!)</button>
 <button onclick="post('/api/test_sound')">Test sound</button></div>
<div id="status"></div>
<script>
let view='rgb',editing=false,pts=[],state={};
const cam=document.getElementById('cam'),cv=document.getElementById('cv'),cx=cv.getContext('2d');
function setView(v){view=v;cam.src='/stream?view='+v+'&t='+Date.now();
 document.querySelectorAll('[data-v]').forEach(b=>b.classList.toggle('on',b.dataset.v==v));}
document.querySelectorAll('[data-v]').forEach(b=>b.onclick=()=>setView(b.dataset.v));
function post(u,body){return fetch(u,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body||{})})}
function tilt(d,abs){let t=abs?d:(state.tilt||0)+d;post('/api/tilt',{deg:t});}
function toggleEdit(){editing=!editing;document.getElementById('edit').classList.toggle('on',editing);
 if(editing&&!pts.length&&state.polygon)pts=state.polygon.map(p=>p.slice());draw();}
cv.onclick=e=>{if(!editing)return;const r=cv.getBoundingClientRect();
 pts.push([Math.round((e.clientX-r.left)/r.width*640),Math.round((e.clientY-r.top)/r.height*480)]);draw();};
function draw(){cx.clearRect(0,0,640,480);if(!editing)return;cx.strokeStyle='#f0f';cx.fillStyle='rgba(255,0,255,.2)';cx.lineWidth=2;
 cx.beginPath();pts.forEach((p,i)=>i?cx.lineTo(p[0],p[1]):cx.moveTo(p[0],p[1]));if(pts.length>2){cx.closePath();cx.fill();}cx.stroke();
 cx.fillStyle='#f0f';pts.forEach(p=>{cx.beginPath();cx.arc(p[0],p[1],5,0,7);cx.fill();});}
async function saveRoi(){if(pts.length<3)return alert('Tap at least 3 corners of the counter');
 const r=await post('/api/roi',{polygon:pts});if(r.ok){editing=false;document.getElementById('edit').classList.remove('on');draw();}}
async function poll(){try{state=await (await fetch('/api/state')).json();
 document.getElementById('tiltv').textContent=state.tilt+'\u00b0';
 document.getElementById('banner').style.display=state.calib_stale?'block':'none';
 document.getElementById('status').textContent=
  `fps ${state.fps}  ${state.label}\nenabled ${state.enabled}  dry run ${state.dry_run}  cat present ${state.present}\n`+
  `depth valid on counter ${state.roi_depth_valid}%   triggers today ${state.triggers_today}`;}catch(e){}
 setTimeout(poll,1000);}
setView('rgb');poll();
</script></body></html>"""
