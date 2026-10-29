"""YOLOv8 (ONNX, CPU) cat confirmation on an RGB crop."""
from __future__ import annotations

import cv2
import numpy as np

from .config import DetectorCfg

COCO_CAT = 15
COCO_PERSON = 0


def brightness(rgb: np.ndarray) -> float:
    return float(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).mean())


def crop_box(bbox, shape, margin: float, min_side: int = 96):
    x, y, w, h = bbox
    H, W = shape[:2]
    side = max(int(max(w, h) * (1 + margin)), min_side)
    cx, cy = x + w // 2, y + h // 2
    x0 = max(0, min(W - side, cx - side // 2))
    y0 = max(0, min(H - side, cy - side // 2))
    return x0, y0, min(side, W), min(side, H)


class CatDetector:
    def __init__(self, cfg: DetectorCfg, model_path, imgsz: int | None = None, threads: int = 4,
                 full_frame_fallback: bool = True):
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        self.sess = ort.InferenceSession(str(model_path), so, providers=["CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0].name
        self.cfg = cfg
        self.imgsz = imgsz or cfg.imgsz
        self.full_frame_fallback = full_frame_fallback

    def _prep(self, rgb: np.ndarray):
        s = self.imgsz
        h, w = rgb.shape[:2]
        r = s / max(h, w)
        nh, nw = int(round(h * r)), int(round(w * r))
        img = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((s, s, 3), 114, np.uint8)
        top, left = (s - nh) // 2, (s - nw) // 2
        canvas[top:top + nh, left:left + nw] = img
        x = canvas.astype(np.float32).transpose(2, 0, 1)[None] / 255.0
        return x, r, left, top

    def cat_score(self, rgb: np.ndarray) -> tuple[float, tuple[int, int, int, int] | None]:
        """Best cat confidence in the image (0..1) and its box in image pixels."""
        x, r, left, top = self._prep(rgb)
        out = self.sess.run(None, {self.inp: x})[0][0]  # (84, N)
        cls = out[4:]
        cat = cls[COCO_CAT]
        # require cat to be the top class for that anchor
        top_cls = cls.argmax(axis=0)
        cat = np.where(top_cls == COCO_CAT, cat, 0.0)
        # ignore "cats" that are mostly inside a detected person (hair, fluffy
        # dressing gowns): seen as cat=0.62 on someone's head
        pers = np.nonzero((top_cls == COCO_PERSON) & (cls[COCO_PERSON] >= self.cfg.person_conf))[0]
        cand = np.nonzero(cat >= 0.1)[0]
        if pers.size and cand.size:
            def xyxy(idx):
                cx, cy, w, h = out[0, idx], out[1, idx], out[2, idx], out[3, idx]
                return cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
            cx0, cy0, cx1, cy1 = (v[:, None] for v in xyxy(cand))
            px0, py0, px1, py1 = (v[None, :] for v in xyxy(pers))
            iw = np.clip(np.minimum(cx1, px1) - np.maximum(cx0, px0), 0, None)
            ih = np.clip(np.minimum(cy1, py1) - np.maximum(cy0, py0), 0, None)
            inside = (iw * ih) / np.maximum(1e-6, (cx1 - cx0) * (cy1 - cy0))
            cat[cand[(inside >= 0.6).any(axis=1)]] = 0.0
        i = int(cat.argmax())
        score = float(cat[i])
        if score <= 0:
            return 0.0, None
        cx, cy, w, h = out[:4, i]
        bx = int((cx - w / 2 - left) / r)
        by = int((cy - h / 2 - top) / r)
        return score, (bx, by, int(w / r), int(h / r))

    def persons(self, rgb: np.ndarray, conf: float) -> tuple[float, list[tuple[int, int, int, int]]]:
        """Full-frame person check: (best score, boxes >= conf) in image pixels."""
        x, r, left, top = self._prep(rgb)
        out = self.sess.run(None, {self.inp: x})[0][0]
        cls = out[4:]
        p = np.where(cls.argmax(axis=0) == COCO_PERSON, cls[COCO_PERSON], 0.0)
        idx = np.nonzero(p >= conf)[0]
        boxes = []
        for i in idx[np.argsort(-p[idx])][:10]:
            cx, cy, w, h = out[:4, i]
            boxes.append((int((cx - w / 2 - left) / r), int((cy - h / 2 - top) / r), int(w / r), int(h / r)))
        return float(p.max()) if p.size else 0.0, boxes

    def confirm(self, rgb: np.ndarray, bbox) -> tuple[bool, float]:
        x0, y0, w, h = crop_box(bbox, rgb.shape, self.cfg.crop_margin)
        crop = rgb[y0:y0 + h, x0:x0 + w]
        score, _ = self.cat_score(crop)
        if score < self.cfg.cat_conf and self.full_frame_fallback:
            # fall back to the full frame (crop can cut a stretched-out cat)
            full, _ = self.cat_score(rgb)
            score = max(score, full)
        return score >= self.cfg.cat_conf, score

    def scan(self, rgb: np.ndarray, region: tuple[int, int, int, int],
             roi: np.ndarray) -> tuple[bool, float, tuple[int, int, int, int] | None]:
        """Look for a cat standing on the counter, without needing depth.

        The ROI's bounding box is cut into square tiles (so a cat stays large
        relative to the model input) and each tile is checked. A detection
        counts only if the bottom-centre of the cat box (its feet) is inside
        the ROI mask.
        """
        x, y, w, h = region
        H, W = rgb.shape[:2]
        side = min(max(h + 2 * int(h * 0.25), self.cfg.scan_tile), W, H)
        y0 = max(0, min(H - side, y + h // 2 - side // 2))
        n = max(1, int(np.ceil(w / (side * 0.8))))
        xs = [x] if n == 1 else np.linspace(x, x + w - side, n).astype(int)
        best = (False, 0.0, None)
        for x0 in xs:
            x0 = int(max(0, min(W - side, x0)))
            score, box = self.cat_score(rgb[y0:y0 + side, x0:x0 + side])
            if box is None or score < self.cfg.cat_conf:
                best = max(best, (False, score, None), key=lambda b: b[1])
                continue
            bx, by, bw, bh = box
            gx, gy = x0 + bx, y0 + by
            fx = int(np.clip(gx + bw // 2, 0, W - 1))
            fy = int(np.clip(gy + bh - 1, 0, H - 1))
            on = bool(roi[fy, fx])
            cand = (on, score, (gx, gy, bw, bh))
            if (on, score) > (best[0], best[1]):
                best = cand
        return best
