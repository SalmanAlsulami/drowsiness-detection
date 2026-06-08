"""
FastAPI backend for the drowsiness detection web app.
Receives JPEG frames via WebSocket, runs MediaPipe + model inference,
returns JSON results. All detection logic mirrors demo.py exactly.
"""

import asyncio
import base64
import json
import sys
import time
from collections import deque
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from torchvision import transforms

# resolve paths relative to this file so the app works from any cwd
_BACKEND_DIR  = Path(__file__).resolve().parent
_WEB_DIR      = _BACKEND_DIR.parent
_PROJECT_ROOT = _WEB_DIR.parent
_FRONTEND_DIR = _WEB_DIR / "frontend"

# add src/ to path so we can import model.py
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
from model import build_model  # noqa: E402  (import after sys.path change)

# ── model paths ────────────────────────────────────────────────────────────
_EYE_MODEL_PATH  = _PROJECT_ROOT / "outputs/models/efficientnetv2s_cbam_main_best.pth"
_YAWN_MODEL_PATH = _PROJECT_ROOT / "outputs/models/efficientnetv2s_cbam_yawn_best.pth"

DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 224

# ── constants (identical to demo.py) ──────────────────────────────────────
PERCLOS_WINDOW       = 90
PERCLOS_MIN_FRAMES   = 15
PERCLOS_THRESHOLD    = 0.35
EYE_PAD              = 0.35
CONSEC_CLOSED_FRAMES = 60
YAWN_FRAME_WINDOW    = 15
YAWN_FRAME_THRESH    = 0.40
YAWN_EVENT_WINDOW    = 120
YAWN_EVENT_THRESH    = 2
YAWN_COOLDOWN        = 2.0
HEAD_PITCH_THRESH    = 20.0
HEAD_POSE_FRAMES     = 20
HEAD_YAW_SKIP        = 45.0
GAZE_THRESH          = 0.35

RIGHT_EYE_IDX = [33,  7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
LEFT_EYE_IDX  = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
LEFT_IRIS_IDX  = [468, 469, 470, 471, 472]
RIGHT_IRIS_IDX = [473, 474, 475, 476, 477]

FACE_3D_POINTS = np.array([
    [0.0,    0.0,    0.0],
    [0.0,  -63.6,  -12.5],
    [-43.3,  32.7, -26.0],
    [43.3,   32.7, -26.0],
    [-28.9, -28.9, -24.1],
    [28.9,  -28.9, -24.1],
], dtype=np.float64)

FACE_2D_LANDMARK_IDX = [1, 152, 33, 263, 61, 291]

TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])


# ── model loading (once at startup) ───────────────────────────────────────
def _load_model(path: Path) -> torch.nn.Module:
    m = build_model(num_classes=2, freeze_backbone=False).to(DEVICE)
    m.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True))
    m.eval()
    return m


print(f"[startup] Device: {DEVICE}")
print(f"[startup] Loading eye model from {_EYE_MODEL_PATH.name} ...")
_eye_model = _load_model(_EYE_MODEL_PATH)

_yawn_model: Optional[torch.nn.Module] = None
if _YAWN_MODEL_PATH.exists():
    print(f"[startup] Loading yawn model from {_YAWN_MODEL_PATH.name} ...")
    _yawn_model = _load_model(_YAWN_MODEL_PATH)
else:
    print("[startup] Yawn model not found — yawn detection disabled.")

print("[startup] Models ready.")


# ── image processing helpers ───────────────────────────────────────────────
def _to_tensor(bgr_crop: np.ndarray) -> torch.Tensor:
    gray    = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2GRAY)
    rgb     = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    resized = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE))
    return TRANSFORM(resized).unsqueeze(0).to(DEVICE)


def _closed_prob(crop: np.ndarray) -> float:
    with torch.no_grad():
        return F.softmax(_eye_model(_to_tensor(crop)), dim=1)[0, 0].item()


def _yawn_prob(crop: np.ndarray) -> float:
    with torch.no_grad():
        return F.softmax(_yawn_model(_to_tensor(crop)), dim=1)[0, 1].item()


def _eye_crop(
    frame: np.ndarray, landmarks, indices: List[int], h: int, w: int
) -> Optional[np.ndarray]:
    xs = [landmarks[i].x * w for i in indices]
    ys = [landmarks[i].y * h for i in indices]
    x1, x2 = int(min(xs)), int(max(xs))
    y1, y2 = int(min(ys)), int(max(ys))
    px = int((x2 - x1) * EYE_PAD)
    py = int((y2 - y1) * EYE_PAD)
    x1, y1 = max(0, x1 - px), max(0, y1 - py)
    x2, y2 = min(w, x2 + px), min(h, y2 + py)
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2]


def _face_crop(
    frame: np.ndarray, landmarks, h: int, w: int, pad: float = 0.15
) -> Optional[np.ndarray]:
    xs = [lm.x * w for lm in landmarks]
    ys = [lm.y * h for lm in landmarks]
    x1, x2 = int(min(xs)), int(max(xs))
    y1, y2 = int(min(ys)), int(max(ys))
    px = int((x2 - x1) * pad)
    py = int((y2 - y1) * pad)
    x1, y1 = max(0, x1 - px), max(0, y1 - py)
    x2, y2 = min(w, x2 + px), min(h, y2 + py)
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2]


def _head_pose(landmarks, h: int, w: int, focal: float) -> Tuple[float, float]:
    pts_2d = np.array(
        [[landmarks[i].x * w, landmarks[i].y * h] for i in FACE_2D_LANDMARK_IDX],
        dtype=np.float64,
    )
    cx, cy = w / 2.0, h / 2.0
    cam_mat = np.array(
        [[focal, 0, cx], [0, focal, cy], [0, 0, 1]], dtype=np.float64
    )
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)
    _, rvec, _ = cv2.solvePnP(
        FACE_3D_POINTS, pts_2d, cam_mat, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    rot_mat, _ = cv2.Rodrigues(rvec)
    proj_mat = np.hstack((rot_mat, np.zeros((3, 1), dtype=np.float64)))
    _, _, _, _, _, _, euler = cv2.decomposeProjectionMatrix(proj_mat)
    pitch = float(euler[0, 0])
    yaw   = float(euler[1, 0])
    if pitch < -90:
        pitch += 180
    elif pitch > 90:
        pitch -= 180
    return pitch, yaw


def _gaze_offset(landmarks, h: int, w: int) -> float:
    offsets = []
    for iris_idx, (lc_idx, rc_idx) in [
        (LEFT_IRIS_IDX[0],  (33,  133)),
        (RIGHT_IRIS_IDX[0], (362, 263)),
    ]:
        ix    = landmarks[iris_idx].x * w
        lx    = landmarks[lc_idx].x * w
        rx    = landmarks[rc_idx].x * w
        eye_w = abs(rx - lx)
        if eye_w > 0:
            offsets.append((ix - (lx + rx) / 2.0) / eye_w)
    return sum(offsets) / len(offsets) if offsets else 0.0


# ── per-session state ──────────────────────────────────────────────────────
class _SessionState:
    def __init__(self):
        self.eye_states            = deque(maxlen=PERCLOS_WINDOW)
        self.yawn_frames           = deque(maxlen=YAWN_FRAME_WINDOW)
        self.yawn_event_times: List[float] = []
        self.last_yawn_event_time  = 0.0
        self.head_pose_counter     = 0
        self.gaze_counter          = 0
        self.yawn_active           = False
        self.consec_closed_count   = 0
        self.face_mesh             = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def close(self):
        self.face_mesh.close()


# ── frame inference (blocking, runs in thread pool) ────────────────────────
def _process_frame(frame: np.ndarray, s: _SessionState) -> dict:
    h, w  = frame.shape[:2]
    focal = float(w)
    rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res   = s.face_mesh.process(rgb)

    perclos       = 0.0
    yawn_events   = 0
    head_alert    = False
    pitch         = 0.0
    yaw_angle     = 0.0
    eye_left_prob: Optional[float]  = None
    eye_right_prob: Optional[float] = None
    yawn_prob_val: Optional[float]  = None
    face_found    = bool(res.multi_face_landmarks)

    if face_found:
        lm = res.multi_face_landmarks[0].landmark

        # head pose
        try:
            pitch, yaw_angle = _head_pose(lm, h, w, focal)
            if pitch > HEAD_PITCH_THRESH:
                s.head_pose_counter += 1
            else:
                s.head_pose_counter = max(0, s.head_pose_counter - 1)
            head_alert = s.head_pose_counter >= HEAD_POSE_FRAMES
        except cv2.error:
            yaw_angle = 0.0

        # gaze
        gaze_off = _gaze_offset(lm, h, w)
        if abs(gaze_off) > GAZE_THRESH:
            s.gaze_counter += 1
        else:
            s.gaze_counter = max(0, s.gaze_counter - 1)

        face_frontal = abs(yaw_angle) <= HEAD_YAW_SKIP
        if face_frontal:
            closed_probs = []
            for eye_idx, side in ((RIGHT_EYE_IDX, "R"), (LEFT_EYE_IDX, "L")):
                crop = _eye_crop(frame, lm, eye_idx, h, w)
                if crop is None:
                    continue
                p = _closed_prob(crop)
                closed_probs.append(p)
                if side == "R":
                    eye_right_prob = p
                else:
                    eye_left_prob = p

            if closed_probs:
                is_closed = all(p > 0.40 for p in closed_probs)
                s.eye_states.append(1 if is_closed else 0)
                s.consec_closed_count = s.consec_closed_count + 1 if is_closed else 0
            else:
                s.consec_closed_count = 0
        else:
            s.consec_closed_count = 0

        # PERCLOS
        if len(s.eye_states) >= PERCLOS_MIN_FRAMES:
            perclos = sum(s.eye_states) / len(s.eye_states)

        # yawn
        if _yawn_model is not None:
            face = _face_crop(frame, lm, h, w)
            if face is not None:
                yawn_p = _yawn_prob(face)
                yawn_prob_val = yawn_p
                s.yawn_frames.append(1 if yawn_p > 0.5 else 0)

            if len(s.yawn_frames) == YAWN_FRAME_WINDOW:
                cur_yawn = (sum(s.yawn_frames) / YAWN_FRAME_WINDOW) > YAWN_FRAME_THRESH
                was_yawn = s.yawn_active
                s.yawn_active = cur_yawn
                now = time.time()
                if cur_yawn and not was_yawn and (now - s.last_yawn_event_time) >= YAWN_COOLDOWN:
                    s.yawn_event_times.append(now)
                    s.last_yawn_event_time = now

            now = time.time()
            s.yawn_event_times = [t for t in s.yawn_event_times if now - t <= YAWN_EVENT_WINDOW]
            yawn_events = len(s.yawn_event_times)
    else:
        s.consec_closed_count = 0

    # alert logic (identical to demo.py)
    perclos_alert = perclos >= PERCLOS_THRESHOLD
    yawn_alert    = yawn_events >= YAWN_EVENT_THRESH
    consec_alert  = s.consec_closed_count >= CONSEC_CLOSED_FRAMES

    active_count = sum([perclos_alert, yawn_alert, head_alert])
    is_drowsy    = perclos_alert or consec_alert or active_count >= 2
    is_warning   = (not is_drowsy) and active_count >= 1

    alert_level = "danger" if is_drowsy else ("warning" if is_warning else "safe")

    return {
        "alert_level":   alert_level,
        "face_found":    face_found,
        "perclos":       round(perclos, 4),
        "perclos_alert": perclos_alert,
        "perclos_frames": len(s.eye_states),
        "yawn_events":   yawn_events,
        "yawn_alert":    yawn_alert,
        "head_alert":    head_alert,
        "pitch":         round(pitch, 1),
        "consec_closed": s.consec_closed_count,
        "eye_left_prob":  round(eye_left_prob,  3) if eye_left_prob  is not None else None,
        "eye_right_prob": round(eye_right_prob, 3) if eye_right_prob is not None else None,
        "yawn_prob":      round(yawn_prob_val,  3) if yawn_prob_val  is not None else None,
    }


# ── FastAPI app ────────────────────────────────────────────────────────────
app = FastAPI(title="Drowsiness Detection")

app.mount("/static", StaticFiles(directory=str(_FRONTEND_DIR)), name="static")


@app.get("/")
async def root():
    return FileResponse(str(_FRONTEND_DIR / "index.html"))


@app.get("/health")
async def health():
    return {"status": "ok", "device": DEVICE}


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    state = _SessionState()
    loop  = asyncio.get_event_loop()

    async def _keepalive():
        """Send a ping every 25 s so Railway's proxy doesn't close the connection."""
        while True:
            await asyncio.sleep(25)
            try:
                await ws.send_text('{"ping":1}')
            except Exception:
                break

    ping_task = asyncio.create_task(_keepalive())
    try:
        while True:
            data = await ws.receive_text()
            # ignore client-side heartbeats if any
            if data == '{"ping":1}':
                continue
            # data is a base64 data-URL: "data:image/jpeg;base64,..."
            _, encoded = data.split(",", 1)
            img_bytes = base64.b64decode(encoded)
            arr   = np.frombuffer(img_bytes, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is None:
                continue
            # run blocking inference in thread pool so the event loop stays free
            result = await loop.run_in_executor(None, _process_frame, frame, state)
            await ws.send_text(json.dumps(result))
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"[ws] session error: {e}")
    finally:
        ping_task.cancel()
        state.close()
