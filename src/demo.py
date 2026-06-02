"""
Usage:
  python src/demo.py
  python src/demo.py --cam 0
  python src/demo.py --no-yawn
"""

import argparse
from collections import deque
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

from model import build_model
from alert_system import AlertManager

PROJECT_ROOT       = Path(__file__).resolve().parent.parent
DEFAULT_EYE_MODEL  = PROJECT_ROOT / "outputs/models/efficientnetv2s_cbam_main_best.pth"
DEFAULT_YAWN_MODEL = PROJECT_ROOT / "outputs/models/efficientnetv2s_cbam_yawn_best.pth"

DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 224

# PERCLOS: we track eye state over the last 90 frames (3 seconds at 30fps)
# if more than 60% of frames show closed eyes, we trigger an alert
PERCLOS_WINDOW    = 90    # 3-second rolling window
PERCLOS_MIN_FRAMES = 30   # start calculating after 1 second (don't wait for full buffer)
PERCLOS_THRESHOLD = 0.35  # 35% closure triggers alert (literature range: 30-40%)
EYE_PAD           = 0.35   # how much extra space to add around the eye crop

# if the eye stays closed for 60 frames in a row (2 seconds) we trigger immediately
CONSEC_CLOSED_FRAMES = 60

# yawn: we look at a short window of 15 frames
# if 40% or more show yawning, we count it as one yawn event
# if 2 or more yawn events happen within 2 minutes, we alert
YAWN_FRAME_WINDOW = 15
YAWN_FRAME_THRESH = 0.40
YAWN_EVENT_WINDOW = 120
YAWN_EVENT_THRESH = 2
YAWN_COOLDOWN     = 2.0   # minimum gap between two yawn events in seconds

# head pose: alert if head is nodding down more than 20 degrees for 20 frames
HEAD_PITCH_THRESH = 20.0
HEAD_POSE_FRAMES  = 20
# if the face is turned more than 25 degrees sideways, we skip eye detection
# because the model would see a side view of the eye and think it's closed
HEAD_YAW_SKIP     = 25.0

# gaze: if the iris moves too far to the side for 30 frames, flag it
GAZE_THRESH = 0.35
GAZE_FRAMES = 30

# MediaPipe landmark indices for the eye contour (16 points per eye)
RIGHT_EYE_IDX = [33,  7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
LEFT_EYE_IDX  = [362,382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]

# iris landmark indices, only available when refine_landmarks=True
LEFT_IRIS_IDX  = [468, 469, 470, 471, 472]
RIGHT_IRIS_IDX = [473, 474, 475, 476, 477]

# 3D coordinates of 6 face points in millimeters (used for head pose estimation)
FACE_3D_POINTS = np.array([
    [0.0,    0.0,    0.0],    # nose tip
    [0.0,  -63.6,  -12.5],   # chin
    [-43.3,  32.7, -26.0],   # left eye outer corner
    [43.3,   32.7, -26.0],   # right eye outer corner
    [-28.9, -28.9, -24.1],   # left mouth corner
    [28.9,  -28.9, -24.1],   # right mouth corner
], dtype=np.float64)

# matching MediaPipe landmark indices for the 6 points above
FACE_2D_LANDMARK_IDX = [1, 152, 33, 263, 61, 291]

# normalization applied to every image before passing to the model
TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])


def _load(path: Path) -> torch.nn.Module:
    # loads a saved model checkpoint from disk
    m = build_model(num_classes=2, freeze_backbone=False).to(DEVICE)
    m.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True))
    m.eval()
    return m


def _to_tensor(bgr_crop: np.ndarray) -> torch.Tensor:
    # converts a BGR image crop to the format the model expects:
    # grayscale -> repeat to 3 channels -> resize -> normalize
    gray    = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2GRAY)
    rgb     = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    resized = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE))
    return TRANSFORM(resized).unsqueeze(0).to(DEVICE)


def _closed_prob(model, crop: np.ndarray) -> float:
    # returns the probability that the eye is closed (index 0 alphabetically)
    with torch.no_grad():
        return F.softmax(model(_to_tensor(crop)), dim=1)[0, 0].item()


def _yawn_prob(model, crop: np.ndarray) -> float:
    # returns the probability that the person is yawning (index 1 alphabetically)
    with torch.no_grad():
        return F.softmax(model(_to_tensor(crop)), dim=1)[0, 1].item()


def _eye_crop(
    frame: np.ndarray,
    landmarks,
    indices: List[int],
    h: int,
    w: int,
) -> Optional[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
    # finds the bounding box around the eye landmarks and returns the cropped region
    xs = [landmarks[i].x * w for i in indices]
    ys = [landmarks[i].y * h for i in indices]
    x1, x2 = int(min(xs)), int(max(xs))
    y1, y2 = int(min(ys)), int(max(ys))
    # add some padding so we don't crop too tight
    px = int((x2 - x1) * EYE_PAD)
    py = int((y2 - y1) * EYE_PAD)
    x1, y1 = max(0, x1 - px), max(0, y1 - py)
    x2, y2 = min(w, x2 + px), min(h, y2 + py)
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2], (x1, y1, x2, y2)


def _face_crop(
    frame: np.ndarray,
    landmarks,
    h: int,
    w: int,
    pad: float = 0.15,
) -> Optional[np.ndarray]:
    # crops the full face region for the yawn model
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


def _head_pose(
    landmarks,
    h: int,
    w: int,
    focal_length: float,
) -> Tuple[float, float, float]:
    # estimates head rotation (pitch, yaw, roll) in degrees
    # uses 6 known 3D face points and their 2D positions in the image
    pts_2d = np.array(
        [[landmarks[i].x * w, landmarks[i].y * h] for i in FACE_2D_LANDMARK_IDX],
        dtype=np.float64,
    )
    cx, cy = w / 2.0, h / 2.0
    # approximate camera matrix using the image width as focal length
    cam_mat = np.array(
        [[focal_length, 0, cx], [0, focal_length, cy], [0, 0, 1]],
        dtype=np.float64,
    )
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)
    # solvePnP finds the rotation that maps 3D face points onto the 2D image
    _, rvec, _ = cv2.solvePnP(
        FACE_3D_POINTS, pts_2d, cam_mat, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    rot_mat, _ = cv2.Rodrigues(rvec)
    proj_mat = np.hstack((rot_mat, np.zeros((3, 1), dtype=np.float64)))
    _, _, _, _, _, _, euler = cv2.decomposeProjectionMatrix(proj_mat)
    pitch = float(euler[0, 0])
    yaw   = float(euler[1, 0])
    roll  = float(euler[2, 0])
    # decomposeProjectionMatrix gives ~180 degrees for looking straight ahead
    # we shift it so that 0 = looking straight, positive = nodding down
    if pitch < -90:
        pitch += 180
    elif pitch > 90:
        pitch -= 180
    return pitch, yaw, roll


def _gaze_offset(landmarks, h: int, w: int) -> float:
    # measures how far the iris is from the center of each eye
    # returns a value close to 0 when looking straight, larger when looking sideways
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



def _draw_indicator(
    frame: np.ndarray,
    name: str,
    value: str,
    active: bool,
    y: int,
) -> None:
    # draws one status line at the bottom of the frame
    # red with [!] when active, grey with [ ] when not
    color  = (0, 0, 255) if active else (200, 200, 200)
    marker = "[!]" if active else "[ ]"
    cv2.putText(frame, f"{marker} {name}: {value}",
                (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def run_demo(
    eye_model_path:  Path,
    yawn_model_path: Optional[Path],
    cam_index:       int,
) -> None:

    print(f"Device         : {DEVICE}")
    eye_model = _load(eye_model_path)
    print(f"Eye model      : {eye_model_path.name}")

    yawn_model = None
    if yawn_model_path and yawn_model_path.exists():
        yawn_model = _load(yawn_model_path)
        print(f"Yawn model     : {yawn_model_path.name}")
    else:
        print("Yawn model     : not loaded (run train.py --task yawn first)")

    alert_mgr = AlertManager(PROJECT_ROOT)

    # MediaPipe face mesh gives us 478 facial landmarks per frame
    # refine_landmarks=True adds iris landmarks which we need for gaze tracking
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {cam_index}")
    cv2.namedWindow("Drowsiness Detection")
    print("Press Q / ESC to quit, or close the window.\n")

    # these buffers track state across multiple frames
    eye_states: deque   = deque(maxlen=PERCLOS_WINDOW)    # 1=closed, 0=open per frame
    yawn_frames         = deque(maxlen=YAWN_FRAME_WINDOW) # 1=yawn, 0=no_yawn per frame
    yawn_event_times: List[float] = []                    # timestamps of yawn events
    last_yawn_event_time: float   = 0.0
    head_pose_counter   = 0
    gaze_counter        = 0
    yawn_active         = False   # remembers if we were yawning last frame
    consec_closed_count = 0       # how many frames in a row the eye has been closed

    import time

    fps_prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # calculate FPS
        now = time.time()
        fps = 1.0 / (now - fps_prev_time) if (now - fps_prev_time) > 0 else 0
        fps_prev_time = now

        h, w  = frame.shape[:2]
        focal = w                          # use image width as approximate focal length
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res   = face_mesh.process(rgb)

        # reset per-frame values
        perclos        = 0.0
        yawn_events_1m = 0
        head_alert = False
        pitch = yaw = 0.0

        if res.multi_face_landmarks:
            lm = res.multi_face_landmarks[0].landmark

            # compute head pose first so we can use yaw to decide whether to run eye detection
            try:
                pitch, yaw, _ = _head_pose(lm, h, w, focal)
                if pitch > HEAD_PITCH_THRESH:
                    head_pose_counter += 1
                else:
                    head_pose_counter = max(0, head_pose_counter - 1)
                head_alert = head_pose_counter >= HEAD_POSE_FRAMES
            except cv2.error:
                yaw = 0.0

            # compute gaze offset first so we can use it to gate eye detection
            gaze_off = _gaze_offset(lm, h, w)
            if abs(gaze_off) > GAZE_THRESH:
                gaze_counter += 1
            else:
                gaze_counter = max(0, gaze_counter - 1)

            # skip eye detection if head is turned sideways OR eyes are looking far to the side
            # both cases make the eye crop look closed even when the eye is open
            face_frontal = abs(yaw) <= HEAD_YAW_SKIP and abs(gaze_off) <= GAZE_THRESH
            if face_frontal:
                closed_probs = []
                for eye_idx, side in ((RIGHT_EYE_IDX, "R"), (LEFT_EYE_IDX, "L")):
                    out = _eye_crop(frame, lm, eye_idx, h, w)
                    if out is None:
                        continue
                    crop, (x1, y1, x2, y2) = out
                    p = _closed_prob(eye_model, crop)
                    closed_probs.append(p)
                    # green box = open, red box = closed
                    color = (0, 0, 255) if p > 0.5 else (0, 255, 0)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"{side}:{p:.2f}",
                                (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

                if closed_probs:
                    # require ALL detected eyes to be above threshold
                    # this prevents one misclassified eye from triggering false alerts
                    is_closed = all(p > 0.5 for p in closed_probs)
                    eye_states.append(1 if is_closed else 0)
                    if is_closed:
                        consec_closed_count += 1
                    else:
                        consec_closed_count = 0
                else:
                    consec_closed_count = 0
            else:
                # face turned sideways or gaze too far: reset the consecutive counter
                consec_closed_count = 0

            # calculate PERCLOS once we have at least 1 second of data
            if len(eye_states) >= PERCLOS_MIN_FRAMES:
                perclos = sum(eye_states) / len(eye_states)

            # yawn detection using the full face crop
            if yawn_model is not None:
                face = _face_crop(frame, lm, h, w)
                if face is not None:
                    p_yawn = _yawn_prob(yawn_model, face)
                    yawn_frames.append(1 if p_yawn > 0.5 else 0)
                    cv2.putText(frame, f"yawn:{p_yawn:.2f}",
                                (w - 130, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (0, 165, 255), 1)

                # detect the moment a yawn starts (rising edge)
                if len(yawn_frames) == YAWN_FRAME_WINDOW:
                    cur_yawn = (sum(yawn_frames) / YAWN_FRAME_WINDOW) > YAWN_FRAME_THRESH
                    was_yawn = yawn_active
                    yawn_active = cur_yawn
                    now = time.time()
                    # only count as a new event if enough time has passed since the last one
                    if cur_yawn and not was_yawn and (now - last_yawn_event_time) >= YAWN_COOLDOWN:
                        yawn_event_times.append(now)
                        last_yawn_event_time = now

                # remove yawn events older than 2 minutes
                now = time.time()
                yawn_event_times = [t for t in yawn_event_times
                                    if now - t <= YAWN_EVENT_WINDOW]
                yawn_events_1m = len(yawn_event_times)


        else:
            # no face found: reset the consecutive eye closed counter
            consec_closed_count = 0

        # decide alert level
        perclos_alert = perclos >= PERCLOS_THRESHOLD
        yawn_alert    = yawn_events_1m >= YAWN_EVENT_THRESH
        consec_alert  = consec_closed_count >= CONSEC_CLOSED_FRAMES

        active_count = sum([perclos_alert, yawn_alert, head_alert])
        # PERCLOS alone or eyes closed for 2 seconds straight = DANGER
        # two or more other indicators together = also DANGER
        is_drowsy  = perclos_alert or consec_alert or active_count >= 2
        is_warning = (not is_drowsy) and active_count >= 1

        if is_drowsy:
            alert_mgr.update("danger")
        elif is_warning:
            alert_mgr.update("warning")
        else:
            alert_mgr.update("none")

        # draw the three indicator lines at the bottom of the frame
        _draw_indicator(frame, "PERCLOS",
                        f"{perclos:.0%} ({len(eye_states)} frames)",
                        perclos_alert, h - 80)
        _draw_indicator(frame, "YAWN",
                        f"{yawn_events_1m} events/2min",
                        yawn_alert, h - 50)
        _draw_indicator(frame, "HEAD",
                        f"pitch={pitch:+.1f}deg",
                        head_alert, h - 20)
        cv2.putText(frame, f"FPS: {fps:.0f}",
                    (w - 90, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

        # show the alert banner at the top
        if is_drowsy:
            cv2.rectangle(frame, (0, 0), (w, 65), (0, 0, 200), -1)
            cv2.putText(frame, "DANGER: Driver is Drowsy",
                        (w // 2 - 195, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
        elif is_warning:
            cv2.rectangle(frame, (0, 0), (w, 65), (0, 140, 255), -1)
            cv2.putText(frame, "WARNING: Drowsiness Signs Detected",
                        (w // 2 - 270, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        else:
            cv2.rectangle(frame, (0, 0), (w, 65), (0, 160, 0), -1)
            cv2.putText(frame, "Driver is Alert",
                        (w // 2 - 120, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        # check if the window was closed before drawing the next frame
        try:
            if cv2.getWindowProperty("Drowsiness Detection", cv2.WND_PROP_VISIBLE) < 1:
                break
        except cv2.error:
            break
        cv2.imshow("Drowsiness Detection", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            break

    face_mesh.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eye-model",  type=Path, default=DEFAULT_EYE_MODEL)
    parser.add_argument("--yawn-model", type=Path, default=DEFAULT_YAWN_MODEL)
    parser.add_argument("--cam",        type=int,  default=0)
    parser.add_argument("--no-yawn",    action="store_true",
                        help="Disable yawn detection.")
    args = parser.parse_args()

    run_demo(
        eye_model_path  = args.eye_model,
        yawn_model_path = None if args.no_yawn else args.yawn_model,
        cam_index       = args.cam,
    )
