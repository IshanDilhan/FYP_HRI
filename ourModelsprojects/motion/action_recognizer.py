"""
Basic Human Action / Motion Recognizer
=======================================
Uses:
  - MediaPipe Pose  → 33 body keypoints
  - ResNet50        → pretrained on Stanford40 (from HuggingFace)
  - Motion tracker  → velocity-based Stationary / Approaching / Backing away / etc.

Supports:
  - Webcam (real-time)
  - Video file input

Install dependencies:
    pip install mediapipe torch torchvision huggingface_hub opencv-python pillow numpy

Usage:
    # Webcam
    python action_recognizer.py --webcam

    # Video file
    python action_recognizer.py --video path/to/video.mp4

    # Video file + save output
    python action_recognizer.py --video path/to/video.mp4 --save output.avi
"""

import argparse
import sys
import time
from collections import deque

import cv2
import numpy as np
import mediapipe as mp
import mediapipe.python.solutions.pose as mp_pose_mod
import mediapipe.python.solutions.drawing_utils as mp_drawing_mod
import torch
from PIL import Image
from huggingface_hub import hf_hub_download
from torchvision import models, transforms

# ─────────────────────────────────────────────────────────────────────────────
# MOTION LABELS from your table
# ─────────────────────────────────────────────────────────────────────────────
MOTION_LABELS = [
    "Stationary",
    "Approaching",
    "Backing Away",
    "Fast Across",
    "Slow Approach",
    "Fast Toward",
    "Approach + Stop",
    "Minimal",
    "Across",
]

# Colour per label (BGR)
LABEL_COLORS = {
    "Stationary":     (180, 180, 180),
    "Approaching":    (0,   200, 255),
    "Backing Away":   (255, 100,   0),
    "Fast Across":    (0,   255, 100),
    "Slow Approach":  (50,  200, 255),
    "Fast Toward":    (0,    80, 255),
    "Approach + Stop":(200, 255,   0),
    "Minimal":        (160, 160, 160),
    "Across":         (0,   255, 200),
}

# ─────────────────────────────────────────────────────────────────────────────
# MOTION ANALYSER  (keypoint-velocity based, no GPU needed)
# ─────────────────────────────────────────────────────────────────────────────
class MotionAnalyser:
    """
    Tracks the centre-of-mass of the detected pose across frames and
    classifies the motion into one of the MOTION_LABELS.

    Axes (image coords):
        x  → left-right   (positive = moving RIGHT / "across")
        y  → up-down      (positive = moving DOWN  / "away" from cam in top-view)
        z  → MediaPipe depth estimate (positive = moving TOWARD camera)
    """

    HISTORY = 12          # frames to smooth over
    STILL_THR  = 0.004    # normalised units – below = Stationary / Minimal
    FAST_THR   = 0.018    # above = Fast
    DEPTH_THR  = 0.005    # z-velocity threshold for toward/away

    def __init__(self):
        self.history = deque(maxlen=self.HISTORY)
        self.stopped_at = None          # frame index when motion stopped

    def update(self, landmarks, frame_idx: int) -> str:
        """landmarks: mediapipe NormalizedLandmarkList or None"""
        if landmarks is None:
            return "Stationary"

        # Centre-of-mass from hips + shoulders
        pts = landmarks.landmark
        key_ids = [11, 12, 23, 24]     # left/right shoulder, left/right hip
        cx = np.mean([pts[i].x for i in key_ids])
        cy = np.mean([pts[i].y for i in key_ids])
        cz = np.mean([pts[i].z for i in key_ids])

        self.history.append((cx, cy, cz, frame_idx))

        if len(self.history) < 4:
            return "Stationary"

        # Velocity over last N frames
        old = self.history[0]
        new = self.history[-1]
        dt  = max(new[3] - old[3], 1)

        vx = (new[0] - old[0]) / dt
        vy = (new[1] - old[1]) / dt
        vz = (new[2] - old[2]) / dt   # negative = moving toward cam

        speed_xy = np.sqrt(vx**2 + vy**2)
        lateral   = abs(vx) > abs(vy) * 1.5   # mostly horizontal movement

        # ── Classification logic ──────────────────────────────────────────
        if speed_xy < self.STILL_THR and abs(vz) < self.DEPTH_THR:
            return "Stationary"

        if speed_xy < self.STILL_THR * 2:
            return "Minimal"

        # Toward / Away  (depth axis)
        depth_dominant = abs(vz) > speed_xy * 0.5

        if depth_dominant:
            if vz < -self.DEPTH_THR:                    # coming closer
                if speed_xy < self.STILL_THR * 1.5:
                    # Almost stopped while approaching → Approach + Stop
                    if self.stopped_at and (frame_idx - self.stopped_at) < 20:
                        return "Approach + Stop"
                    self.stopped_at = frame_idx
                    return "Slow Approach"
                return "Fast Toward" if speed_xy > self.FAST_THR else "Approaching"
            else:                                        # moving away
                return "Backing Away"

        # Lateral movement
        if lateral:
            return "Fast Across" if speed_xy > self.FAST_THR else "Across"

        # General approach
        if vy > self.STILL_THR:                         # moving down in frame
            return "Approaching"

        return "Stationary"


# ─────────────────────────────────────────────────────────────────────────────
# POSE CLASSIFIER  (MediaPipe → 33 keypoints → simple rule-based pose)
# ─────────────────────────────────────────────────────────────────────────────
class PoseClassifier:
    """
    Classifies the body pose (standing / sitting / lying / crouching)
    purely from keypoint geometry — no model weights needed.
    """

    def classify(self, landmarks) -> str:
        if landmarks is None:
            return "Unknown"
        lm = landmarks.landmark

        # Vertical span: nose (0) vs mid-hip (23/24)
        nose_y     = lm[0].y
        hip_y      = (lm[23].y + lm[24].y) / 2
        knee_y     = (lm[25].y + lm[26].y) / 2
        ankle_y    = (lm[27].y + lm[28].y) / 2
        shoulder_y = (lm[11].y + lm[12].y) / 2

        body_height = abs(ankle_y - nose_y) + 1e-6
        torso_ratio = abs(hip_y - shoulder_y) / body_height
        leg_ratio   = abs(ankle_y - knee_y)  / body_height

        if body_height < 0.25:
            return "Lying"
        if torso_ratio > 0.35 and leg_ratio < 0.15:
            return "Sitting"
        if knee_y < hip_y + 0.05:
            return "Crouching"
        return "Standing"


# ─────────────────────────────────────────────────────────────────────────────
# OPTIONAL: HuggingFace ResNet50 (Stanford40) for action label
# ─────────────────────────────────────────────────────────────────────────────
STANFORD40_CLASSES = [
    "applauding","blowing_bubbles","brushing_teeth","cleaning_floor",
    "climbing","cooking","cutting_trees","cutting_vegetables","drinking",
    "feeding_a_horse","fishing","fixing_a_bike","fixing_a_car","gardening",
    "holding_an_umbrella","jumping","looking_through_a_microscope",
    "looking_through_a_telescope","milking_a_cow","moving_furniture",
    "phoning","photographing","playing_guitar","playing_violin",
    "pouring_liquid","pushing_a_cart","reading","riding_a_bike",
    "riding_a_horse","rowing_a_boat","running","shooting_an_arrow",
    "smoking","taking_photos","texting_message","throwing_frisby",
    "using_a_computer","walking_the_dog","washing_dishes","watching_TV",
]

class ActionModel:
    def __init__(self):
        self.model = None
        self.transform = None
        self._loaded = False

    def load(self):
        if self._loaded:
            return
        try:
            import torch
            from torchvision import transforms, models
            from huggingface_hub import hf_hub_download
            import os

            print("[INFO] Downloading ResNet50 weights from HuggingFace …")
            weight_path = hf_hub_download(
                repo_id="dronefreak/human-action-classification-stanford40",
                filename="resnet50/model.pth",
            )
            model = models.resnet50(weights=None)
            model.fc = torch.nn.Linear(model.fc.in_features, 40)
            state = torch.load(weight_path, map_location="cpu")
            # handle various checkpoint formats
            if isinstance(state, dict) and "model_state_dict" in state:
                state = state["model_state_dict"]
            elif isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            model.load_state_dict(state, strict=False)
            model.eval()

            self.model = model
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225]),
            ])
            self._loaded = True
            print("[INFO] ResNet50 loaded successfully.")
        except Exception as e:
            print(f"[WARN] Could not load ResNet50: {e}")
            print("[WARN] Falling back to pose+motion classification only.")
            self._loaded = True   # don't retry

    def predict(self, frame_rgb) -> str | None:
        if self.model is None:
            return None
        try:
            import torch
            tensor = self.transform(frame_rgb).unsqueeze(0)
            with torch.no_grad():
                out = self.model(tensor)
            idx = out.argmax(dim=1).item()
            return STANFORD40_CLASSES[idx].replace("_", " ").title()
        except Exception:
            return None


# ─────────────────────────────────────────────────────────────────────────────
# OVERLAY DRAWING
# ─────────────────────────────────────────────────────────────────────────────
def draw_overlay(frame, pose_label, motion_label, action_label,
                 fps, landmarks, mp_drawing, mp_pose):
    h, w = frame.shape[:2]
    color = LABEL_COLORS.get(motion_label, (255, 255, 255))

    # Draw skeleton
    if landmarks:
        mp_drawing.draw_landmarks(
            frame, landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
            mp_drawing.DrawingSpec(color=(0, 180, 255), thickness=2),
        )

    # Semi-transparent info panel
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (320, 130), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    def txt(text, y, scale=0.6, col=(255,255,255), bold=False):
        th = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, text, (10, y), th, scale, (0,0,0), 4)
        cv2.putText(frame, text, (10, y), th, scale, col,
                    2 if bold else 1)

    txt(f"FPS: {fps:.1f}", 22, 0.55)
    txt(f"Pose:   {pose_label}", 48, 0.6)
    txt(f"Motion: {motion_label}", 76, 0.7, color, bold=True)
    if action_label:
        txt(f"Action: {action_label}", 104, 0.55, (200, 255, 200))

    # Motion badge (bottom-right)
    label_w = max(len(motion_label) * 14 + 20, 180)
    cv2.rectangle(frame, (w - label_w, h - 48), (w, h), color, -1)
    cv2.putText(frame, motion_label,
                (w - label_w + 10, h - 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (20, 20, 20), 2)

    return frame


# ─────────────────────────────────────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────────────────────────────────────
def run(source, save_path=None, use_action_model=False):
    mp_pose    = mp_pose_mod
    mp_drawing = mp_drawing_mod

    motion_analyser  = MotionAnalyser()
    pose_classifier  = PoseClassifier()
    action_model     = ActionModel()

    if use_action_model:
        action_model.load()

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open source: {source}")
        sys.exit(1)

    fps_cam = cap.get(cv2.CAP_PROP_FPS) or 30
    writer  = None

    if save_path:
        fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(save_path, fourcc, fps_cam, (fw, fh))
        print(f"[INFO] Saving output to: {save_path}")

    print("[INFO] Press  Q  to quit.")

    frame_idx  = 0
    t_prev     = time.time()
    fps_display = 0.0
    action_label = None
    ACTION_INTERVAL = 10   # run ResNet every N frames

    with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=1,
    ) as pose:

        while True:
            ret, frame = cap.read()
            if not ret:
                if isinstance(source, str):   # end of video file
                    break
                continue

            frame_idx += 1
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = pose.process(rgb)
            rgb.flags.writeable = True

            landmarks = results.pose_landmarks

            # ── Classify ──────────────────────────────────────────────────
            motion_label = motion_analyser.update(landmarks, frame_idx)
            pose_label   = pose_classifier.classify(landmarks)

            if use_action_model and frame_idx % ACTION_INTERVAL == 0:
                action_label = action_model.predict(rgb)

            # ── FPS ───────────────────────────────────────────────────────
            now = time.time()
            fps_display = 0.9 * fps_display + 0.1 * (1.0 / max(now - t_prev, 1e-6))
            t_prev = now

            # ── Draw ──────────────────────────────────────────────────────
            frame = draw_overlay(
                frame, pose_label, motion_label,
                action_label, fps_display,
                landmarks, mp_drawing, mp_pose,
            )

            if writer:
                writer.write(frame)

            cv2.imshow("Human Action Recognizer  [Q = quit]", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    print("[INFO] Done.")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Basic Human Motion Recognizer (MediaPipe + ResNet50)"
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--webcam", action="store_true",
                     help="Use default webcam (index 0)")
    src.add_argument("--video",  type=str, metavar="PATH",
                     help="Path to a video file")
    src.add_argument("--camera", type=int, metavar="INDEX",
                     help="Camera device index (e.g. 1 for external cam)")

    parser.add_argument("--save",  type=str, metavar="OUT",
                        help="Save annotated output to this file (e.g. out.avi)")
    parser.add_argument("--action-model", action="store_true",
                        help="Also run ResNet50 Stanford40 action model (slower)")

    args = parser.parse_args()

    if args.webcam:
        source = 0
    elif args.camera is not None:
        source = args.camera
    else:
        source = args.video

    run(source, save_path=args.save, use_action_model=args.action_model)
