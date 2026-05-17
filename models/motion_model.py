"""
Body Motion Recognition Model
===============================
Uses MediaPipe Pose for 33-landmark body tracking, then classifies
motion state from velocity vectors, pose angles, and bounding box dynamics.

Output: {"state": str, "confidence": float}
Categories: Sitting, Walking, Running, Stationary, Approaching, Leaving,
            Passing Across, Backing Away, Minimal, Slow Approach,
            Fast Across, Fast Toward Robot, Approach And Stop
"""

import numpy as np
import math
from collections import deque
from typing import Dict, List, Optional, Tuple

from models.__init__ import UpstreamModel


class MotionModel(UpstreamModel):
    """
    Body motion recognition using MediaPipe Pose + velocity analysis.

    Pipeline:
        1. MediaPipe Pose extracts 33 body landmarks
        2. Hip centroid velocity is computed across frames
        3. Pose angles determine sitting vs standing
        4. Bounding box dynamics determine approaching vs leaving
        5. Movement direction determines passing across vs approaching
    """

    CATEGORIES = [
        "Sitting", "Walking", "Running", "Stationary", "Approaching",
        "Leaving", "Passing Across", "Backing Away", "Minimal",
        "Slow Approach", "Fast Across", "Fast Toward Robot",
        "Approach And Stop",
    ]

    # MediaPipe Pose landmark indices
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    NOSE = 0

    def __init__(self, min_detection_confidence: float = 0.5):
        import mediapipe as mp
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=0.5,
        )

        # Temporal buffers for velocity and bbox tracking
        self._hip_history: deque = deque(maxlen=15)
        self._bbox_history: deque = deque(maxlen=15)
        self._velocity_history: deque = deque(maxlen=10)
        self._last_landmarks = None

    def predict(self, frame: np.ndarray) -> Dict[str, object]:
        """Detect body motion state from a BGR frame."""
        import cv2

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)

        if not results.pose_landmarks:
            self._hip_history.append(None)
            self._bbox_history.append(None)
            return {"state": "Minimal", "confidence": 0.4}

        h, w = frame.shape[:2]
        lm = results.pose_landmarks.landmark
        self._last_landmarks = lm

        # Extract key positions
        hip_center = self._get_hip_center(lm, w, h)
        bbox_area = self._get_bbox_area(lm, w, h)

        self._hip_history.append(hip_center)
        self._bbox_history.append(bbox_area)

        # Compute velocity
        velocity = self._compute_velocity()
        self._velocity_history.append(velocity)

        # Compute pose features
        is_sitting = self._check_sitting(lm, h)
        bbox_trend = self._compute_bbox_trend()
        move_direction = self._compute_direction()
        avg_velocity = self._average_velocity()

        # Classify motion
        motion, confidence = self._classify_motion(
            avg_velocity, is_sitting, bbox_trend, move_direction
        )

        return {"state": motion, "confidence": round(float(confidence), 3)}

    def _get_hip_center(self, lm, w: int, h: int) -> Tuple[float, float]:
        """Get the center point between left and right hips."""
        lh = lm[self.LEFT_HIP]
        rh = lm[self.RIGHT_HIP]
        return ((lh.x + rh.x) / 2 * w, (lh.y + rh.y) / 2 * h)

    def _get_bbox_area(self, lm, w: int, h: int) -> float:
        """Compute approximate body bounding box area from landmarks."""
        xs = [lm[i].x * w for i in range(33) if lm[i].visibility > 0.5]
        ys = [lm[i].y * h for i in range(33) if lm[i].visibility > 0.5]
        if len(xs) < 4:
            return 0
        return (max(xs) - min(xs)) * (max(ys) - min(ys))

    def _compute_velocity(self) -> float:
        """Compute instantaneous velocity from hip centroid displacement."""
        positions = [p for p in self._hip_history if p is not None]
        if len(positions) < 2:
            return 0.0
        dx = positions[-1][0] - positions[-2][0]
        dy = positions[-1][1] - positions[-2][1]
        return math.sqrt(dx * dx + dy * dy)

    def _average_velocity(self) -> float:
        """Average velocity over recent frames."""
        vels = list(self._velocity_history)
        if not vels:
            return 0.0
        return sum(vels) / len(vels)

    def _compute_bbox_trend(self) -> float:
        """
        Compute bounding box size trend over time.
        Positive = getting larger (approaching), Negative = getting smaller (leaving).
        """
        bboxes = [b for b in self._bbox_history if b is not None and b > 0]
        if len(bboxes) < 5:
            return 0.0
        # Linear trend: compare recent average to older average
        mid = len(bboxes) // 2
        old_avg = sum(bboxes[:mid]) / mid
        new_avg = sum(bboxes[mid:]) / (len(bboxes) - mid)
        if old_avg == 0:
            return 0.0
        return (new_avg - old_avg) / old_avg  # Fractional change

    def _compute_direction(self) -> str:
        """Determine primary movement direction."""
        positions = [p for p in self._hip_history if p is not None]
        if len(positions) < 5:
            return "unknown"

        # Compare start and end positions
        dx = positions[-1][0] - positions[-3][0]
        dy = positions[-1][1] - positions[-3][1]

        abs_dx = abs(dx)
        abs_dy = abs(dy)

        if abs_dx < 3 and abs_dy < 3:
            return "stationary"
        elif abs_dx > abs_dy * 1.5:
            return "horizontal"  # Passing across
        elif abs_dy > abs_dx * 1.5:
            return "vertical"    # Up/down in frame
        else:
            return "diagonal"

    def _check_sitting(self, lm, h: int) -> bool:
        """
        Detect sitting posture from hip-knee-ankle angles.
        When sitting, the knee angle is typically < 120 degrees.
        """
        try:
            # Left side angle
            hip = np.array([lm[self.LEFT_HIP].x, lm[self.LEFT_HIP].y])
            knee = np.array([lm[self.LEFT_KNEE].x, lm[self.LEFT_KNEE].y])
            ankle = np.array([lm[self.LEFT_ANKLE].x, lm[self.LEFT_ANKLE].y])

            # Vectors
            v1 = hip - knee
            v2 = ankle - knee

            # Angle between hip-knee and knee-ankle
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
            angle = math.degrees(math.acos(np.clip(cos_angle, -1, 1)))

            # Also check if hip Y is close to knee Y (sitting characteristic)
            hip_knee_y_diff = abs(lm[self.LEFT_HIP].y - lm[self.LEFT_KNEE].y) * h

            return angle < 130 and hip_knee_y_diff < h * 0.1
        except:
            return False

    def _classify_motion(
        self,
        velocity: float,
        is_sitting: bool,
        bbox_trend: float,
        direction: str,
    ) -> Tuple[str, float]:
        """Classify the motion state from computed features."""

        # Priority 1: Sitting
        if is_sitting and velocity < 5:
            return ("Sitting", 0.88)

        # Priority 2: Stationary
        if velocity < 3:
            if is_sitting:
                return ("Sitting", 0.85)
            return ("Stationary", 0.85)

        # Priority 3: High velocity = Running
        if velocity > 20:
            if bbox_trend > 0.05:
                return ("Fast Toward Robot", 0.85)
            elif direction == "horizontal":
                return ("Fast Across", 0.82)
            else:
                return ("Running", 0.80)

        # Priority 4: Medium velocity
        if velocity > 8:
            if bbox_trend > 0.08:
                return ("Approaching", 0.82)
            elif bbox_trend < -0.08:
                return ("Leaving", 0.80)
            elif direction == "horizontal":
                return ("Passing Across", 0.78)
            else:
                return ("Walking", 0.75)

        # Priority 5: Low velocity
        if velocity > 3:
            if bbox_trend > 0.05:
                return ("Slow Approach", 0.75)
            elif bbox_trend < -0.05:
                return ("Backing Away", 0.72)
            elif direction == "horizontal":
                return ("Passing Across", 0.70)
            else:
                return ("Walking", 0.68)

        # Check for Approach And Stop pattern
        vels = list(self._velocity_history)
        if len(vels) >= 5:
            recent = vels[-3:]
            older = vels[:-3]
            if older and sum(older)/len(older) > 8 and sum(recent)/len(recent) < 3:
                return ("Approach And Stop", 0.78)

        return ("Minimal", 0.5)

    def get_name(self) -> str:
        return "Body Motion (MediaPipe Pose)"

    def get_categories(self) -> List[str]:
        return self.CATEGORIES

    def get_landmarks(self):
        """Return last detected pose landmarks for visualization."""
        return self._last_landmarks
