"""
Skeletal Hand Gesture Recognition Model
=========================================
Uses MediaPipe Hands for 21-landmark hand detection, then classifies
gestures using geometric rules on landmark positions.

Output: {"state": str, "confidence": float}
Categories: One Hand Up, Hand Waving, Pointing, Open Palm, Stop Signal,
            No Gesture, Beckoning, Reaching, Arms Up, Arms Waving, Brief Wave
"""

import numpy as np
import math
from collections import deque
from typing import Dict, List, Optional, Tuple

from models.__init__ import UpstreamModel


class GestureModel(UpstreamModel):
    """
    Hand gesture recognition using MediaPipe Hands + geometric classification.

    Pipeline:
        1. MediaPipe detects up to 2 hands with 21 3D landmarks each
        2. Geometric rules classify the hand configuration into a gesture
        3. Temporal smoothing filters out flickering
    """

    CATEGORIES = [
        "One Hand Up", "Hand Waving", "Pointing", "Open Palm",
        "Stop Signal", "No Gesture", "Beckoning", "Reaching",
        "Arms Up", "Arms Waving", "Brief Wave",
    ]

    # MediaPipe landmark indices
    WRIST = 0
    THUMB_TIP = 4
    INDEX_TIP = 8
    MIDDLE_TIP = 12
    RING_TIP = 16
    PINKY_TIP = 20
    INDEX_MCP = 5
    MIDDLE_MCP = 9
    INDEX_PIP = 6
    MIDDLE_PIP = 10
    RING_PIP = 14
    PINKY_PIP = 18

    def __init__(self, max_hands: int = 2, min_detection_confidence: float = 0.5):
        import mediapipe as mp
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=0.5,
        )
        # Temporal buffer for waving detection
        self._wrist_history: deque = deque(maxlen=15)
        self._last_landmarks = None

    def predict(self, frame: np.ndarray) -> Dict[str, object]:
        """Detect hand gesture from a BGR frame."""
        import cv2

        # MediaPipe expects RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)

        if not results.multi_hand_landmarks:
            self._wrist_history.append(None)
            return {"state": "No Gesture", "confidence": 0.8}

        # Process the first detected hand
        hand = results.multi_hand_landmarks[0]
        h, w = frame.shape[:2]

        # Extract normalized landmarks to pixel coords
        lm = [(hand.landmark[i].x * w, hand.landmark[i].y * h, hand.landmark[i].z)
              for i in range(21)]
        self._last_landmarks = lm

        # Track wrist position for wave detection
        wrist_pos = (lm[self.WRIST][0], lm[self.WRIST][1])
        self._wrist_history.append(wrist_pos)

        # Classify gesture
        gesture, confidence = self._classify_gesture(lm, h, w)

        # Check for temporal gestures (waving)
        if self._is_waving():
            # Override with waving if detected
            wave_conf = self._wave_confidence()
            if wave_conf > confidence * 0.8:
                gesture = "Hand Waving"
                confidence = wave_conf

        return {"state": gesture, "confidence": round(float(confidence), 3)}

    def _classify_gesture(
        self, lm: list, h: int, w: int
    ) -> Tuple[str, float]:
        """Classify the static hand gesture from landmarks."""

        # Count extended fingers
        fingers_up = self._count_fingers_up(lm)

        # Wrist position relative to frame
        wrist_y_ratio = lm[self.WRIST][1] / h

        # Check specific gestures in priority order

        # 1. Pointing — only index finger extended
        if fingers_up == [False, True, False, False, False]:
            return ("Pointing", 0.88)

        # 2. Open Palm — all 5 fingers extended
        if all(fingers_up):
            # Distinguish between Open Palm and Stop Signal
            # Stop signal: arm more extended (wrist higher in frame)
            if wrist_y_ratio < 0.4:
                return ("Stop Signal", 0.85)
            elif wrist_y_ratio < 0.55:
                return ("Open Palm", 0.85)
            else:
                return ("Open Palm", 0.80)

        # 3. One Hand Up — hand detected above midpoint
        if wrist_y_ratio < 0.45 and sum(fingers_up) >= 2:
            return ("One Hand Up", 0.82)

        # 4. Beckoning — index+middle up, others down, hand below midpoint
        if fingers_up == [False, True, True, False, False]:
            return ("Beckoning", 0.78)

        # 5. Reaching — hand with fingers partially extended, low position
        if wrist_y_ratio > 0.6 and sum(fingers_up) >= 3:
            return ("Reaching", 0.75)

        # 6. Fist / ambiguous
        if sum(fingers_up) <= 1:
            if wrist_y_ratio < 0.5:
                return ("One Hand Up", 0.65)
            return ("No Gesture", 0.6)

        return ("No Gesture", 0.5)

    def _count_fingers_up(self, lm: list) -> List[bool]:
        """
        Determine which fingers are extended.
        Returns [thumb, index, middle, ring, pinky] as booleans.
        """
        fingers = []

        # Thumb: compare tip X to MCP X (works for right hand)
        # Use horizontal distance for thumb
        thumb_extended = lm[self.THUMB_TIP][0] < lm[self.THUMB_TIP - 2][0]
        fingers.append(thumb_extended)

        # Other fingers: tip Y < PIP Y means extended (image coords, Y increases downward)
        tips = [self.INDEX_TIP, self.MIDDLE_TIP, self.RING_TIP, self.PINKY_TIP]
        pips = [self.INDEX_PIP, self.MIDDLE_PIP, self.RING_PIP, self.PINKY_PIP]

        for tip, pip_idx in zip(tips, pips):
            fingers.append(lm[tip][1] < lm[pip_idx][1])

        return fingers

    def _is_waving(self) -> bool:
        """Detect waving motion from wrist position history."""
        positions = [p for p in self._wrist_history if p is not None]
        if len(positions) < 8:
            return False

        # Check for lateral oscillation in X coordinates
        x_positions = [p[0] for p in positions[-10:]]
        if len(x_positions) < 6:
            return False

        # Count direction changes (oscillations)
        direction_changes = 0
        for i in range(2, len(x_positions)):
            prev_dir = x_positions[i-1] - x_positions[i-2]
            curr_dir = x_positions[i] - x_positions[i-1]
            if prev_dir * curr_dir < 0 and abs(curr_dir) > 5:
                direction_changes += 1

        return direction_changes >= 3

    def _wave_confidence(self) -> float:
        """Estimate waving confidence from oscillation amplitude."""
        positions = [p for p in self._wrist_history if p is not None]
        if len(positions) < 5:
            return 0.0
        x_positions = [p[0] for p in positions[-10:]]
        amplitude = max(x_positions) - min(x_positions)
        # Normalize: good wave has amplitude > 50px
        conf = min(amplitude / 100.0, 1.0)
        return max(0.6, conf)

    def get_name(self) -> str:
        return "Hand Gesture (MediaPipe)"

    def get_categories(self) -> List[str]:
        return self.CATEGORIES

    def get_landmarks(self) -> Optional[list]:
        """Return last detected hand landmarks for visualization."""
        return self._last_landmarks
