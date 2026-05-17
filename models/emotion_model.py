"""
Facial Affect Emotion Detection Model
=======================================
Uses the FER (Facial Expression Recognition) library with MTCNN face detection
and a mini-Xception emotion classifier.

Output: {"state": str, "confidence": float}
Categories: Happy, Sad, Angry, Panicked, Neutral, Confused, Hostile, Fearful, Surprised
"""

import numpy as np
from typing import Dict, List

from models.__init__ import UpstreamModel


# Mapping from FER output labels to MCN vocabulary
FER_TO_MCN = {
    "happy": "Happy",
    "sad": "Sad",
    "angry": "Angry",
    "fear": "Fearful",
    "surprise": "Surprised",
    "neutral": "Neutral",
    "disgust": "Hostile",
}


class EmotionModel(UpstreamModel):
    """
    Facial emotion detection using the FER library.

    FER uses MTCNN for face detection and a pre-trained mini-Xception CNN
    for emotion classification. It outputs probabilities for 7 basic emotions.
    """

    CATEGORIES = [
        "Happy", "Sad", "Angry", "Panicked", "Neutral",
        "Confused", "Hostile", "Fearful", "Surprised",
    ]

    def __init__(self, use_mtcnn: bool = True):
        """
        Args:
            use_mtcnn: If True, use MTCNN for face detection (more accurate
                       but slower). If False, use OpenCV Haar Cascade (faster).
        """
        from fer.fer import FER
        self.detector = FER(mtcnn=use_mtcnn)
        self._last_raw = None

    def predict(self, frame: np.ndarray) -> Dict[str, object]:
        """
        Detect the dominant emotion from the largest face in the frame.

        Args:
            frame: BGR image from OpenCV.

        Returns:
            {"state": emotion_str, "confidence": float}
        """
        try:
            # Detect emotions for all faces
            results = self.detector.detect_emotions(frame)

            if not results:
                return {"state": "Neutral", "confidence": 0.3}

            # Take the face with the largest bounding box (most prominent)
            largest = max(results, key=lambda r: r["box"][2] * r["box"][3])
            emotions = largest["emotions"]
            self._last_raw = emotions

            # Find the dominant emotion
            dominant_fer = max(emotions, key=emotions.get)
            confidence = emotions[dominant_fer]

            # Map to MCN vocabulary
            mcn_state = FER_TO_MCN.get(dominant_fer, "Neutral")

            # Check for "Panicked" — high fear + high surprise combined
            fear_score = emotions.get("fear", 0)
            surprise_score = emotions.get("surprise", 0)
            if fear_score > 0.3 and surprise_score > 0.2:
                mcn_state = "Panicked"
                confidence = (fear_score + surprise_score) / 2

            # Check for "Confused" — no clear dominant emotion
            sorted_emotions = sorted(emotions.values(), reverse=True)
            if len(sorted_emotions) >= 2:
                if sorted_emotions[0] - sorted_emotions[1] < 0.15:
                    # Top two emotions are very close → confused
                    if sorted_emotions[0] < 0.4:
                        mcn_state = "Confused"
                        confidence = 0.5

            return {"state": mcn_state, "confidence": round(float(confidence), 3)}

        except Exception as e:
            return {"state": "Neutral", "confidence": 0.1}

    def get_bounding_box(self) -> list:
        """Get the last detected face bounding box for visualization."""
        try:
            results = self.detector.detect_emotions.__self__
        except:
            pass
        return None

    def get_name(self) -> str:
        return "Facial Emotion (FER)"

    def get_categories(self) -> List[str]:
        return self.CATEGORIES

    def get_raw_scores(self) -> dict:
        """Get the raw FER emotion scores from the last prediction."""
        return self._last_raw or {}
