"""
Temporal Sliding Window Buffer
==============================
Maintains a FIFO buffer of the most recent `window_size` frames (default 12,
≈1.2 seconds at 10Hz). This provides the temporal context needed for the
transformer to trace feature trajectories and filter out instantaneous
frame-skipping noise or flickering anomalies.
"""

import torch
from collections import deque
from typing import Dict, Optional, Tuple

from .config import (
    MCNConfig,
    CONTEXT_VOCAB,
    EMOTION_VOCAB,
    GESTURE_VOCAB,
    MOTION_VOCAB,
)


class TemporalSlidingWindow:
    """
    FIFO buffer that accumulates per-frame modality readings and outputs
    a batched tensor window for the MCN transformer.

    Usage:
        window = TemporalSlidingWindow(config)
        for frame in stream:
            window.push(frame)
            if window.is_ready():
                inputs = window.get_tensor()
                # feed inputs to MCN model
    """

    def __init__(self, config: MCNConfig):
        self.config = config
        self.window_size = config.window_size

        # Each element in the deque is a dict with 8 values:
        #   context_idx, context_conf, emotion_idx, emotion_conf,
        #   gesture_idx, gesture_conf, motion_idx, motion_conf
        self._buffer: deque = deque(maxlen=self.window_size)

    def _encode_category(
        self, state: str, vocab: Dict[str, int]
    ) -> int:
        """Map a categorical string to its vocabulary index, defaulting to <UNK>=0."""
        return vocab.get(state, 0)

    def push(self, frame: Dict) -> None:
        """
        Add a new frame to the sliding window.

        Args:
            frame: Dict matching the input_packet schema:
                {
                    "environment_context": {"state": str, "confidence": float},
                    "facial_affect_emotion": {"state": str, "confidence": float},
                    "skeletal_hand_gesture": {"state": str, "confidence": float},
                    "body_motion_vector": {"state": str, "confidence": float},
                }
        """
        encoded = {
            "context_idx": self._encode_category(
                frame["environment_context"]["state"], CONTEXT_VOCAB
            ),
            "context_conf": float(frame["environment_context"]["confidence"]),
            "emotion_idx": self._encode_category(
                frame["facial_affect_emotion"]["state"], EMOTION_VOCAB
            ),
            "emotion_conf": float(frame["facial_affect_emotion"]["confidence"]),
            "gesture_idx": self._encode_category(
                frame["skeletal_hand_gesture"]["state"], GESTURE_VOCAB
            ),
            "gesture_conf": float(frame["skeletal_hand_gesture"]["confidence"]),
            "motion_idx": self._encode_category(
                frame["body_motion_vector"]["state"], MOTION_VOCAB
            ),
            "motion_conf": float(frame["body_motion_vector"]["confidence"]),
        }
        self._buffer.append(encoded)

    def is_ready(self) -> bool:
        """Returns True when the buffer has accumulated `window_size` frames."""
        return len(self._buffer) >= self.window_size

    def current_length(self) -> int:
        """Number of frames currently in the buffer."""
        return len(self._buffer)

    def clear(self) -> None:
        """Reset the buffer."""
        self._buffer.clear()

    def get_tensor(
        self, device: Optional[torch.device] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Convert the current window buffer into batched tensors suitable
        for the MultiModalEmbedder.

        Returns:
            Dict with keys matching MultiModalEmbedder.forward() args:
                context_idx:  (1, W)  int64
                context_conf: (1, W)  float32
                emotion_idx:  (1, W)  int64
                emotion_conf: (1, W)  float32
                gesture_idx:  (1, W)  int64
                gesture_conf: (1, W)  float32
                motion_idx:   (1, W)  int64
                motion_conf:  (1, W)  float32

        Raises:
            RuntimeError: If the buffer doesn't have enough frames yet.
        """
        if not self.is_ready():
            raise RuntimeError(
                f"Window not ready: have {len(self._buffer)}/{self.window_size} frames. "
                f"Call is_ready() before get_tensor()."
            )

        frames = list(self._buffer)

        # Build per-field lists
        context_idx = [f["context_idx"] for f in frames]
        context_conf = [f["context_conf"] for f in frames]
        emotion_idx = [f["emotion_idx"] for f in frames]
        emotion_conf = [f["emotion_conf"] for f in frames]
        gesture_idx = [f["gesture_idx"] for f in frames]
        gesture_conf = [f["gesture_conf"] for f in frames]
        motion_idx = [f["motion_idx"] for f in frames]
        motion_conf = [f["motion_conf"] for f in frames]

        # Convert to tensors with batch dim = 1
        result = {
            "context_idx": torch.tensor([context_idx], dtype=torch.long),
            "context_conf": torch.tensor([context_conf], dtype=torch.float32),
            "emotion_idx": torch.tensor([emotion_idx], dtype=torch.long),
            "emotion_conf": torch.tensor([emotion_conf], dtype=torch.float32),
            "gesture_idx": torch.tensor([gesture_idx], dtype=torch.long),
            "gesture_conf": torch.tensor([gesture_conf], dtype=torch.float32),
            "motion_idx": torch.tensor([motion_idx], dtype=torch.long),
            "motion_conf": torch.tensor([motion_conf], dtype=torch.float32),
        }

        if device is not None:
            result = {k: v.to(device) for k, v in result.items()}

        return result

    def get_padded_tensor(
        self, device: Optional[torch.device] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Like get_tensor(), but pads with zeros if the buffer isn't full yet.
        This allows inference before the window is completely filled (warm-up).

        Returns:
            Same schema as get_tensor(), with leading zeros for missing frames.
        """
        n_have = len(self._buffer)
        n_pad = self.window_size - n_have

        frames = list(self._buffer)

        # Pad with zero-frames
        zero_frame = {
            "context_idx": 0, "context_conf": 0.0,
            "emotion_idx": 0, "emotion_conf": 0.0,
            "gesture_idx": 0, "gesture_conf": 0.0,
            "motion_idx": 0, "motion_conf": 0.0,
        }
        padded_frames = [zero_frame] * n_pad + frames

        context_idx = [f["context_idx"] for f in padded_frames]
        context_conf = [f["context_conf"] for f in padded_frames]
        emotion_idx = [f["emotion_idx"] for f in padded_frames]
        emotion_conf = [f["emotion_conf"] for f in padded_frames]
        gesture_idx = [f["gesture_idx"] for f in padded_frames]
        gesture_conf = [f["gesture_conf"] for f in padded_frames]
        motion_idx = [f["motion_idx"] for f in padded_frames]
        motion_conf = [f["motion_conf"] for f in padded_frames]

        result = {
            "context_idx": torch.tensor([context_idx], dtype=torch.long),
            "context_conf": torch.tensor([context_conf], dtype=torch.float32),
            "emotion_idx": torch.tensor([emotion_idx], dtype=torch.long),
            "emotion_conf": torch.tensor([emotion_conf], dtype=torch.float32),
            "gesture_idx": torch.tensor([gesture_idx], dtype=torch.long),
            "gesture_conf": torch.tensor([gesture_conf], dtype=torch.float32),
            "motion_idx": torch.tensor([motion_idx], dtype=torch.long),
            "motion_conf": torch.tensor([motion_conf], dtype=torch.float32),
        }

        if device is not None:
            result = {k: v.to(device) for k, v in result.items()}

        return result
