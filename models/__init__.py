"""
Upstream Model Base Interface
==============================
Abstract base class that all 4 upstream models must implement.
Ensures consistent output format for the MCN fusion engine.
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional
import numpy as np


class UpstreamModel(ABC):
    """
    Base class for upstream feature extraction models.

    Each model processes a video frame and returns a dict:
        {"state": str, "confidence": float}

    where `state` is a categorical label from the model's vocabulary
    and `confidence` is a softmax probability in [0.0, 1.0].
    """

    @abstractmethod
    def predict(self, frame: np.ndarray) -> Dict[str, object]:
        """
        Process a single BGR video frame and return the prediction.

        Args:
            frame: OpenCV BGR image, shape (H, W, 3), dtype uint8.

        Returns:
            {"state": str, "confidence": float}
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Return the human-readable model name."""
        pass

    @abstractmethod
    def get_categories(self) -> list:
        """Return the list of categorical labels this model can output."""
        pass
