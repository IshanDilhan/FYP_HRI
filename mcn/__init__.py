"""
Multimodal Cross-Modal Network (MCN)
=====================================
Late-fusion transformer-based engine for adaptive Human-Robot Interaction.

Ingests 4 upstream vision model streams:
  - Environment Context (C)
  - Facial Affect Emotion (E)
  - Skeletal Hand Gesture (G)
  - Body Motion Recognition (M)

Applies self-attention cross-modal fusion with temporal sliding windows,
resolves behavioral dissonance, and outputs unified Scenario IDs mapped
to ROS2 behavioral policies.
"""

from .config import MCNConfig
from .model import MultimodalCrossModalNetwork
from .inference import MCNInferencePipeline
from .policy_mapper import PolicyMapper

__version__ = "0.1.0"
__all__ = [
    "MCNConfig",
    "MultimodalCrossModalNetwork",
    "MCNInferencePipeline",
    "PolicyMapper",
]
