"""
MCN Dataset
============
Dataset class for training the Multimodal Cross-Modal Network.

Each training sample is a temporal window of 12 frames, each containing
4 modality readings (category + confidence), labeled with:
  - Intent class (ground truth scenario)
  - Conflict type (auto-generated or manually labeled)
  - Intent confidence target
"""

import json
import random
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple

from .config import (
    MCNConfig,
    CONTEXT_VOCAB, EMOTION_VOCAB, GESTURE_VOCAB, MOTION_VOCAB,
    INTENT_VOCAB, CONFLICT_VOCAB,
    CONTEXT_CATEGORIES, EMOTION_CATEGORIES, GESTURE_CATEGORIES, MOTION_CATEGORIES,
)
from .dissonance import DissonanceLabeler


class MCNDataset(Dataset):
    """
    PyTorch Dataset for MCN training.

    Each sample is a dict:
    {
        "context_idx":  (W,) int64,
        "context_conf": (W,) float32,
        "emotion_idx":  (W,) int64,
        "emotion_conf": (W,) float32,
        "gesture_idx":  (W,) int64,
        "gesture_conf": (W,) float32,
        "motion_idx":   (W,) int64,
        "motion_conf":  (W,) float32,
        "intent_label": int64,          # ground-truth intent index
        "conflict_label": int64,        # ground-truth conflict index
        "confidence_target": float32,   # target intent confidence
    }
    """

    def __init__(
        self,
        data: List[Dict],
        config: MCNConfig = None,
        augment: bool = False,
    ):
        """
        Args:
            data: List of scenario dicts. Each dict has:
                - "frames": list of W frame dicts (each with 4 modality pairs)
                - "intent": str — ground-truth intent label
                - "confidence_target": float — target confidence (optional, default 1.0)
            config: MCNConfig instance.
            augment: Whether to apply data augmentation.
        """
        if config is None:
            config = MCNConfig()
        self.config = config
        self.data = data
        self.augment = augment

    def __len__(self) -> int:
        return len(self.data)

    def _encode(self, state: str, vocab: Dict[str, int]) -> int:
        return vocab.get(state, 0)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.data[idx]
        frames = sample["frames"]
        W = self.config.window_size

        # Pad or truncate frames to window_size
        if len(frames) < W:
            # Pad with zero-frames at the beginning
            pad_frame = {
                "environment_context": {"state": "<UNK>", "confidence": 0.0},
                "facial_affect_emotion": {"state": "<UNK>", "confidence": 0.0},
                "skeletal_hand_gesture": {"state": "<UNK>", "confidence": 0.0},
                "body_motion_vector": {"state": "<UNK>", "confidence": 0.0},
            }
            frames = [pad_frame] * (W - len(frames)) + frames
        elif len(frames) > W:
            frames = frames[-W:]

        # Encode each frame
        context_idx = []
        context_conf = []
        emotion_idx = []
        emotion_conf = []
        gesture_idx = []
        gesture_conf = []
        motion_idx = []
        motion_conf = []

        for frame in frames:
            c = frame["environment_context"]
            e = frame["facial_affect_emotion"]
            g = frame["skeletal_hand_gesture"]
            m = frame["body_motion_vector"]

            c_conf = float(c["confidence"])
            e_conf = float(e["confidence"])
            g_conf = float(g["confidence"])
            m_conf = float(m["confidence"])

            # Apply augmentation
            if self.augment:
                c_conf = self._jitter_confidence(c_conf)
                e_conf = self._jitter_confidence(e_conf)
                g_conf = self._jitter_confidence(g_conf)
                m_conf = self._jitter_confidence(m_conf)

            context_idx.append(self._encode(c["state"], CONTEXT_VOCAB))
            context_conf.append(c_conf)
            emotion_idx.append(self._encode(e["state"], EMOTION_VOCAB))
            emotion_conf.append(e_conf)
            gesture_idx.append(self._encode(g["state"], GESTURE_VOCAB))
            gesture_conf.append(g_conf)
            motion_idx.append(self._encode(m["state"], MOTION_VOCAB))
            motion_conf.append(m_conf)

        # Auto-label conflict from the last frame (most recent observation)
        last_frame = frames[-1]
        conflict_str = DissonanceLabeler.label(
            emotion=last_frame["facial_affect_emotion"]["state"],
            gesture=last_frame["skeletal_hand_gesture"]["state"],
            motion=last_frame["body_motion_vector"]["state"],
        )
        conflict_label = CONFLICT_VOCAB.get(conflict_str, 0)

        # Intent label
        intent_str = sample.get("intent", "UNKNOWN")
        intent_label = INTENT_VOCAB.get(intent_str, 0)

        # Confidence target
        conf_target = sample.get("confidence_target", 1.0)

        # Apply modality dropout augmentation (randomly zero out one modality)
        if self.augment and random.random() < 0.15:
            drop_mod = random.randint(0, 3)
            if drop_mod == 0:
                context_idx = [0] * W
                context_conf = [0.0] * W
            elif drop_mod == 1:
                emotion_idx = [0] * W
                emotion_conf = [0.0] * W
            elif drop_mod == 2:
                gesture_idx = [0] * W
                gesture_conf = [0.0] * W
            else:
                motion_idx = [0] * W
                motion_conf = [0.0] * W

        return {
            "context_idx": torch.tensor(context_idx, dtype=torch.long),
            "context_conf": torch.tensor(context_conf, dtype=torch.float32),
            "emotion_idx": torch.tensor(emotion_idx, dtype=torch.long),
            "emotion_conf": torch.tensor(emotion_conf, dtype=torch.float32),
            "gesture_idx": torch.tensor(gesture_idx, dtype=torch.long),
            "gesture_conf": torch.tensor(gesture_conf, dtype=torch.float32),
            "motion_idx": torch.tensor(motion_idx, dtype=torch.long),
            "motion_conf": torch.tensor(motion_conf, dtype=torch.float32),
            "intent_label": torch.tensor(intent_label, dtype=torch.long),
            "conflict_label": torch.tensor(conflict_label, dtype=torch.long),
            "confidence_target": torch.tensor(conf_target, dtype=torch.float32),
        }

    @staticmethod
    def _jitter_confidence(conf: float, noise: float = 0.1) -> float:
        """Add small random noise to confidence score for augmentation."""
        jittered = conf + random.uniform(-noise, noise)
        return max(0.0, min(1.0, jittered))


# ──────────────────────────────────────────────────────────────────────
# SYNTHETIC DATA GENERATOR (for bootstrapping training)
# ──────────────────────────────────────────────────────────────────────

class SyntheticDataGenerator:
    """
    Generates synthetic training samples from scenario definitions.
    This bridges the gap until Isaac Sim data or real-world data is available.
    """

    # Scenario templates matching HRI_Scenarios.pdf
    SCENARIO_TEMPLATES = [
        # === CLASSROOM ===
        {
            "id": "C1", "intent": "HELP_REQUEST",
            "context": "Classroom", "emotion": "Sad",
            "gesture": "One Hand Up", "motion": "Stationary",
        },
        {
            "id": "C2", "intent": "NEUTRAL_PASS",
            "context": "Classroom", "emotion": "Happy",
            "gesture": "Brief Wave", "motion": "Fast Across",
        },
        {
            "id": "C3", "intent": "GIVE_WAY",
            "context": "Classroom", "emotion": "Angry",
            "gesture": "Pointing", "motion": "Approaching",
        },
        {
            "id": "C4", "intent": "GIVE_WAY",
            "context": "Classroom", "emotion": "Neutral",
            "gesture": "No Gesture", "motion": "Slow Approach",
        },
        {
            "id": "C5", "intent": "EMERGENCY",
            "context": "Classroom", "emotion": "Fearful",
            "gesture": "Arms Waving", "motion": "Fast Toward Robot",
        },
        # === OFFICE ===
        {
            "id": "O1", "intent": "NEUTRAL_PASS",
            "context": "Office", "emotion": "Neutral",
            "gesture": "Brief Wave", "motion": "Fast Across",
        },
        {
            "id": "O2", "intent": "GIVE_WAY",
            "context": "Office", "emotion": "Angry",
            "gesture": "Pointing", "motion": "Approaching",
        },
        {
            "id": "O3", "intent": "GREETING",
            "context": "Office", "emotion": "Happy",
            "gesture": "Hand Waving", "motion": "Approach And Stop",
        },
        {
            "id": "O4", "intent": "HELP_REQUEST",
            "context": "Office", "emotion": "Sad",
            "gesture": "Beckoning", "motion": "Stationary",
        },
        {
            "id": "O5", "intent": "EMERGENCY",
            "context": "Office", "emotion": "Fearful",
            "gesture": "Reaching", "motion": "Minimal",
        },
        # === KITCHEN ===
        {
            "id": "K1", "intent": "HELP_REQUEST",
            "context": "Kitchen", "emotion": "Angry",
            "gesture": "Beckoning", "motion": "Stationary",
        },
        {
            "id": "K2", "intent": "EMERGENCY",
            "context": "Kitchen", "emotion": "Surprised",
            "gesture": "Arms Up", "motion": "Backing Away",
        },
        {
            "id": "K3", "intent": "TASK_ASSIST",
            "context": "Kitchen", "emotion": "Neutral",
            "gesture": "Pointing", "motion": "Stationary",
        },
        {
            "id": "K4", "intent": "GIVE_WAY",
            "context": "Kitchen", "emotion": "Neutral",
            "gesture": "No Gesture", "motion": "Approaching",
        },
        {
            "id": "K5", "intent": "NEUTRAL_PASS",
            "context": "Kitchen", "emotion": "Happy",
            "gesture": "No Gesture", "motion": "Passing Across",
        },
        # === ADDITIONAL SCENARIOS (from system prompt) ===
        {
            "id": "H1", "intent": "DISTRESSED_STUDENT_QUERY",
            "context": "Classroom", "emotion": "Sad",
            "gesture": "One Hand Up", "motion": "Sitting",
        },
        {
            "id": "H2", "intent": "EMERGENCY",
            "context": "Hospital", "emotion": "Panicked",
            "gesture": "Hand Waving", "motion": "Running",
        },
        {
            "id": "H3", "intent": "HOSTILE_CONFRONTATION",
            "context": "Clinic", "emotion": "Hostile",
            "gesture": "Pointing", "motion": "Approaching",
        },
    ]

    @classmethod
    def generate(
        cls,
        n_samples_per_scenario: int = 100,
        config: MCNConfig = None,
    ) -> List[Dict]:
        """
        Generate synthetic training data from scenario templates.

        Each scenario template is expanded into multiple training samples
        with:
        - Varying confidence scores
        - Temporal consistency (12 frames of similar readings)
        - Occasional mid-window transitions
        """
        if config is None:
            config = MCNConfig()

        all_samples = []

        for template in cls.SCENARIO_TEMPLATES:
            for _ in range(n_samples_per_scenario):
                frames = []
                for t in range(config.window_size):
                    # Base confidences with per-frame variation
                    base_conf = random.uniform(0.7, 0.99)
                    frame = {
                        "environment_context": {
                            "state": template["context"],
                            "confidence": cls._vary_conf(base_conf),
                        },
                        "facial_affect_emotion": {
                            "state": template["emotion"],
                            "confidence": cls._vary_conf(base_conf),
                        },
                        "skeletal_hand_gesture": {
                            "state": template["gesture"],
                            "confidence": cls._vary_conf(base_conf),
                        },
                        "body_motion_vector": {
                            "state": template["motion"],
                            "confidence": cls._vary_conf(base_conf),
                        },
                    }

                    # Occasionally inject noise in early frames (simulate transition)
                    if t < 3 and random.random() < 0.3:
                        # Replace one modality with random category
                        noise_mod = random.choice(["emotion", "gesture", "motion"])
                        if noise_mod == "emotion":
                            frame["facial_affect_emotion"]["state"] = random.choice(
                                EMOTION_CATEGORIES
                            )
                            frame["facial_affect_emotion"]["confidence"] = random.uniform(0.3, 0.6)
                        elif noise_mod == "gesture":
                            frame["skeletal_hand_gesture"]["state"] = random.choice(
                                GESTURE_CATEGORIES
                            )
                            frame["skeletal_hand_gesture"]["confidence"] = random.uniform(0.3, 0.6)
                        else:
                            frame["body_motion_vector"]["state"] = random.choice(
                                MOTION_CATEGORIES
                            )
                            frame["body_motion_vector"]["confidence"] = random.uniform(0.3, 0.6)

                    frames.append(frame)

                all_samples.append({
                    "frames": frames,
                    "intent": template["intent"],
                    "confidence_target": random.uniform(0.85, 1.0),
                    "scenario_id": template["id"],
                })

        random.shuffle(all_samples)
        return all_samples

    @staticmethod
    def _vary_conf(base: float, noise: float = 0.05) -> float:
        """Small per-frame confidence variation."""
        varied = base + random.uniform(-noise, noise)
        return max(0.0, min(1.0, round(varied, 3)))


def create_dataloaders(
    config: MCNConfig = None,
    n_samples_per_scenario: int = 100,
    val_split: float = 0.2,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    """
    Convenience function to create train/val dataloaders with synthetic data.

    Returns:
        (train_loader, val_loader)
    """
    if config is None:
        config = MCNConfig()

    all_data = SyntheticDataGenerator.generate(n_samples_per_scenario, config)
    n_val = int(len(all_data) * val_split)
    n_train = len(all_data) - n_val

    train_data = all_data[:n_train]
    val_data = all_data[n_train:]

    train_ds = MCNDataset(train_data, config, augment=True)
    val_ds = MCNDataset(val_data, config, augment=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader
