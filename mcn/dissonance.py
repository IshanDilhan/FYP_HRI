"""
Dissonance Resolution Module
=============================
Detects cross-modal conflicts (e.g., "Angry face + Waving hand")
and treats them as informative features rather than errors.

This is an auxiliary head that runs in parallel with the intent classifier.
The detected conflict type is concatenated with the pooled features before
final intent classification, allowing the model to learn that certain
conflict patterns map to specific intents (e.g., Angry + Wave → HELP_REQUEST).
"""

import torch
import torch.nn as nn

from .config import MCNConfig, CONFLICT_TYPES


class DissonanceDetector(nn.Module):
    """
    Auxiliary module that classifies the type of cross-modal dissonance
    present in the fused representation.

    Architecture:
        pooled_features (d_model) → FC → ReLU → FC → conflict_logits (n_conflicts)

    The conflict logits are then used in two ways:
    1. Auxiliary training loss (supervised with ground-truth conflict labels)
    2. Concatenated with pooled features for the main intent classifier
    """

    def __init__(self, config: MCNConfig):
        super().__init__()
        self.config = config

        self.conflict_classifier = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, config.n_conflicts),
        )

        # Soft conflict embedding: project conflict logits back into feature space
        # so the intent classifier can leverage conflict information
        self.conflict_embedding = nn.Linear(config.n_conflicts, config.d_model)

    def forward(
        self, pooled_features: torch.Tensor
    ) -> tuple:
        """
        Args:
            pooled_features: (B, d_model) — globally pooled transformer output.

        Returns:
            conflict_logits:   (B, n_conflicts)  — raw logits for conflict classification.
            conflict_enriched: (B, d_model)      — features enriched with conflict info.
        """
        # Classify the conflict type
        conflict_logits = self.conflict_classifier(pooled_features)  # (B, n_conflicts)

        # Soft conflict feature (use softmax to get a distribution)
        conflict_probs = torch.softmax(conflict_logits, dim=-1)      # (B, n_conflicts)
        conflict_feat = self.conflict_embedding(conflict_probs)      # (B, d_model)

        # Enrich pooled features with conflict information (residual addition)
        conflict_enriched = pooled_features + conflict_feat          # (B, d_model)

        return conflict_logits, conflict_enriched


class DissonanceLabeler:
    """
    Utility to auto-generate ground-truth conflict labels from raw
    input modality states. Used during training data preparation.

    Rules:
    - EMOTION_GESTURE_CONFLICT: Hostile/Angry emotion with friendly gesture
      (Wave, Open Palm, Beckoning) or Happy/Neutral emotion with threatening
      gesture (Stop Signal).
    - EMOTION_MOTION_CONFLICT: Happy/Neutral emotion with Running/Approaching
      rapidly, or Panicked/Hostile emotion with Stationary/Sitting.
    - GESTURE_MOTION_CONFLICT: Beckoning/Waving with Leaving/Passing Across,
      or Stop Signal with Approaching.
    - MULTI_CONFLICT: Two or more of the above conditions hold simultaneously.
    - NO_CONFLICT: None of the above.
    """

    # Define which emotion-gesture pairs are considered conflicting
    NEGATIVE_EMOTIONS = {"Angry", "Hostile", "Panicked", "Fearful"}
    POSITIVE_EMOTIONS = {"Happy", "Neutral"}
    FRIENDLY_GESTURES = {"Hand Waving", "Open Palm", "Beckoning", "Brief Wave", "One Hand Up"}
    THREATENING_GESTURES = {"Stop Signal", "Pointing"}
    APPROACH_MOTIONS = {"Running", "Approaching", "Fast Toward Robot"}
    RETREAT_MOTIONS = {"Leaving", "Passing Across", "Backing Away", "Fast Across"}

    @classmethod
    def label(
        cls,
        emotion: str,
        gesture: str,
        motion: str,
    ) -> str:
        """
        Determine the conflict type from raw modality states.

        Returns:
            One of CONFLICT_TYPES strings.
        """
        conflicts = []

        # Check emotion-gesture conflict
        eg_conflict = (
            (emotion in cls.NEGATIVE_EMOTIONS and gesture in cls.FRIENDLY_GESTURES) or
            (emotion in cls.POSITIVE_EMOTIONS and gesture in cls.THREATENING_GESTURES)
        )
        if eg_conflict:
            conflicts.append("EMOTION_GESTURE_CONFLICT")

        # Check emotion-motion conflict
        em_conflict = (
            (emotion in cls.POSITIVE_EMOTIONS and motion in cls.APPROACH_MOTIONS) or
            (emotion in cls.NEGATIVE_EMOTIONS and motion in {"Stationary", "Sitting"})
        )
        # Note: Angry + Stationary could be a valid state (simmering anger)
        # but we still flag it as a weak conflict for the model to learn from
        if em_conflict:
            conflicts.append("EMOTION_MOTION_CONFLICT")

        # Check gesture-motion conflict
        gm_conflict = (
            (gesture in cls.FRIENDLY_GESTURES and motion in cls.RETREAT_MOTIONS) or
            (gesture in cls.THREATENING_GESTURES and motion in cls.APPROACH_MOTIONS)
        )
        if gm_conflict:
            conflicts.append("GESTURE_MOTION_CONFLICT")

        if len(conflicts) >= 2:
            return "MULTI_CONFLICT"
        elif len(conflicts) == 1:
            return conflicts[0]
        else:
            return "NO_CONFLICT"
