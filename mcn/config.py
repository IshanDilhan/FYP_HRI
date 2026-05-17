"""
MCN Configuration
=================
Central configuration for all hyperparameters, categorical vocabularies,
scenario definitions, and behavioral policy mappings.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


# ──────────────────────────────────────────────────────────────────────
# 1. CATEGORICAL VOCABULARIES (merged from report + system prompt)
# ──────────────────────────────────────────────────────────────────────

CONTEXT_CATEGORIES: List[str] = [
    "Classroom", "Hospital", "Museum", "Retail Store", "Clinic",
    "Open Lobby", "Narrow Hallway", "Office", "Kitchen",
]

EMOTION_CATEGORIES: List[str] = [
    "Happy", "Sad", "Angry", "Panicked", "Neutral",
    "Confused", "Hostile", "Fearful", "Surprised",
]

GESTURE_CATEGORIES: List[str] = [
    "One Hand Up", "Hand Waving", "Pointing", "Open Palm",
    "Stop Signal", "No Gesture", "Beckoning", "Reaching",
    "Arms Up", "Arms Waving", "Brief Wave",
]

MOTION_CATEGORIES: List[str] = [
    "Sitting", "Walking", "Running", "Stationary", "Approaching",
    "Leaving", "Passing Across", "Backing Away", "Minimal",
    "Slow Approach", "Fast Across", "Fast Toward Robot",
    "Approach And Stop",
]

# Intent / scenario classes the MCN classifies into
INTENT_CATEGORIES: List[str] = [
    "HELP_REQUEST",
    "NEUTRAL_PASS",
    "GIVE_WAY",
    "EMERGENCY",
    "GREETING",
    "TASK_ASSIST",
    "HOSTILE_CONFRONTATION",
    "DISTRESSED_STUDENT_QUERY",
    "UNKNOWN",
]

# Conflict / dissonance types detected by the auxiliary head
CONFLICT_TYPES: List[str] = [
    "NO_CONFLICT",
    "EMOTION_GESTURE_CONFLICT",
    "EMOTION_MOTION_CONFLICT",
    "GESTURE_MOTION_CONFLICT",
    "MULTI_CONFLICT",
]


# ──────────────────────────────────────────────────────────────────────
# 2. VOCABULARY INDEX MAPS (auto-generated)
# ──────────────────────────────────────────────────────────────────────

def _build_vocab(categories: List[str]) -> Dict[str, int]:
    """Build a string → index mapping with a special <UNK> token at 0."""
    vocab = {"<UNK>": 0}
    for i, cat in enumerate(categories, start=1):
        vocab[cat] = i
    return vocab


CONTEXT_VOCAB: Dict[str, int] = _build_vocab(CONTEXT_CATEGORIES)
EMOTION_VOCAB: Dict[str, int] = _build_vocab(EMOTION_CATEGORIES)
GESTURE_VOCAB: Dict[str, int] = _build_vocab(GESTURE_CATEGORIES)
MOTION_VOCAB: Dict[str, int] = _build_vocab(MOTION_CATEGORIES)
INTENT_VOCAB: Dict[str, int] = _build_vocab(INTENT_CATEGORIES)
CONFLICT_VOCAB: Dict[str, int] = _build_vocab(CONFLICT_TYPES)

# Reverse maps (index → string)
CONTEXT_IDX2STR: Dict[int, str] = {v: k for k, v in CONTEXT_VOCAB.items()}
EMOTION_IDX2STR: Dict[int, str] = {v: k for k, v in EMOTION_VOCAB.items()}
GESTURE_IDX2STR: Dict[int, str] = {v: k for k, v in GESTURE_VOCAB.items()}
MOTION_IDX2STR: Dict[int, str] = {v: k for k, v in MOTION_VOCAB.items()}
INTENT_IDX2STR: Dict[int, str] = {v: k for k, v in INTENT_VOCAB.items()}


# ──────────────────────────────────────────────────────────────────────
# 3. MODEL HYPERPARAMETERS
# ──────────────────────────────────────────────────────────────────────

@dataclass
class MCNConfig:
    """Hyperparameters for the Multimodal Cross-Modal Network."""

    # --- Embedding ---
    d_emb: int = 32               # Categorical embedding dimension per modality
    d_conf_proj: int = 32         # Confidence scalar projection dimension
    d_model: int = 64             # Token dimension (d_emb + d_conf_proj)

    # --- Transformer ---
    n_heads: int = 4              # Multi-head attention heads
    n_layers: int = 3             # Transformer encoder layers
    d_ff: int = 128               # Feed-forward hidden dimension (2× d_model)
    dropout: float = 0.1          # Dropout rate

    # --- Temporal ---
    window_size: int = 12         # Sliding window frames (≈1.2s at 10Hz)
    n_modalities: int = 4         # Number of upstream modalities (C, E, G, M)
    seq_len: int = 48             # window_size × n_modalities

    # --- Classification ---
    n_intents: int = len(INTENT_CATEGORIES)  # Number of intent classes
    n_conflicts: int = len(CONFLICT_TYPES)   # Number of conflict types

    # --- Vocabulary sizes (including <UNK>) ---
    context_vocab_size: int = len(CONTEXT_VOCAB)
    emotion_vocab_size: int = len(EMOTION_VOCAB)
    gesture_vocab_size: int = len(GESTURE_VOCAB)
    motion_vocab_size: int = len(MOTION_VOCAB)

    # --- Training ---
    learning_rate: float = 1e-3
    batch_size: int = 64
    epochs: int = 100
    alpha_intent: float = 1.0     # Weight for intent classification loss
    beta_conflict: float = 0.3    # Weight for conflict detection loss
    gamma_confidence: float = 0.2 # Weight for confidence regression loss

    # --- Inference ---
    input_hz: int = 10            # Input frame rate from upstream models
    confidence_threshold: float = 0.5  # Minimum intent probability to act

    def __post_init__(self):
        assert self.d_model == self.d_emb + self.d_conf_proj, \
            f"d_model ({self.d_model}) must equal d_emb ({self.d_emb}) + d_conf_proj ({self.d_conf_proj})"
        assert self.seq_len == self.window_size * self.n_modalities, \
            f"seq_len ({self.seq_len}) must equal window_size × n_modalities"
