"""
Multimodal Cross-Modal Network (MCN) — Core Model
===================================================
Late-fusion transformer encoder that processes 48 tokens
(12 frames × 4 modalities) through self-attention to decode
a unified intent classification.

Architecture:
    Input → MultiModalEmbedder → TransformerEncoder → GlobalAvgPool
         → DissonanceDetector → IntentClassifier → (intent_logits, confidence)
"""

import torch
import torch.nn as nn
import math

from .config import MCNConfig
from .embeddings import MultiModalEmbedder
from .dissonance import DissonanceDetector


class MCNTransformerBlock(nn.Module):
    """
    Single transformer encoder block with pre-norm architecture.

    Pre-norm (LayerNorm → Attention/FFN) is more stable for training
    small models compared to post-norm.
    """

    def __init__(self, config: MCNConfig):
        super().__init__()

        # Multi-head self-attention
        self.attn_norm = nn.LayerNorm(config.d_model)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=config.d_model,
            num_heads=config.n_heads,
            dropout=config.dropout,
            batch_first=True,
        )

        # Feed-forward network
        self.ff_norm = nn.LayerNorm(config.d_model)
        self.ffn = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_ff, config.d_model),
            nn.Dropout(config.dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, S, d_model) — input token sequence.
            attn_mask: Optional attention mask.

        Returns:
            (B, S, d_model) — transformed token sequence.
        """
        # Pre-norm self-attention with residual
        normed = self.attn_norm(x)
        attn_out, _ = self.self_attn(
            normed, normed, normed,
            attn_mask=attn_mask,
            need_weights=False,
        )
        x = x + attn_out

        # Pre-norm FFN with residual
        normed = self.ff_norm(x)
        ff_out = self.ffn(normed)
        x = x + ff_out

        return x


class MultimodalCrossModalNetwork(nn.Module):
    """
    The complete MCN model.

    Pipeline:
        1. MultiModalEmbedder: Encode 4 modalities → 48 tokens of dim 64
        2. TransformerEncoder:  3 layers of 4-head self-attention
        3. GlobalAvgPool:       Aggregate to single (B, d_model) vector
        4. DissonanceDetector:  Detect cross-modal conflicts
        5. IntentClassifier:    Classify into scenario/intent categories
        6. ConfidenceHead:      Predict intent probability (0-1)
    """

    def __init__(self, config: MCNConfig = None):
        super().__init__()
        if config is None:
            config = MCNConfig()
        self.config = config

        # 1. Multi-modal embedding
        self.embedder = MultiModalEmbedder(config)

        # 2. Transformer encoder stack
        self.encoder_layers = nn.ModuleList([
            MCNTransformerBlock(config) for _ in range(config.n_layers)
        ])
        self.final_norm = nn.LayerNorm(config.d_model)

        # 3. Dissonance detection (auxiliary head)
        self.dissonance = DissonanceDetector(config)

        # 4. Intent classification head
        self.intent_classifier = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, config.n_intents),
        )

        # 5. Confidence regression head (predicts intent probability)
        self.confidence_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            nn.GELU(),
            nn.Linear(config.d_model // 2, 1),
            nn.Sigmoid(),
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Xavier uniform initialization for better convergence on small models."""
        for name, param in self.named_parameters():
            if "weight" in name and param.dim() >= 2:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(
        self,
        context_idx: torch.Tensor,    # (B, W)
        context_conf: torch.Tensor,   # (B, W)
        emotion_idx: torch.Tensor,    # (B, W)
        emotion_conf: torch.Tensor,   # (B, W)
        gesture_idx: torch.Tensor,    # (B, W)
        gesture_conf: torch.Tensor,   # (B, W)
        motion_idx: torch.Tensor,     # (B, W)
        motion_conf: torch.Tensor,    # (B, W)
    ) -> dict:
        """
        Full forward pass.

        Args:
            *_idx:  Category indices.    (B, W), int64
            *_conf: Confidence scores.   (B, W), float32
            W = window_size = 12

        Returns:
            Dict with:
                "intent_logits":    (B, n_intents)  — raw logits for intent classification
                "intent_probs":     (B, n_intents)  — softmax probabilities
                "conflict_logits":  (B, n_conflicts) — raw logits for conflict detection
                "confidence":       (B, 1)          — predicted intent confidence [0,1]
        """
        # 1. Embed all modalities → (B, 48, 64)
        tokens = self.embedder(
            context_idx, context_conf,
            emotion_idx, emotion_conf,
            gesture_idx, gesture_conf,
            motion_idx, motion_conf,
        )

        # 2. Transformer encoder
        x = tokens
        for layer in self.encoder_layers:
            x = layer(x)
        x = self.final_norm(x)  # (B, 48, 64)

        # 3. Global average pooling over sequence dimension
        pooled = x.mean(dim=1)  # (B, 64)

        # 4. Dissonance detection
        conflict_logits, enriched = self.dissonance(pooled)  # (B, n_conflicts), (B, 64)

        # 5. Intent classification (using conflict-enriched features)
        intent_logits = self.intent_classifier(enriched)     # (B, n_intents)
        intent_probs = torch.softmax(intent_logits, dim=-1)  # (B, n_intents)

        # 6. Confidence prediction
        confidence = self.confidence_head(enriched)          # (B, 1)

        return {
            "intent_logits": intent_logits,
            "intent_probs": intent_probs,
            "conflict_logits": conflict_logits,
            "confidence": confidence,
        }
