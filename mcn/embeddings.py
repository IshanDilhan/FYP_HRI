"""
Modality Embedding Layers
=========================
Converts each upstream modality's (categorical_state, confidence_score) pair
into a dense vector of dimension d_model.

Each modality token = [categorical_embedding ⊕ confidence_projection]
The confidence score gates the embedding magnitude, implementing the
"dynamic modality re-weighting" requirement from the system specification.
"""

import torch
import torch.nn as nn

from .config import MCNConfig


class ModalityEmbedding(nn.Module):
    """
    Embeds a single modality's categorical label + confidence score
    into a d_model-dimensional token.

    Args:
        vocab_size: Number of categories (including <UNK>).
        d_emb: Embedding dimension for the categorical label.
        d_conf_proj: Projection dimension for the confidence scalar.
    """

    def __init__(self, vocab_size: int, d_emb: int, d_conf_proj: int):
        super().__init__()
        self.categorical_emb = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_emb,
            padding_idx=0,  # <UNK> gets zero vector
        )
        # Project scalar confidence (1D) → d_conf_proj dimensions
        self.confidence_proj = nn.Sequential(
            nn.Linear(1, d_conf_proj),
            nn.GELU(),
        )
        # Confidence-aware gating: scale embedding by confidence
        self.gate = nn.Sequential(
            nn.Linear(1, d_emb),
            nn.Sigmoid(),
        )

    def forward(
        self,
        category_idx: torch.Tensor,  # (batch, seq)  int64
        confidence: torch.Tensor,     # (batch, seq)  float32
    ) -> torch.Tensor:
        """
        Args:
            category_idx: Integer indices into the vocabulary.  Shape (B, T).
            confidence:   Softmax confidence scores [0, 1].    Shape (B, T).

        Returns:
            Token embeddings of shape (B, T, d_model) where
            d_model = d_emb + d_conf_proj.
        """
        # (B, T, d_emb)
        cat_emb = self.categorical_emb(category_idx)

        # Confidence gating — low-confidence inputs get dampened
        conf_unsqueezed = confidence.unsqueeze(-1)  # (B, T, 1)
        gate_weights = self.gate(conf_unsqueezed)    # (B, T, d_emb)
        gated_emb = cat_emb * gate_weights           # (B, T, d_emb)

        # Confidence projection
        conf_proj = self.confidence_proj(conf_unsqueezed)  # (B, T, d_conf_proj)

        # Concatenate: [gated_categorical ⊕ confidence_projection]
        token = torch.cat([gated_emb, conf_proj], dim=-1)  # (B, T, d_model)
        return token


class MultiModalEmbedder(nn.Module):
    """
    Embeds all 4 modalities and adds positional encodings for both
    temporal position (frame index) and modality type.

    Input:  4 × (category_idx, confidence) pairs per frame, over a window.
    Output: (B, seq_len, d_model) tensor ready for the transformer.
    """

    def __init__(self, config: MCNConfig):
        super().__init__()
        self.config = config

        # Per-modality embedding layers
        self.context_emb = ModalityEmbedding(
            config.context_vocab_size, config.d_emb, config.d_conf_proj
        )
        self.emotion_emb = ModalityEmbedding(
            config.emotion_vocab_size, config.d_emb, config.d_conf_proj
        )
        self.gesture_emb = ModalityEmbedding(
            config.gesture_vocab_size, config.d_emb, config.d_conf_proj
        )
        self.motion_emb = ModalityEmbedding(
            config.motion_vocab_size, config.d_emb, config.d_conf_proj
        )

        # Learnable positional encoding: temporal position (0..window_size-1)
        self.temporal_pos_emb = nn.Embedding(config.window_size, config.d_model)

        # Learnable modality-type encoding (0=Context, 1=Emotion, 2=Gesture, 3=Motion)
        self.modality_type_emb = nn.Embedding(config.n_modalities, config.d_model)

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
    ) -> torch.Tensor:
        """
        Args:
            *_idx:  Integer category indices.    Shape (B, W) where W = window_size.
            *_conf: Confidence scores [0, 1].    Shape (B, W).

        Returns:
            Fused token sequence of shape (B, seq_len, d_model)
            where seq_len = W × 4 modalities = 48.
        """
        B, W = context_idx.shape
        device = context_idx.device

        # Embed each modality: (B, W, d_model)
        c_tokens = self.context_emb(context_idx, context_conf)
        e_tokens = self.emotion_emb(emotion_idx, emotion_conf)
        g_tokens = self.gesture_emb(gesture_idx, gesture_conf)
        m_tokens = self.motion_emb(motion_idx, motion_conf)

        # Temporal position indices: [0, 1, ..., W-1]
        t_pos = torch.arange(W, device=device).unsqueeze(0).expand(B, -1)  # (B, W)
        t_pos_enc = self.temporal_pos_emb(t_pos)  # (B, W, d_model)

        # Modality type indices
        mod_ids = torch.arange(self.config.n_modalities, device=device)  # (4,)

        # Add positional + modality encodings to each modality
        c_tokens = c_tokens + t_pos_enc + self.modality_type_emb(mod_ids[0])
        e_tokens = e_tokens + t_pos_enc + self.modality_type_emb(mod_ids[1])
        g_tokens = g_tokens + t_pos_enc + self.modality_type_emb(mod_ids[2])
        m_tokens = m_tokens + t_pos_enc + self.modality_type_emb(mod_ids[3])

        # Interleave: for each frame, place [C, E, G, M] tokens sequentially
        # Stack: (B, W, 4, d_model) → reshape to (B, W*4, d_model)
        stacked = torch.stack([c_tokens, e_tokens, g_tokens, m_tokens], dim=2)
        fused_seq = stacked.reshape(B, W * self.config.n_modalities, self.config.d_model)

        return fused_seq  # (B, 48, 64)
