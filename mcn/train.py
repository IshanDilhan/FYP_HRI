"""
MCN Training Loop
=================
Multi-task training with:
  - L_intent:     Cross-entropy loss for intent/scenario classification
  - L_conflict:   Cross-entropy loss for conflict type detection
  - L_confidence: MSE loss for intent confidence regression

L_total = α·L_intent + β·L_conflict + γ·L_confidence
"""

import os
import time
import json
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import Dict, Optional, Tuple

from .config import MCNConfig
from .model import MultimodalCrossModalNetwork
from .dataset import create_dataloaders


class MCNTrainer:
    """
    Trainer for the Multimodal Cross-Modal Network.

    Usage:
        config = MCNConfig()
        trainer = MCNTrainer(config)
        trainer.train()
    """

    def __init__(
        self,
        config: MCNConfig = None,
        save_dir: str = "checkpoints",
        device: str = None,
    ):
        if config is None:
            config = MCNConfig()
        self.config = config
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        # Auto-detect device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Model
        self.model = MultimodalCrossModalNetwork(config).to(self.device)
        print(f"[MCN] Model parameters: {self.model.count_parameters():,}")
        print(f"[MCN] Device: {self.device}")

        # Losses
        self.intent_loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.conflict_loss_fn = nn.CrossEntropyLoss()
        self.confidence_loss_fn = nn.MSELoss()

        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=1e-4,
        )

        # LR scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config.epochs,
            eta_min=1e-6,
        )

        # Training history
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_intent_acc": [],
            "val_intent_acc": [],
            "train_conflict_acc": [],
            "val_conflict_acc": [],
        }
        self.best_val_loss = float("inf")

    def _compute_loss(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute the multi-task loss."""
        intent_loss = self.intent_loss_fn(
            outputs["intent_logits"],
            batch["intent_label"].to(self.device),
        )
        conflict_loss = self.conflict_loss_fn(
            outputs["conflict_logits"],
            batch["conflict_label"].to(self.device),
        )
        confidence_loss = self.confidence_loss_fn(
            outputs["confidence"].squeeze(-1),
            batch["confidence_target"].to(self.device),
        )

        total_loss = (
            self.config.alpha_intent * intent_loss
            + self.config.beta_conflict * conflict_loss
            + self.config.gamma_confidence * confidence_loss
        )

        metrics = {
            "total": total_loss.item(),
            "intent": intent_loss.item(),
            "conflict": conflict_loss.item(),
            "confidence": confidence_loss.item(),
        }
        return total_loss, metrics

    def _compute_accuracy(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[float, float]:
        """Compute intent and conflict classification accuracy."""
        # Intent accuracy
        intent_preds = outputs["intent_logits"].argmax(dim=-1)
        intent_correct = (intent_preds == batch["intent_label"].to(self.device)).float().mean()

        # Conflict accuracy
        conflict_preds = outputs["conflict_logits"].argmax(dim=-1)
        conflict_correct = (conflict_preds == batch["conflict_label"].to(self.device)).float().mean()

        return intent_correct.item(), conflict_correct.item()

    def _forward_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Move batch to device and run forward pass."""
        return self.model(
            context_idx=batch["context_idx"].to(self.device),
            context_conf=batch["context_conf"].to(self.device),
            emotion_idx=batch["emotion_idx"].to(self.device),
            emotion_conf=batch["emotion_conf"].to(self.device),
            gesture_idx=batch["gesture_idx"].to(self.device),
            gesture_conf=batch["gesture_conf"].to(self.device),
            motion_idx=batch["motion_idx"].to(self.device),
            motion_conf=batch["motion_conf"].to(self.device),
        )

    def train_epoch(self, train_loader) -> Dict[str, float]:
        """Run one training epoch."""
        self.model.train()
        epoch_metrics = {"total": 0, "intent": 0, "conflict": 0, "confidence": 0}
        intent_acc_sum = 0.0
        conflict_acc_sum = 0.0
        n_batches = 0

        for batch in train_loader:
            self.optimizer.zero_grad()

            outputs = self._forward_batch(batch)
            loss, metrics = self._compute_loss(outputs, batch)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            # Accumulate metrics
            for k in epoch_metrics:
                epoch_metrics[k] += metrics[k]
            i_acc, c_acc = self._compute_accuracy(outputs, batch)
            intent_acc_sum += i_acc
            conflict_acc_sum += c_acc
            n_batches += 1

        # Average
        for k in epoch_metrics:
            epoch_metrics[k] /= n_batches
        epoch_metrics["intent_acc"] = intent_acc_sum / n_batches
        epoch_metrics["conflict_acc"] = conflict_acc_sum / n_batches

        return epoch_metrics

    @torch.no_grad()
    def validate(self, val_loader) -> Dict[str, float]:
        """Run validation."""
        self.model.eval()
        epoch_metrics = {"total": 0, "intent": 0, "conflict": 0, "confidence": 0}
        intent_acc_sum = 0.0
        conflict_acc_sum = 0.0
        n_batches = 0

        for batch in val_loader:
            outputs = self._forward_batch(batch)
            _, metrics = self._compute_loss(outputs, batch)

            for k in epoch_metrics:
                epoch_metrics[k] += metrics[k]
            i_acc, c_acc = self._compute_accuracy(outputs, batch)
            intent_acc_sum += i_acc
            conflict_acc_sum += c_acc
            n_batches += 1

        for k in epoch_metrics:
            epoch_metrics[k] /= n_batches
        epoch_metrics["intent_acc"] = intent_acc_sum / n_batches
        epoch_metrics["conflict_acc"] = conflict_acc_sum / n_batches

        return epoch_metrics

    def train(
        self,
        n_samples_per_scenario: int = 200,
        val_split: float = 0.2,
    ):
        """
        Full training loop.

        Args:
            n_samples_per_scenario: Synthetic samples per scenario template.
            val_split: Fraction of data for validation.
        """
        print(f"[MCN] Generating synthetic training data...")
        train_loader, val_loader = create_dataloaders(
            self.config,
            n_samples_per_scenario=n_samples_per_scenario,
            val_split=val_split,
        )
        print(f"[MCN] Train samples: {len(train_loader.dataset)}, "
              f"Val samples: {len(val_loader.dataset)}")

        print(f"\n[MCN] Starting training for {self.config.epochs} epochs...")
        print("-" * 85)
        print(f"{'Epoch':>5} | {'Train Loss':>10} | {'Val Loss':>10} | "
              f"{'Train Acc':>10} | {'Val Acc':>10} | {'LR':>10} | {'Time':>6}")
        print("-" * 85)

        for epoch in range(1, self.config.epochs + 1):
            t0 = time.time()

            train_metrics = self.train_epoch(train_loader)
            val_metrics = self.validate(val_loader)

            self.scheduler.step()
            lr = self.scheduler.get_last_lr()[0]

            elapsed = time.time() - t0

            # Log
            self.history["train_loss"].append(train_metrics["total"])
            self.history["val_loss"].append(val_metrics["total"])
            self.history["train_intent_acc"].append(train_metrics["intent_acc"])
            self.history["val_intent_acc"].append(val_metrics["intent_acc"])
            self.history["train_conflict_acc"].append(train_metrics["conflict_acc"])
            self.history["val_conflict_acc"].append(val_metrics["conflict_acc"])

            print(
                f"{epoch:>5} | {train_metrics['total']:>10.4f} | {val_metrics['total']:>10.4f} | "
                f"{train_metrics['intent_acc']:>9.1%} | {val_metrics['intent_acc']:>9.1%} | "
                f"{lr:>10.6f} | {elapsed:>5.1f}s"
            )

            # Save best model
            if val_metrics["total"] < self.best_val_loss:
                self.best_val_loss = val_metrics["total"]
                self.save_checkpoint(os.path.join(self.save_dir, "best_model.pt"))

            # Save every 10 epochs
            if epoch % 10 == 0:
                self.save_checkpoint(
                    os.path.join(self.save_dir, f"checkpoint_epoch_{epoch}.pt")
                )

        print("-" * 85)
        print(f"[MCN] Training complete. Best val loss: {self.best_val_loss:.4f}")

        # Save final model and history
        self.save_checkpoint(os.path.join(self.save_dir, "final_model.pt"))
        self._save_history()

        return self.history

    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config,
            "best_val_loss": self.best_val_loss,
        }, path)

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        print(f"[MCN] Loaded checkpoint from {path}")

    def _save_history(self):
        """Save training history to JSON."""
        history_path = os.path.join(self.save_dir, "training_history.json")
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=2)
        print(f"[MCN] Training history saved to {history_path}")


def main():
    """CLI entry point for training."""
    config = MCNConfig(
        epochs=100,
        batch_size=64,
        learning_rate=1e-3,
    )
    trainer = MCNTrainer(config, save_dir="d:/FYP_Tranformer/checkpoints")
    trainer.train(n_samples_per_scenario=200)


if __name__ == "__main__":
    main()
