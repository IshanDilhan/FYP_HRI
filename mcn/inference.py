"""
MCN Real-Time Inference Pipeline
=================================
End-to-end inference pipeline that:
  1. Accepts raw frame packets from upstream models
  2. Buffers them in the temporal sliding window
  3. Runs the MCN transformer model
  4. Decodes the intent and maps to a ROS2 behavioral policy JSON

Designed for 10Hz real-time operation on NVIDIA Jetson Orin Nano.
"""

import time
import json
import torch
from typing import Dict, Optional

from .config import MCNConfig, INTENT_IDX2STR, CONTEXT_IDX2STR, CONTEXT_VOCAB
from .model import MultimodalCrossModalNetwork
from .temporal_window import TemporalSlidingWindow
from .policy_mapper import PolicyMapper


class MCNInferencePipeline:
    """
    Real-time inference pipeline for the Multimodal Cross-Modal Network.

    Usage:
        pipeline = MCNInferencePipeline.from_checkpoint("checkpoints/best_model.pt")

        # Process frames from upstream models
        for frame_packet in upstream_stream:
            result = pipeline.process_frame(frame_packet)
            if result:
                print(json.dumps(result, indent=2))
    """

    def __init__(
        self,
        model: MultimodalCrossModalNetwork,
        config: MCNConfig = None,
        device: str = None,
    ):
        if config is None:
            config = MCNConfig()
        self.config = config

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = model.to(self.device)
        self.model.eval()

        self.window = TemporalSlidingWindow(config)
        self.policy_mapper = PolicyMapper()

        self._frame_counter = 0
        self._last_inference_time = 0.0

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        device: str = None,
    ) -> "MCNInferencePipeline":
        """Load the pipeline from a saved checkpoint."""
        if device is None:
            device_obj = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device_obj = torch.device(device)

        checkpoint = torch.load(checkpoint_path, map_location=device_obj)
        config = checkpoint.get("config", MCNConfig())

        model = MultimodalCrossModalNetwork(config)
        model.load_state_dict(checkpoint["model_state_dict"])

        pipeline = cls(model, config, device)
        print(f"[MCN] Loaded model from {checkpoint_path}")
        print(f"[MCN] Parameters: {model.count_parameters():,}")
        print(f"[MCN] Device: {device_obj}")
        return pipeline

    @torch.no_grad()
    def process_frame(
        self,
        frame_packet: Dict,
        allow_warmup: bool = True,
    ) -> Optional[Dict]:
        """
        Process a single frame packet from upstream models.

        Args:
            frame_packet: Dict matching the input_packet schema:
                {
                    "environment_context": {"state": str, "confidence": float},
                    "facial_affect_emotion": {"state": str, "confidence": float},
                    "skeletal_hand_gesture": {"state": str, "confidence": float},
                    "body_motion_vector": {"state": str, "confidence": float},
                }
            allow_warmup: If True, run inference even before the window is full
                         (using padded tensors). If False, returns None until
                         the window has 12 frames.

        Returns:
            Policy JSON dict if inference is performed, None otherwise.
        """
        t_start = time.perf_counter()
        self._frame_counter += 1

        # Push frame into sliding window
        self.window.push(frame_packet)

        # Check if we can run inference
        if self.window.is_ready():
            inputs = self.window.get_tensor(device=self.device)
        elif allow_warmup and self.window.current_length() > 0:
            inputs = self.window.get_padded_tensor(device=self.device)
        else:
            return None

        # Run model forward pass
        outputs = self.model(**inputs)

        # Decode intent
        intent_probs = outputs["intent_probs"][0]  # (n_intents,)
        intent_idx = intent_probs.argmax().item()
        intent_prob = intent_probs[intent_idx].item()
        intent_label = INTENT_IDX2STR.get(intent_idx, "UNKNOWN")

        # Get predicted confidence
        model_confidence = outputs["confidence"][0, 0].item()

        # Use the higher of model confidence and softmax probability
        final_confidence = max(intent_prob, model_confidence)

        # Get current context for policy adaptation
        context_state = frame_packet["environment_context"]["state"]

        # Map to behavioral policy
        policy = self.policy_mapper.map(
            frame_id=self._frame_counter,
            intent_label=intent_label,
            intent_probability=final_confidence,
            context_label=context_state,
        )

        # Add timing info
        t_end = time.perf_counter()
        self._last_inference_time = (t_end - t_start) * 1000  # ms
        policy["inference_time_ms"] = round(self._last_inference_time, 2)
        policy["window_frames"] = self.window.current_length()

        return policy

    def process_input_packet(self, input_json: Dict) -> Optional[Dict]:
        """
        Process a full input_packet JSON (as defined in the system spec).

        Args:
            input_json: Dict with "input_packet" key containing the frame.

        Returns:
            Policy JSON dict.
        """
        frame = input_json.get("input_packet", input_json)
        return self.process_frame(frame)

    def reset(self):
        """Reset the pipeline state (clear window buffer)."""
        self.window.clear()
        self._frame_counter = 0

    def get_stats(self) -> Dict:
        """Get pipeline statistics."""
        return {
            "frames_processed": self._frame_counter,
            "window_length": self.window.current_length(),
            "window_ready": self.window.is_ready(),
            "last_inference_ms": round(self._last_inference_time, 2),
            "device": str(self.device),
            "model_params": self.model.count_parameters(),
        }


# ──────────────────────────────────────────────────────────────────────
# STANDALONE DEMO
# ──────────────────────────────────────────────────────────────────────

def demo():
    """
    Demo: Run inference on the example input from the system prompt.
    Uses a randomly initialized model (untrained) for demonstration.
    """
    config = MCNConfig()
    model = MultimodalCrossModalNetwork(config)
    pipeline = MCNInferencePipeline(model, config)

    print("[MCN Demo] Running inference with untrained model...")
    print("[MCN Demo] (Intent predictions will be random until trained)\n")

    # Example input from system prompt
    example_frame = {
        "environment_context": {"state": "Classroom", "confidence": 0.94},
        "facial_affect_emotion": {"state": "Sad", "confidence": 0.89},
        "skeletal_hand_gesture": {"state": "One Hand Up", "confidence": 0.93},
        "body_motion_vector": {"state": "Sitting", "confidence": 0.97},
    }

    # Feed 12 identical frames to fill the window
    result = None
    for i in range(12):
        result = pipeline.process_frame(example_frame)

    if result:
        print(json.dumps(result, indent=2))

    print(f"\n[MCN Demo] Stats: {json.dumps(pipeline.get_stats(), indent=2)}")


if __name__ == "__main__":
    demo()
