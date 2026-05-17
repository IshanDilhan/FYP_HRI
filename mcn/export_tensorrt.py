"""
TensorRT / ONNX Export Script
==============================
Exports the trained MCN model for edge deployment on NVIDIA Jetson Orin Nano.

Export pipeline:
  1. PyTorch model → ONNX (with dynamic batch size)
  2. ONNX → TensorRT engine (on the Jetson device itself)

The ONNX export can be done on any machine. TensorRT conversion should
be performed on the target Jetson hardware for optimal kernel tuning.
"""

import os
import torch
import torch.onnx
from typing import Optional

from .config import MCNConfig
from .model import MultimodalCrossModalNetwork


def export_to_onnx(
    checkpoint_path: str,
    output_path: str = "mcn_model.onnx",
    config: MCNConfig = None,
    opset_version: int = 17,
) -> str:
    """
    Export trained MCN model to ONNX format.

    Args:
        checkpoint_path: Path to the saved PyTorch checkpoint.
        output_path: Path for the output ONNX file.
        config: MCNConfig (loaded from checkpoint if not provided).
        opset_version: ONNX opset version (17 recommended for transformer ops).

    Returns:
        Path to the exported ONNX file.
    """
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if config is None:
        config = checkpoint.get("config", MCNConfig())

    model = MultimodalCrossModalNetwork(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Create dummy inputs matching the model's forward signature
    B = 1
    W = config.window_size

    dummy_inputs = (
        torch.randint(0, config.context_vocab_size, (B, W)),   # context_idx
        torch.rand(B, W),                                       # context_conf
        torch.randint(0, config.emotion_vocab_size, (B, W)),   # emotion_idx
        torch.rand(B, W),                                       # emotion_conf
        torch.randint(0, config.gesture_vocab_size, (B, W)),   # gesture_idx
        torch.rand(B, W),                                       # gesture_conf
        torch.randint(0, config.motion_vocab_size, (B, W)),    # motion_idx
        torch.rand(B, W),                                       # motion_conf
    )

    input_names = [
        "context_idx", "context_conf",
        "emotion_idx", "emotion_conf",
        "gesture_idx", "gesture_conf",
        "motion_idx", "motion_conf",
    ]

    output_names = [
        "intent_logits", "intent_probs",
        "conflict_logits", "confidence",
    ]

    # Dynamic axes for variable batch size
    dynamic_axes = {}
    for name in input_names:
        dynamic_axes[name] = {0: "batch_size"}
    for name in output_names:
        dynamic_axes[name] = {0: "batch_size"}

    # Export
    print(f"[MCN Export] Exporting to ONNX: {output_path}")
    print(f"[MCN Export] Model params: {model.count_parameters():,}")

    # Wrap model to return tuple instead of dict (ONNX requirement)
    class MCNONNXWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, *args):
            out = self.model(*args)
            return (
                out["intent_logits"],
                out["intent_probs"],
                out["conflict_logits"],
                out["confidence"],
            )

    wrapper = MCNONNXWrapper(model)
    wrapper.eval()

    torch.onnx.export(
        wrapper,
        dummy_inputs,
        output_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=opset_version,
        do_constant_folding=True,
    )

    # Verify
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"[MCN Export] ONNX model saved: {output_path} ({file_size_mb:.2f} MB)")

    # Optional: validate with onnxruntime
    try:
        import onnxruntime as ort
        session = ort.InferenceSession(output_path)
        print(f"[MCN Export] ONNX validation passed ✓")
        print(f"[MCN Export] ONNX inputs: {[i.name for i in session.get_inputs()]}")
        print(f"[MCN Export] ONNX outputs: {[o.name for o in session.get_outputs()]}")
    except ImportError:
        print("[MCN Export] onnxruntime not installed, skipping validation")

    return output_path


def generate_tensorrt_script(onnx_path: str, output_script: str = "convert_tensorrt.sh"):
    """
    Generate a shell script to convert ONNX to TensorRT on the Jetson device.

    This script should be run ON the Jetson Orin Nano, not on the dev machine.
    """
    script = f"""#!/bin/bash
# ============================================================
# TensorRT Conversion Script for Jetson Orin Nano
# Run this script ON the Jetson device with TensorRT installed
# ============================================================

ONNX_MODEL="{onnx_path}"
TRT_ENGINE="mcn_model.engine"

echo "[MCN] Converting ONNX to TensorRT engine..."
echo "[MCN] Input: $ONNX_MODEL"
echo "[MCN] Output: $TRT_ENGINE"

# Convert with FP16 for Jetson (half the size, ~same accuracy)
/usr/src/tensorrt/bin/trtexec \\
    --onnx=$ONNX_MODEL \\
    --saveEngine=$TRT_ENGINE \\
    --fp16 \\
    --workspace=1024 \\
    --minShapes=context_idx:1x12,context_conf:1x12,emotion_idx:1x12,emotion_conf:1x12,gesture_idx:1x12,gesture_conf:1x12,motion_idx:1x12,motion_conf:1x12 \\
    --optShapes=context_idx:1x12,context_conf:1x12,emotion_idx:1x12,emotion_conf:1x12,gesture_idx:1x12,gesture_conf:1x12,motion_idx:1x12,motion_conf:1x12 \\
    --maxShapes=context_idx:8x12,context_conf:8x12,emotion_idx:8x12,emotion_conf:8x12,gesture_idx:8x12,gesture_conf:8x12,motion_idx:8x12,motion_conf:8x12 \\
    --verbose

echo "[MCN] TensorRT engine saved: $TRT_ENGINE"

# Benchmark
echo "[MCN] Running benchmark..."
/usr/src/tensorrt/bin/trtexec \\
    --loadEngine=$TRT_ENGINE \\
    --iterations=100 \\
    --avgRuns=10

echo "[MCN] Done!"
"""
    with open(output_script, "w", newline="\n") as f:
        f.write(script)
    print(f"[MCN Export] TensorRT conversion script saved: {output_script}")
    print(f"[MCN Export] Transfer this to your Jetson and run: bash {output_script}")
    return output_script


def main():
    """CLI entry point for export."""
    import argparse

    parser = argparse.ArgumentParser(description="Export MCN model to ONNX/TensorRT")
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to PyTorch checkpoint (.pt file)",
    )
    parser.add_argument(
        "--output", type=str, default="mcn_model.onnx",
        help="Output ONNX file path",
    )
    parser.add_argument(
        "--opset", type=int, default=17,
        help="ONNX opset version",
    )
    args = parser.parse_args()

    onnx_path = export_to_onnx(args.checkpoint, args.output, opset_version=args.opset)
    generate_tensorrt_script(onnx_path)


if __name__ == "__main__":
    main()
