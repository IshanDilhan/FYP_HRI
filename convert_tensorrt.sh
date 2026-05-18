#!/bin/bash
# ============================================================
# TensorRT Conversion Script for Jetson Orin Nano
# Run this script ON the Jetson device with TensorRT installed
# ============================================================

ONNX_MODEL="mcn_model.onnx"
TRT_ENGINE="mcn_model.engine"

echo "[MCN] Converting ONNX to TensorRT engine..."
echo "[MCN] Input: $ONNX_MODEL"
echo "[MCN] Output: $TRT_ENGINE"

# Convert with FP16 for Jetson (half the size, ~same accuracy)
/usr/src/tensorrt/bin/trtexec \
    --onnx=$ONNX_MODEL \
    --saveEngine=$TRT_ENGINE \
    --fp16 \
    --workspace=1024 \
    --minShapes=context_idx:1x12,context_conf:1x12,emotion_idx:1x12,emotion_conf:1x12,gesture_idx:1x12,gesture_conf:1x12,motion_idx:1x12,motion_conf:1x12 \
    --optShapes=context_idx:1x12,context_conf:1x12,emotion_idx:1x12,emotion_conf:1x12,gesture_idx:1x12,gesture_conf:1x12,motion_idx:1x12,motion_conf:1x12 \
    --maxShapes=context_idx:8x12,context_conf:8x12,emotion_idx:8x12,emotion_conf:8x12,gesture_idx:8x12,gesture_conf:8x12,motion_idx:8x12,motion_conf:8x12 \
    --verbose

echo "[MCN] TensorRT engine saved: $TRT_ENGINE"

# Benchmark
echo "[MCN] Running benchmark..."
/usr/src/tensorrt/bin/trtexec \
    --loadEngine=$TRT_ENGINE \
    --iterations=100 \
    --avgRuns=10

echo "[MCN] Done!"
