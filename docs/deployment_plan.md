# MCN Robot Deployment Plan

This document outlines the systematic deployment of the Multimodal Cross-Modal Network (MCN) pipeline onto a service robot (e.g., NVIDIA Jetson Orin Nano running ROS2).

## Phase 1: Local Optimization & Export
Before moving the code to the robot, we must lock the models.
1. **Train all upstream models**: Ensure the `scene.pth`, `model_v1.pth` (emotion), and `keypoint_classifier.tflite` are final.
2. **Train the MCN**: Run `python -m mcn.train` one last time with all data to generate the final `best_model.pt`.
3. **ONNX Export**: Convert the PyTorch MCN transformer to ONNX format to remove Python overhead and allow TensorRT compilation.
   ```bash
   python -m mcn.export_tensorrt --checkpoint checkpoints/best_model.pt
   ```

## Phase 2: Hardware Preparation (NVIDIA Jetson)
1. **Flash JetPack SDK**: Ensure the Jetson is flashed with the latest JetPack (which includes CUDA, cuDNN, and TensorRT).
2. **ROS2 Installation**: Install ROS2 Humble (or Foxy).
3. **Environment Setup**: Clone the GitHub repo onto the Jetson. Create a virtual environment and install the strict dependencies from `requirements.txt`.

## Phase 3: TensorRT Compilation
The Transformer and any PyTorch vision models (like Scene Classification) need to be compiled for the Jetson's specific GPU architecture.
1. Run TensorRT `trtexec` on the exported `.onnx` models to generate `.engine` files.
   ```bash
   trtexec --onnx=mcn.onnx --saveEngine=mcn.engine --fp16
   ```
2. Update the MCN code to use the `tensorrt` Python bindings instead of PyTorch for inference (this drops the 5ms inference time down to <1ms).

## Phase 4: ROS2 Node Integration
We must wrap the `VideoPipeline` inside a ROS2 Node.
1. **Input Node (CameraSubscriber)**: Subscribe to the robot's `/camera/color/image_raw` topic instead of reading from `cv2.VideoCapture()`.
2. **Execution Node (PolicyPublisher)**: Modify the `VideoPipeline` to publish the JSON behavioral policy (Proxemic action, zone radius, tone) to a custom ROS2 topic (e.g., `/mcn/behavioral_policy`).
3. **Navigation Stack Hook**: Ensure the robot's Nav2 stack listens to the `/mcn/behavioral_policy` topic so it can actually perform the "Give Way" or "Approach + Stop" maneuvers.

## Phase 5: Live HRI Testing
1. **Dry Run**: Place the robot in a static environment and act out scenarios in front of it while monitoring the `/mcn/behavioral_policy` output via `ros2 topic echo`.
2. **Dynamic Run**: Allow the robot to move freely and verify that the spatial buffer zones (e.g., backing away 1.5 meters during a Hostile Confrontation) are accurately executed by the motor controllers without crashing.
