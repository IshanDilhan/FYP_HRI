# Jetson Orin Nano Deployment Guide

This guide provides step-by-step instructions on how to take the completed Multimodal Cross-Modal Network (MCN) code from GitHub and deploy it as a live ROS2 node on your NVIDIA Jetson robot.

---

## Prerequisites (On the Jetson Nano)
Before you begin, ensure your Jetson Nano is booted up and connected to the internet.
1. **NVIDIA JetPack SDK**: Ensure JetPack 5.x or 6.x is flashed (this provides CUDA and TensorRT).
2. **ROS2**: ROS2 Humble or Foxy must be installed.
3. **Camera**: A camera (e.g., Intel RealSense or a USB webcam) must be connected to the Jetson, and its ROS2 driver should be running and publishing to `/camera/color/image_raw`.

---

## Expected Build & Checkpoint Files (From your PC)
Before you move to the Jetson Nano, your local Windows machine must have generated the following files during Phase 1. Make sure these are pushed to GitHub so the Jetson Nano can pull them:

**1. The `checkpoints/` folder:**
When you ran the training script (`mcn.train`), it generated the following files in the `checkpoints/` directory:
- `best_model.pt`: The final, most accurate PyTorch weights for the MCN Transformer.
- `final_model.pt`: The weights from the very last training epoch.
- `training_history.json`: The accuracy and loss metrics recorded during training.
- *(You only need to ensure `best_model.pt` is pushed to GitHub).*

**2. The ONNX Export File:**
- `mcn_model.onnx`: This is the standalone, compiled graph of the MCN model. It removes all PyTorch dependencies and allows the Jetson Nano to run the model extremely fast.

---

## Step 1: Download the Project
Open a terminal on your Jetson Nano and clone your repository.

```bash
cd ~
git clone https://github.com/IshanDilhan/FYP_HRI.git
cd FYP_HRI
```

---

## Step 2: Set Up the Environment
Create an isolated Python environment on the Jetson to prevent dependency conflicts.

```bash
# Create and activate a virtual environment
python3 -m venv env
source env/bin/activate

# Install the strict dependencies required for the project
pip install -r requirements.txt

# Also install huggingface_hub for your custom motion model
pip install huggingface_hub
```

---

## Step 3: TensorRT Compilation (Optional but Recommended)
To achieve ultra-fast inference (< 1ms latency), compile the ONNX model into a TensorRT Engine specifically for the Jetson's GPU architecture.

*Note: You must have successfully generated `mcn.onnx` on your Windows machine and pushed it to GitHub before doing this.*

```bash
# Convert ONNX to TensorRT Engine
/usr/src/tensorrt/bin/trtexec --onnx=mcn.onnx --saveEngine=mcn.engine --fp16
```
*(If you skip this step, the pipeline will still run using standard PyTorch via CPU/CUDA, but it will be slightly slower).*

---

## Step 4: Launch the MCN ROS2 Node
Now that the code is downloaded and the environment is ready, you can start the ROS2 wrapper. This script automatically connects to the camera and runs your exact custom logic.

1. Ensure the ROS2 environment is sourced:
```bash
source /opt/ros/humble/setup.bash
```

2. Run the deployment script:
```bash
# Ensure you are in the project root and your virtual environment is active
python3 pipeline/ros2_mcn_node.py
```

You should see terminal output confirming that the models are loaded and the node has successfully subscribed to `/camera/color/image_raw`.

---

## Step 5: Verify the Output
Open a **second terminal** on the Jetson Nano to verify that the MCN is correctly interpreting the visual data and generating social policies for the robot.

```bash
source /opt/ros/humble/setup.bash
ros2 topic echo /mcn/behavioral_policy
```

Stand in front of the robot's camera and act out a scenario (e.g., wave your hands, or stand aggressively). You will see real-time JSON outputs appearing in the terminal, like this:

```json
{
  "predicted_intent": "HOSTILE_CONFRONTATION",
  "confidence": 0.89,
  "action": "Back Away",
  "proxemic_zone_radius": 1.5,
  "vocal_tone": "Calm/De-escalating"
}
```

The robot's Navigation Stack can now subscribe to this `/mcn/behavioral_policy` topic to automatically adjust its movement based on your social interactions!
