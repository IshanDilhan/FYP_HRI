# Standard Ubuntu PC / Laptop Deployment Guide

If you do not have an NVIDIA Jetson Orin Nano, you can deploy and run the MCN pipeline on **any standard computer or laptop running Ubuntu Linux** (or WSL2 on Windows) that has ROS2 installed. 

Because we are running on a standard PC/Laptop, we do not need the complex TensorRT compilation step. The pipeline will run in real-time using standard CPU or GPU via PyTorch/ONNX Runtime.

---

## Prerequisites
1. **Operating System**: Ubuntu Linux (20.04 or 22.04) or WSL2 (Ubuntu).
2. **ROS2**: ROS2 Humble or Foxy installed.
3. **Camera**: A standard USB webcam or built-in laptop camera publishing to `/camera/color/image_raw` (e.g., using `usb_cam` ROS2 node).

---

## Step 1: Clone the Project
Open a terminal on your Linux machine/laptop and clone your repository:

```bash
cd ~
git clone https://github.com/IshanDilhan/FYP_HRI.git
cd FYP_HRI
```

---

## Step 2: Set Up the Environment
Create an isolated Python environment and install the required dependencies:

```bash
# Create and activate a virtual environment
python3 -m venv env
source env/bin/activate

# Install the strict dependencies
pip install -r requirements.txt

# Install huggingface_hub for your motion model
pip install huggingface_hub
```

---

## Step 3: Launch the ROS2 Node
The node will automatically run inference using your PyTorch weights (`checkpoints/best_model.pt`) and your custom models.

1. Source your ROS2 workspace:
   ```bash
   source /opt/ros/humble/setup.bash
   ```

2. Run the MCN ROS2 Node:
   ```bash
   python3 pipeline/ros2_mcn_node.py
   ```

The terminal will log that all 4 custom upstream models (Emotion, Gesture, Motion, Context) have successfully initialized and the node is listening for frames.

---

## Step 4: Verify the Behavioral Policies
Open a **second terminal** to monitor the decisions the MCN is making:

```bash
source /opt/ros/humble/setup.bash
ros2 topic echo /mcn/behavioral_policy
```

When you stand in front of your laptop/USB webcam, it will publish live JSON decisions:

```json
{
  "predicted_intent": "GREETING",
  "confidence": 0.95,
  "action": "Approach And Stop",
  "proxemic_zone_radius": 0.8,
  "vocal_tone": "Polite/Friendly"
}
```
