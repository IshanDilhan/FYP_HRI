# Multimodal Cross-Modal Network (MCN)

An **Adaptive Human-Robot Interaction (HRI) System** designed to act as the "Social Brain" for service robots.

This project processes real-time video streams and fuses 4 upstream vision modalities (Emotion, Gesture, Motion, Context) into a lightweight Transformer network. By looking at 1.2-second temporal windows of human behavior, the robot resolves conflicting signals and decodes ambiguous human intents into direct ROS2 behavioral policies.

---

## 🚀 Features

- **4 Upstream Vision Models**:
  - **Emotion** (FER + MTCNN/Haar): Detects facial expressions.
  - **Gesture** (MediaPipe Hands): Detects skeletal finger counting and pointing.
  - **Motion** (MediaPipe Pose): Detects full-body kinematics, velocity, and sitting.
  - **Context** (MobileNetV3): Classifies the environmental scene (Classroom, Kitchen, Office, etc.).
- **MCN Fusion Engine**:
  - 3-Layer Self-Attention Transformer with Dissonance Detection.
  - Highly optimized: only **115K parameters**.
  - Runs in `<5ms` per frame on CPU.
- **HRI Intent Classifier**: Identifies 9 intent categories including `EMERGENCY`, `HELP_REQUEST`, and `HOSTILE_CONFRONTATION`.
- **Policy Mapper**: Translates human intent into structured JSON policies containing proxemic target actions and social buffer zone radii.

---

## 🛠️ Installation & Setup

> **⚠️ CRITICAL:** This project relies on highly specific library versions. A mismatch in `numpy`, `tensorflow`, or `protobuf` will cause the pipeline to crash due to dependency hell. Please follow these exact steps.

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/FYP_Transformer.git
cd FYP_Transformer
```

### 2. Create the virtual environment
```bash
python -m venv env

# On Windows:
.\env\Scripts\activate

# On Mac/Linux:
source env/bin/activate
```

### 3. Install strictly locked dependencies
```bash
pip install -r requirements.txt
```

---

## 🏃 How to Run

### Option A: Interactive Menu (Recommended)
Place your test video files (`.mp4`, `.avi`, etc.) inside a folder named `testVideos/` in the root directory. Then run:
```bash
python select_and_run.py
```
This will open an interactive menu allowing you to quickly process individual videos or batch-process them all without saving the output.

If you also want to save the output annotated video to the `testVideos/` folder, run:
```bash
python select_and_save.py
```

### Option B: Run on Live Webcam
```bash
python run_video.py --input 0
```
*(Press `q` or `ESC` to close the webcam window).*

### Option C: Manual Command Line
```bash
python run_video.py --input testVideos/your_video.mp4 --output testVideos/result.mp4 --no-mtcnn
```

---

## 🧠 Training the MCN Model

By default, the MCN Engine initializes with random weights. To produce accurate, contextual decisions, you must train the model using the synthetic multi-modal HRI dataset generator.

1. Generate data and train the model:
```bash
python -m mcn.train
```
2. The best model will be saved to `checkpoints/best_model.pt`.
3. The video pipeline (`select_and_run.py` and `run_video.py`) will automatically detect and load this checkpoint if it exists.

---

## 📂 Project Structure

```text
├── mcn/                     # Transformer Fusion Engine ("The Brain")
│   ├── model.py             # 115K Param Transformer Architecture
│   ├── temporal_window.py   # 12-Frame sliding window buffer
│   ├── dissonance.py        # Conflicting signal detection logic
│   ├── policy_mapper.py     # Maps Intent -> ROS2 JSON Actions
│   └── train.py             # Training loop
├── models/                  # Upstream Vision Models
│   ├── emotion/             # Custom Emotion Model wrapper (ourModelsprojects)
│   ├── gesture/             # Custom Gesture Model wrapper (ourModelsprojects)
│   ├── motion/              # Custom Motion Model wrapper (ourModelsprojects)
│   ├── context/             # Custom Context Model wrapper (ourModelsprojects)
│   ├── emotion_model.py     # FER
│   ├── gesture_model.py     # MediaPipe Hands
│   ├── motion_model.py      # MediaPipe Pose
│   └── context_model.py     # MobileNetV3 Context
├── pipeline/                # Real-time orchestration
│   └── video_pipeline.py    # Merges 4 models -> MCN -> Video Overlay
├── tests/                   # Unit Tests
├── testVideos/              # Put your video files here (ignored by Git)
├── checkpoints/             # Trained MCN weights (ignored by Git)
├── run_video.py             # CLI entry point
├── select_and_run.py        # Interactive UI entry point (no save)
├── select_and_save.py       # Interactive UI entry point (saves output)
└── requirements.txt         # Locked dependency file
```

---

## 🤖 Robot Hardware Deployment (Jetson)

The MCN is designed to run on the **NVIDIA Jetson Orin Nano** via TensorRT.

1. Export the trained `.pt` model to ONNX:
```bash
python -m mcn.export_tensorrt --checkpoint checkpoints/best_model.pt
```
2. Transfer the ONNX file to the Jetson platform.
3. Run the generated bash script to compile the TensorRT engine for low-latency ROS2 node execution.
