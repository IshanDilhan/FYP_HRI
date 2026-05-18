# Hand Gesture Recognition using MediaPipe

This project estimates hand pose using MediaPipe and recognizes specific Human-Robot Interaction (HRI) gestures using custom-trained TensorFlow Lite neural networks. 

The system supports two-handed gesture detection, dynamically resolving conflicts between resting hands and active motions.

## Features
* **Two-Hand Tracking:** Tracks and evaluates both hands independently.
* **Dynamic Prioritization:** Action gestures (like Waving) automatically override static misclassifications (like resting hands defaulting to Pointing).
* **Targeted Thresholds:** Strict 85% confidence filtering specifically on 'Pointing' to reduce false positives.
* **5 FPS Data Logging:** Core camera processes run at a locked 15 FPS for smooth tracking, while output vectors and UI scenario text update cleanly at exactly 5 FPS.

## Detected Scenarios
Based on the defined Core Scenarios, the system outputs the following:
1. **Arms waving** (Both hands moving side-to-side)
2. **Arms up** (Both hands raised and open)
3. **Wave** (One hand, large horizontal amplitude)
4. **Brief wave** (One hand, small horizontal amplitude)
5. **Beckoning / Reaching** (Vertical "come here" motion, or a flattened palm extending)
6. **Pointing** (Index finger extended)
7. **One hand raised** (Open palm, stationary)
8. **None** (Resting hands, fists, or unrecognized)

---

## Setup and Running (Windows)

### 1. Create a Virtual Environment
It is recommended to use a standard Windows Python installation (3.11 recommended).
```powershell
python -m venv venv
.\venv\Scripts\activate
```

### 2. Install Dependencies
```powershell
pip install opencv-python==4.9.0.80 mediapipe==0.10.11 tensorflow==2.15.1 numpy==1.26.4
```

### 3. Run the Application
You have two ways to run the gesture detection:

**Live Webcam:**
```powershell
python app.py
```

**Pre-recorded Video Test:**
```powershell
python test_video.py --video "testVideo/your_video.mp4"
```

## Documentation
For detailed information on how the custom AI models were trained using the HaGRID dataset, please refer to [TRAINING.md](./TRAINING.md).