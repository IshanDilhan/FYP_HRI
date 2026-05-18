# Human Action & Motion Recognizer

A robust computer vision system that combines skeletal tracking, deep learning, and geometric analysis to classify human actions, poses, and movement patterns in real-time.

---

## 📋 Project Overview

### **Model Name**
A Hybrid Multi-Stage Recognition System:
- **MediaPipe Pose Engine:** For high-speed skeletal landmark detection.
- **ResNet50 (Residual Network):** Fine-tuned on the **Stanford40 Actions** dataset for complex action classification.
- **Geometric Motion Analyser:** Custom-built velocity-based tracker for directional movement.
- **Rule-based Pose Classifier:** Geometric analysis for basic postures (Standing, Sitting, etc.).

### **Objective**
To develop a comprehensive human monitoring system capable of understanding not just *what* a person is doing (e.g., "Reading"), but also *how* they are moving (e.g., "Approaching") and their *physical posture* (e.g., "Sitting").

### **Dataset**
- **Stanford 40 Actions:** Consists of 9,532 images across 40 diverse categories (e.g., applauding, fishing, phoning, riding a bike).
- **MediaPipe Internal Data:** Used for the underlying 33-keypoint pose estimation.

### **Techniques Used**
- **Transfer Learning:** Leveraging ImageNet-pretrained weights for ResNet50 and fine-tuning on Stanford40.
- **Pose Estimation:** Using MediaPipe's Blazepose architecture for real-time keypoint extraction.
- **Velocity Tracking:** Using a sliding window of frames to calculate Center-of-Mass (CoM) movement in 3D space.
- **Normalization:** Image transforms (Resize, Crop, Normalization) to match model input requirements.

### **Extracted Features**
- **Landmarks:** 33 body keypoints (x, y, z coordinates).
- **Motion Vectors:** Velocity (vx, vy, vz) derived from shoulder and hip coordinates.
- **Geometric Ratios:** Torso-to-body and leg-to-body ratios to distinguish Sitting/Standing/Lying.
- **Visual Encodings:** High-dimensional feature maps extracted by the ResNet backbone.

### **Output**
- **Real-time Annotated Video:** Skeleton overlay with live tracking.
- **Pose Label:** Standing, Sitting, Lying, Crouching.
- **Motion Label:** Stationary, Approaching, Backing Away, Fast Across, Slow Approach, etc.
- **Action Label:** 40 classes (e.g., Phoning, Texting, Drinking).
- **Performance Data:** Real-time FPS (Frames Per Second).

---

## 📈 Results & Performance

### **Accuracies**
- **ResNet50 Action Model:** ~**88.5%** accuracy on the Stanford40 test set.
- **Pose Tracking:** High precision (MediaPipe standard) even in varying lighting conditions.

### **Training Process**
1. **Backbone Selection:** ResNet50 was chosen for its excellent balance between depth (50 layers) and inference speed.
2. **Pre-training:** The model starts with weights trained on **ImageNet**, allowing it to already understand basic shapes, edges, and textures.
3. **Fine-tuning:** The final layers were replaced and trained specifically on the 40 Stanford Action classes to map visual features to specific human behaviors.
4. **Optimization:** Standard PyTorch transforms (224x224 resize, Mean/Std normalization) ensure the model receives data in the exact format it was trained on.

---

## 🎓 Viva Section: Expected Questions & Answers

**1. Why use a hybrid approach instead of just one Deep Learning model?**
> MediaPipe provides extremely fast skeletal tracking on the CPU, which is perfect for movement direction. ResNet50 is much better at "understanding" the visual context of an action (like holding a phone) which skeletal points alone might miss. Combining them gives us the best of both worlds: speed and context.

**2. How does the system handle "Depth" (Z-axis) with a standard 2D webcam?**
> MediaPipe estimates a "relative depth" based on the pose size and perspective. The Motion Analyser tracks this Z-value over time. If Z is decreasing significantly, the system classifies the motion as "Approaching."

**3. What is Transfer Learning and why did you use it here?**
> Transfer Learning is taking a model trained on a massive dataset (ImageNet) and repurposing it for a smaller, specific task (Stanford40). It saves days of training time and achieves much higher accuracy than training from scratch.

**4. How is the 'Center of Mass' (CoM) calculated?**
> We average the coordinates of the shoulders and hips (Landmarks 11, 12, 23, 24). This provides a stable point that represents the main body trunk, reducing "noise" from waving hands or moving feet.

**5. Why ResNet50 specifically?**
> ResNet (Residual Network) solves the "vanishing gradient" problem using skip connections. ResNet50 is deep enough to capture complex actions but light enough to run near real-time on modern hardware.

---

## 🎮 Usage Guide

### 1. Process All Videos in a Folder
Put your `.mp4` files in `testVideo/` and run:
```powershell
.\env\Scripts\python.exe run_all_and_save.py
```

### 2. Live Webcam Recognition
```powershell
.\env\Scripts\python.exe action_recognizer.py --webcam --action-model
```

### 3. Save Output for a Single Video
```powershell
.\env\Scripts\python.exe action_recognizer.py --video path/to/video.mp4 --save output.avi --action-model
```

---

## 📁 Project Structure
- `action_recognizer.py`: Core logic and model implementations.
- `run_all_and_save.py`: Batch processing script for `testVideo/`.
- `testVideo/`: Input directory for video files.
- `output/`: Directory for annotated results.
- `env/`: Virtual environment with dependencies.

---

## 🧠 Theoretical Background (Viva Cheat Sheet)

If you are asked about the "Why" and "How" of the technology, use these simplified explanations:

### 1. Machine Learning (ML) vs. Deep Learning (DL)
- **Machine Learning:** Think of this as the "Geometric Motion Analyser" in our code. We (the humans) write specific rules (e.g., "if velocity > 0.05, then it is Fast"). The computer follows these hard-coded mathematical rules.
- **Deep Learning:** This is the **ResNet50** part. We don't give the computer rules. Instead, we show it 9,000+ images from the **Stanford40** dataset. The computer "learns" to recognize a "Guitar" or a "Phone" by identifying patterns in pixels across many layers.

### 2. Convolutional Neural Networks (CNN)
The ResNet50 model is a **CNN**. CNNs are special neural networks designed to "see" like humans:
- **Lower Layers:** Detect simple things like edges, lines, and corners.
- **Middle Layers:** Detect shapes like circles, rectangles, or textures.
- **Higher Layers:** Detect complex objects like a human face, a hand, or a bicycle.

### 3. Residual Networks (ResNet) & Skip Connections
Standard deep networks often "forget" information as it passes through too many layers (called the *Vanishing Gradient* problem).
- **ResNet** introduced **Skip Connections** (or Identity Shortcuts). These allow information to "skip" some layers, ensuring the model stays accurate even when it is very deep (50 layers in our case).

### 4. Transfer Learning
We didn't train ResNet50 from scratch (which would take weeks and millions of images).
- We used **Transfer Learning**: We took a model already trained on **ImageNet** (1.2 million images) so it already knew how to see "edges" and "shapes." 
- We only "fine-tuned" the very last part of the model to recognize our specific **40 actions**.

### 5. MediaPipe (Pose Estimation)
MediaPipe uses a model called **BlazePose**.
- It is optimized for mobile/real-time use.
- It doesn't just look at pixels; it predicts a **Skeletal Topology** (33 landmarks).
- This is much more efficient than traditional image processing because it reduces a whole person into just 33 (x, y, z) points.

### 6. Velocity & Temporal Smoothing
To prevent the labels from flickering (e.g., jumping between "Stationary" and "Moving"), we use a **Sliding Window (deque)** of 12 frames.
- We calculate the average movement over these 12 frames. This is called **Temporal Smoothing**. It makes the detection look "stable" and "smooth" to the user.
