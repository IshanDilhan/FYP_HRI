# Model Training Documentation

This document explains how the gesture recognition models for this project were trained, the architecture used, and how to re-train them.

## 1. The Dataset (HaGRID)
For the static signs (Open hand, Closed fist, Pointing, OK), this project utilizes the **HaGRID (HAnd Gesture Recognition Image Dataset)**. 

Because the full dataset is over 700GB, a localized subset (`hagrid-sample-30k-384p`) was used to train the `keypoint_classifier`.

### Why HaGRID?
Pre-trained models often struggle in real-world scenarios (messy kitchens, classrooms) because they are trained in sterile lab environments. HaGRID consists of diverse, full-color images taken in real-world environments, making the resulting model highly robust to varied lighting and backgrounds.

*Note: The `dataset/` folder containing these images is excluded from GitHub via `.gitignore` to save space.*

## 2. Model Architecture
This project does not pass raw pixels into a Convolutional Neural Network (CNN). Instead, it uses a two-step pipeline:

1. **MediaPipe Processing:** The raw image/frame is passed to Google's MediaPipe, which extracts the exact (x, y) coordinates of 21 hand landmarks (joints).
2. **Custom Multi-Layer Perceptron (MLP):** Those 21 coordinates are normalized relative to the wrist to ensure distance/hand size does not affect accuracy. They are flattened into a 1D array of 42 values. This small array is passed into a lightweight custom TensorFlow Neural Network.

**Network Layout:**
* Input Layer (42 neurons)
* Dropout (0.2)
* Dense Layer (20 neurons, ReLU)
* Dropout (0.4)
* Dense Layer (10 neurons, ReLU)
* Output Layer (4 neurons, Softmax)

This architecture ensures the model is tiny (~6 KB) and runs incredibly fast on CPU.

## 3. Training Pipeline

### Step A: Extracting Coordinates from Images
If you download a new batch of HaGRID images into the `dataset/` folder, you cannot train on them directly. You must run a script to process them through MediaPipe first.

A script (e.g., `extract_dataset.py`) loops through the images, grabs the 21 normalized coordinates using MediaPipe, and appends them to `model/keypoint_classifier/keypoint.csv`.

### Step B: Jupyter Notebook Training
Inside the `train/` folder are two Jupyter Notebooks:
* `keypoint_classification.ipynb` (For static poses: Open, Close, Point, OK)
* `point_history_classification.ipynb` (For dynamic motions: Waving, Beckoning)

**To train:**
1. Activate your virtual environment.
2. Ensure you have installed Jupyter: `pip install jupyter notebook scikit-learn`
3. Run `jupyter notebook` in your terminal.
4. Open the notebook in the `train/` folder.
5. Click **"Run All"**.

The notebook will automatically read the `.csv` data, split it into training and testing sets, train the neural network for up to 1000 epochs (with Early Stopping), and output the final, optimized `keypoint_classifier.tflite` directly into the `model/` directory for the main app to use.

## 4. Specific Technical Adjustments
* **Confidence Filtering:** TFLite models often output low confidence scores even when correct. A strict 85% confidence threshold is applied **exclusively to the Pointing gesture**. This prevents resting hands from defaulting mathematically to "Pointing".
* **Dynamic Priority:** In two-handed tracking, dynamic motions (waving, beckoning) take logical precedence over static signs to prevent a resting hand from interrupting a gesture.
* **15 FPS Sync:** The system captures and processes video at a locked 15 FPS to maintain accurate point history for motion detection, but UI text and console data logging are throttled to exactly 5 FPS using a frame-count modulo logic.