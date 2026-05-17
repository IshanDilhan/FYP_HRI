"""Quick smoke test for all 4 upstream models + pipeline integration."""
import sys, os, time
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

import numpy as np
import cv2

print("=" * 60)
print("MCN FULL PIPELINE SMOKE TEST")
print("=" * 60)

# Create a synthetic test frame (640x480 BGR)
frame = np.random.randint(50, 200, (480, 640, 3), dtype=np.uint8)
# Add a face-like circle for emotion detection
cv2.circle(frame, (320, 200), 80, (200, 180, 160), -1)  # Face area
cv2.circle(frame, (300, 185), 10, (50, 50, 50), -1)  # Left eye
cv2.circle(frame, (340, 185), 10, (50, 50, 50), -1)  # Right eye
cv2.ellipse(frame, (320, 220), (25, 10), 0, 0, 180, (50, 50, 50), 2)  # Mouth

print("\n1. Testing Emotion Model (FER)...")
t0 = time.time()
from models.emotion_model import EmotionModel
emo = EmotionModel(use_mtcnn=False)  # Haar Cascade for speed
result = emo.predict(frame)
print(f"   Result: {result}")
print(f"   Time: {(time.time()-t0)*1000:.0f}ms")

print("\n2. Testing Gesture Model (MediaPipe Hands)...")
t0 = time.time()
from models.gesture_model import GestureModel
ges = GestureModel()
result = ges.predict(frame)
print(f"   Result: {result}")
print(f"   Time: {(time.time()-t0)*1000:.0f}ms")

print("\n3. Testing Motion Model (MediaPipe Pose)...")
t0 = time.time()
from models.motion_model import MotionModel
mot = MotionModel()
result = mot.predict(frame)
print(f"   Result: {result}")
print(f"   Time: {(time.time()-t0)*1000:.0f}ms")

print("\n4. Testing Context Model (MobileNetV3)...")
t0 = time.time()
from models.context_model import ContextModel
ctx = ContextModel()
result = ctx.predict(frame)
print(f"   Result: {result}")
print(f"   Time: {(time.time()-t0)*1000:.0f}ms")

print("\n5. Testing Full Pipeline Integration...")
from pipeline.video_pipeline import VideoPipeline
import json

pipe = VideoPipeline(use_mtcnn=False)

# Process 12 frames to fill the MCN window
for i in range(12):
    result = pipe.process_frame(frame)

if result["policy"]:
    print(f"   Policy Output:")
    print(f"   {json.dumps(result['policy'], indent=2)}")
else:
    print("   [INFO] No policy yet (window filling)")

print("\n" + "=" * 60)
print("ALL MODELS LOADED AND WORKING!")
print("=" * 60)
print("\nTo run on a video:")
print("  python run_video.py --input your_video.mp4 --output result.mp4")
print("  python run_video.py --input 0  # for webcam")
