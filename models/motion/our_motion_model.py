import cv2
import numpy as np
import sys
import os
from typing import Dict, List

from models.__init__ import UpstreamModel
from mcn.config import MOTION_CATEGORIES

class OurMotionModel(UpstreamModel):
    """
    Custom Motion Model wrapper that natively uses the MediaPipe + ResNet50 logic 
    from your action_recognizer.py inside ourModelsprojects/motion.
    """
    def __init__(self, use_resnet=False):
        self.categories = MOTION_CATEGORIES
        self.frame_idx = 0
        
        # Add the folder to the path so we can import your action_recognizer
        sys.path.insert(0, os.path.abspath("ourModelsprojects/motion"))
        
        try:
            import mediapipe.python.solutions.pose as mp_pose
            self.mp_pose = mp_pose
            self.pose_tracker = self.mp_pose.Pose(
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
                model_complexity=1,
            )
            
            from action_recognizer import MotionAnalyser, PoseClassifier
            self.motion_analyser = MotionAnalyser()
            self.pose_classifier = PoseClassifier()
            
            # Optional: Load the ResNet50 Stanford40 model
            self.use_resnet = use_resnet
            if self.use_resnet:
                from action_recognizer import ActionModel
                self.action_model = ActionModel()
                self.action_model.load()
                
            self.loaded = True
            print("[OurMotionModel] Successfully hooked into MediaPipe + ResNet50 logic!")
        except Exception as e:
            print(f"[OurMotionModel] Error loading custom logic: {e}")
            self.loaded = False

    def predict(self, frame: np.ndarray) -> Dict[str, object]:
        if not self.loaded:
            return {"state": "Stationary", "confidence": 0.0}
            
        self.frame_idx += 1
        
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = self.pose_tracker.process(rgb)
            
            landmarks = results.pose_landmarks
            
            # Use your custom logic to get the motion label
            motion_label = self.motion_analyser.update(landmarks, self.frame_idx)
            pose_label = self.pose_classifier.classify(landmarks)
            
            # Just to map sitting if needed, though motion_label handles most
            if pose_label == "Sitting" and motion_label in ["Stationary", "Minimal"]:
                motion_label = "Sitting"
                
            # If using ResNet50, we could merge action_label logic here, 
            # but MCN's categories primarily expect the motion_label values.
            if self.use_resnet and self.frame_idx % 10 == 0:
                action_label = self.action_model.predict(rgb)
                
            # Default confidence since your heuristic doesn't output probabilities
            confidence = 0.85 if landmarks else 0.4
            
            # Ensure it strictly matches MCN categories
            if motion_label not in self.categories:
                motion_label = "Stationary"
            
            return {"state": motion_label, "confidence": confidence}
            
        except Exception as e:
            return {"state": "Stationary", "confidence": 0.0}

    def get_name(self) -> str:
        return "Our Motion Model (MediaPipe + ResNet50)"

    def get_categories(self) -> List[str]:
        return self.categories
