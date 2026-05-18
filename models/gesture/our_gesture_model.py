import cv2
import numpy as np
import sys
import os
import copy
from typing import Dict, List
from collections import deque, Counter

from models.__init__ import UpstreamModel
from mcn.config import GESTURE_CATEGORIES

class OurGestureModel(UpstreamModel):
    def __init__(self):
        self.categories = GESTURE_CATEGORIES
        self.frame_idx = 0
        
        # Add their path so it can import their module
        sys.path.insert(0, os.path.abspath("ourModelsprojects/gesture/gesture_detection"))
        
        try:
            import mediapipe.python.solutions.hands as mp_hands
            from model import KeyPointClassifier, PointHistoryClassifier
            import app as gesture_app # Import their app.py to use its helper functions
            
            self.gesture_app = gesture_app
            self.mp_hands = mp_hands
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.5,
            )
            
            # KeyPointClassifier is hardcoded to look for "model/..." in its constructor
            # We must change cwd temporarily or pass the path. But it uses relative paths.
            original_cwd = os.getcwd()
            os.chdir(os.path.abspath("ourModelsprojects/gesture/gesture_detection"))
            
            self.keypoint_classifier = KeyPointClassifier()
            self.point_history_classifier = PointHistoryClassifier()
            
            # Read labels
            import csv
            with open('model/keypoint_classifier/keypoint_classifier_label.csv', encoding='utf-8-sig') as f:
                self.keypoint_labels = [row[0] for row in csv.reader(f)]
                
            os.chdir(original_cwd)
            
            self.history_length = 16
            self.point_history = {0: deque(maxlen=self.history_length), 1: deque(maxlen=self.history_length)}
            self.finger_gesture_history = {0: deque(maxlen=self.history_length), 1: deque(maxlen=self.history_length)}
            for i in range(2):
                for _ in range(self.history_length):
                    self.point_history[i].append([0, 0])
                    self.finger_gesture_history[i].append(0)
            
            self.loaded = True
            print("[OurGestureModel] Successfully hooked into your Custom Gesture Model (KeyPoint+PointHistory)!")
        except Exception as e:
            print(f"[OurGestureModel] Error loading custom logic: {e}")
            self.loaded = False

    def predict(self, frame: np.ndarray) -> Dict[str, object]:
        if not self.loaded:
            return {"state": "No Gesture", "confidence": 0.0}
            
        self.frame_idx += 1
        
        try:
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = self.hands.process(image)
            image.flags.writeable = True
            
            detected_hand_indices = []
            hand_states = {}
            
            if results.multi_hand_landmarks is not None:
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    hand_index = handedness.classification[0].index
                    detected_hand_indices.append(hand_index)
                    
                    brect = self.gesture_app.calc_bounding_rect(frame, hand_landmarks)
                    landmark_list = self.gesture_app.calc_landmark_list(frame, hand_landmarks)
                    pre_processed_landmark_list = self.gesture_app.pre_process_landmark(landmark_list)
                    pre_processed_history_list = self.gesture_app.pre_process_point_history(frame, self.point_history[hand_index])
                    
                    hand_sign_id, hand_sign_conf = self.keypoint_classifier(pre_processed_landmark_list)
                    
                    if hand_sign_id == 2 and hand_sign_conf < 0.85:
                        hand_sign_id = -1
                        
                    if hand_sign_id == 0 or hand_sign_id == 2:
                        self.point_history[hand_index].append(landmark_list[8])
                    else:
                        self.point_history[hand_index].append([0, 0])
                        
                    finger_gesture_id = 0
                    fg_conf = 0.0
                    wave_detected, wave_amplitude = self.gesture_app.detect_wave(self.point_history[hand_index])
                    
                    if len(pre_processed_history_list) == (self.history_length * 2):
                        if wave_detected:
                            finger_gesture_id = 4
                            fg_conf = 0.95
                        elif self.gesture_app.detect_come_here(self.point_history[hand_index]):
                            finger_gesture_id = 5
                            fg_conf = 0.92
                        elif hand_sign_id == 2:
                            finger_gesture_id, fg_conf = self.point_history_classifier(pre_processed_history_list)
                            
                    self.finger_gesture_history[hand_index].append(finger_gesture_id)
                    most_common_fg_id = Counter(self.finger_gesture_history[hand_index]).most_common()
                    current_fg_id = most_common_fg_id[0][0]
                    
                    hand_states[hand_index] = {
                        'sign': hand_sign_id,
                        'action': current_fg_id,
                        'brect': brect,
                        'wave_amp': wave_amplitude,
                        'sign_conf': hand_sign_conf,
                        'action_conf': fg_conf
                    }
                    
            for i in range(2):
                if i not in detected_hand_indices:
                    self.point_history[i].append([0, 0])
                    
            # Resolve global scenario using their exact logic
            global_scenario_text = "No Gesture"
            global_conf = 0.0
            
            num_hands = len(hand_states)
            if num_hands == 2:
                h1 = hand_states[list(hand_states.keys())[0]]
                h2 = hand_states[list(hand_states.keys())[1]]
                if h1['action'] == 4 and h2['action'] == 4:
                    global_scenario_text = "Arms Waving"
                    global_conf = min(h1['action_conf'], h2['action_conf'])
                elif h1['sign'] == 0 and h2['sign'] == 0:
                    global_scenario_text = "Arms Up"
                    global_conf = min(h1['sign_conf'], h2['sign_conf'])
                elif h1['action'] == 4 or h2['action'] == 4:
                    global_scenario_text = "Wave"
                    global_conf = h1['action_conf'] if h1['action'] == 4 else h2['action_conf']
                elif h1['action'] == 5 or h2['action'] == 5:
                    global_scenario_text = "Beckoning"
                    global_conf = h1['action_conf'] if h1['action'] == 5 else h2['action_conf']
                elif h1['sign'] == 2 or h2['sign'] == 2:
                    global_scenario_text = "Pointing"
                    global_conf = h1['sign_conf'] if h1['sign'] == 2 else h2['sign_conf']
                elif h1['sign'] == 0 or h2['sign'] == 0:
                    target_h = h1 if h1['sign'] == 0 else h2
                    w = target_h['brect'][2] - target_h['brect'][0]
                    h = target_h['brect'][3] - target_h['brect'][1]
                    if w > h * 1.2:
                        global_scenario_text = "Reaching"
                    else:
                        global_scenario_text = "One Hand Raised"
                    global_conf = target_h['sign_conf']
            elif num_hands == 1:
                h1 = hand_states[list(hand_states.keys())[0]]
                if h1['action'] == 4:
                    if h1['wave_amp'] > 150:
                        global_scenario_text = "Wave"
                    else:
                        global_scenario_text = "Brief Wave"
                    global_conf = h1['action_conf']
                elif h1['action'] == 5:
                    global_scenario_text = "Beckoning"
                    global_conf = h1['action_conf']
                elif h1['sign'] == 2:
                    global_scenario_text = "Pointing"
                    global_conf = h1['sign_conf']
                elif h1['sign'] == 0:
                    w = h1['brect'][2] - h1['brect'][0]
                    h = h1['brect'][3] - h1['brect'][1]
                    if w > h * 1.2:
                        global_scenario_text = "Reaching"
                    else:
                        global_scenario_text = "One Hand Raised"
                    global_conf = h1['sign_conf']
                    
            if global_scenario_text == "No Gesture" and global_conf == 0.0:
                global_conf = 1.0 # high confidence of nothing
                
            return {"state": global_scenario_text, "confidence": float(global_conf)}
            
        except Exception as e:
            return {"state": "No Gesture", "confidence": 0.0}

    def get_name(self) -> str:
        return "Our Gesture Model (KeyPoint+PointHistory)"

    def get_categories(self) -> List[str]:
        return self.categories
