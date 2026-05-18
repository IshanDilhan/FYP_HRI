import cv2
import torch
import sys
import os
import numpy as np
from pathlib import Path
from torchvision import transforms
from typing import Dict, List

from models.__init__ import UpstreamModel
from mcn.config import CONTEXT_CATEGORIES

class OurContextModel(UpstreamModel):
    def __init__(self):
        self.categories = CONTEXT_CATEGORIES
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Add their path so it can import their module
        sys.path.insert(0, os.path.abspath("ourModelsprojects/context/scene_classification"))
        
        try:
            from scene_model import SceneModel
            self.model = SceneModel(num_classes=3).to(self.device)
            
            # Load weights
            weights_path = os.path.abspath("ourModelsprojects/context/scene_classification/weights/scene.pth")
            if os.path.exists(weights_path):
                self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
                print(f"[OurContextModel] Successfully loaded weights from {weights_path}")
            else:
                print(f"[OurContextModel] WARNING: Weights not found at {weights_path}. Predictions will be random.")
                
            self.model.eval()
            
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            self.classes = ["Classroom", "Office", "Kitchen"]
            self.loaded = True
            print("[OurContextModel] Successfully hooked into your Custom Scene Classification Model!")
        except Exception as e:
            print(f"[OurContextModel] Error loading custom logic: {e}")
            self.loaded = False

    def predict(self, frame: np.ndarray) -> Dict[str, object]:
        if not self.loaded:
            return {"state": "Office", "confidence": 0.0}
            
        try:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = self.transform(frame_rgb).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                output = self.model(img)
                probs = torch.softmax(output, dim=1)
                pred = torch.argmax(probs, 1).item()
                
            label = self.classes[pred]
            confidence = probs[0, pred].item()
            
            return {"state": label, "confidence": confidence}
        except Exception as e:
            return {"state": "Office", "confidence": 0.0}

    def get_name(self) -> str:
        return "Our Context Model (Scene Classification)"

    def get_categories(self) -> List[str]:
        return self.categories
