import torch
import cv2
import numpy as np
from typing import Dict, List
import torchvision.transforms as transforms
import os
import sys

from models.__init__ import UpstreamModel
from mcn.config import CONTEXT_CATEGORIES

class OurContextModel(UpstreamModel):
    def __init__(self, model_path="ourModelsprojects/context/checkpoints/model_v1.pth", device=None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        self.categories = CONTEXT_CATEGORIES
        self.model = None
        
        try:
            # Try to load the custom model architecture from ourModelsprojects
            sys.path.insert(0, os.path.abspath("ourModelsprojects/context"))
            try:
                from models.custom_context import get_model
                self.model = get_model()
            except ImportError:
                # Default architecture if not found
                import torchvision.models as models
                import torch.nn as nn
                self.model = models.mobilenet_v2(pretrained=False)
                self.model.classifier[1] = nn.Linear(self.model.last_channel, len(self.categories))

            if os.path.exists(model_path):
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                print(f"[OurContextModel] Loaded weights from {model_path}")
            else:
                print(f"[OurContextModel] Weights not found at {model_path}. Using uninitialized weights.")
        except Exception as e:
            print(f"[OurContextModel] Error initializing model: {e}")

        if self.model:
            self.model.to(self.device)
            self.model.eval()

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def predict(self, frame: np.ndarray) -> Dict[str, object]:
        if self.model is None:
            return {"state": "Classroom", "confidence": 0.0}
            
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            input_tensor = self.transform(rgb).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probs = torch.softmax(outputs, dim=1)[0]
                
            conf, idx = torch.max(probs, dim=0)
            state = self.categories[idx.item()]
            
            return {"state": state, "confidence": round(float(conf.cpu()), 3)}
        except Exception as e:
            return {"state": "Classroom", "confidence": 0.0}

    def get_name(self) -> str:
        return "Our Context Model (Custom)"

    def get_categories(self) -> List[str]:
        return self.categories
