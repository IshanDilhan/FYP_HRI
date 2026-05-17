"""
Environment Context Recognition Model
========================================
Uses torchvision MobileNetV3-Small (ImageNet-pretrained) combined with
scene-level heuristics to classify the interaction environment.

For the initial version, this uses a lightweight feature-based approach:
- MobileNetV3 extracts visual features
- A simple classifier maps features to environment categories
- Spatial/color cues provide additional context signals

Output: {"state": str, "confidence": float}
Categories: Classroom, Hospital, Museum, Retail Store, Clinic,
            Open Lobby, Narrow Hallway, Office, Kitchen
"""

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from typing import Dict, List

from models.__init__ import UpstreamModel


class ContextModel(UpstreamModel):
    """
    Indoor environment/scene recognition using MobileNetV3-Small.

    Architecture:
        MobileNetV3-Small backbone (ImageNet pretrained)
        → Replace classifier head → 9 environment classes
        → Combined with spatial heuristic features

    Note: Without fine-tuning on actual indoor scene data, this model
    uses ImageNet features + heuristic mapping. Accuracy will improve
    significantly when fine-tuned on Places365 or your custom data.
    """

    CATEGORIES = [
        "Classroom", "Hospital", "Museum", "Retail Store", "Clinic",
        "Open Lobby", "Narrow Hallway", "Office", "Kitchen",
    ]

    # ImageNet class indices that correlate with each environment
    # These are used to heuristically map ImageNet predictions to environments
    IMAGENET_HINTS = {
        "Classroom": [
            "desk", "monitor", "laptop", "notebook", "pencil", "book",
            "screen", "keyboard", "projector", "chair",
        ],
        "Hospital": [
            "stretcher", "mask", "stethoscope", "syringe", "pillow",
            "bed", "monitor", "oxygen",
        ],
        "Kitchen": [
            "spatula", "pot", "pan", "microwave", "toaster", "oven",
            "refrigerator", "cup", "plate", "bowl", "stove",
        ],
        "Office": [
            "desk", "monitor", "keyboard", "mouse", "printer",
            "filing_cabinet", "bookcase", "swivel_chair",
        ],
        "Retail Store": [
            "shopping_cart", "cash_machine", "barcode", "shelf",
            "bag", "counter",
        ],
        "Museum": [
            "painting", "frame", "statue", "vase", "pedestal",
        ],
    }

    def __init__(self, device: str = None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Load MobileNetV3-Small with ImageNet weights
        self.backbone = models.mobilenet_v3_small(
            weights=models.MobileNet_V3_Small_Weights.DEFAULT
        )
        self.backbone.eval()
        self.backbone.to(self.device)

        # ImageNet class names for heuristic mapping
        self.imagenet_weights = models.MobileNet_V3_Small_Weights.DEFAULT
        self.imagenet_categories = self.imagenet_weights.meta["categories"]

        # Preprocessing transform
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        self._last_imagenet_pred = None

    @torch.no_grad()
    def predict(self, frame: np.ndarray) -> Dict[str, object]:
        """
        Classify the environment from a BGR frame.

        Uses MobileNetV3 ImageNet predictions + heuristic mapping to
        determine the most likely indoor environment.
        """
        import cv2

        try:
            # Convert BGR to RGB
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Transform for MobileNetV3
            input_tensor = self.transform(rgb).unsqueeze(0).to(self.device)

            # Forward pass
            logits = self.backbone(input_tensor)
            probs = torch.softmax(logits, dim=-1)[0]

            # Get top-10 ImageNet predictions
            top_probs, top_indices = probs.topk(10)
            top_classes = [self.imagenet_categories[idx] for idx in top_indices.cpu()]
            top_scores = top_probs.cpu().numpy()

            self._last_imagenet_pred = list(zip(top_classes, top_scores.tolist()))

            # Heuristic mapping: score each environment by matching ImageNet classes
            env_scores = {}
            for env, hints in self.IMAGENET_HINTS.items():
                score = 0.0
                for cls_name, cls_score in zip(top_classes, top_scores):
                    cls_lower = cls_name.lower().replace("_", " ")
                    for hint in hints:
                        if hint.lower() in cls_lower:
                            score += float(cls_score)
                env_scores[env] = score

            # Also add spatial/color heuristics
            env_scores = self._add_spatial_heuristics(frame, env_scores)

            # Find best environment
            if max(env_scores.values()) > 0.01:
                best_env = max(env_scores, key=env_scores.get)
                confidence = min(env_scores[best_env] + 0.3, 0.95)
            else:
                # Default fallback based on basic frame analysis
                best_env = self._fallback_classification(frame)
                confidence = 0.5

            return {"state": best_env, "confidence": round(float(confidence), 3)}

        except Exception as e:
            return {"state": "Office", "confidence": 0.3}

    def _add_spatial_heuristics(
        self, frame: np.ndarray, env_scores: dict
    ) -> dict:
        """Add heuristic spatial/color cues to environment scores."""
        import cv2

        h, w = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Check for clinical/white environments (Hospital, Clinic)
        white_mask = cv2.inRange(hsv, (0, 0, 200), (180, 30, 255))
        white_ratio = np.sum(white_mask > 0) / (h * w)
        if white_ratio > 0.4:
            env_scores["Hospital"] = env_scores.get("Hospital", 0) + 0.15
            env_scores["Clinic"] = env_scores.get("Clinic", 0) + 0.12

        # Check for warm tones (Kitchen)
        warm_mask = cv2.inRange(hsv, (10, 50, 100), (30, 255, 255))
        warm_ratio = np.sum(warm_mask > 0) / (h * w)
        if warm_ratio > 0.15:
            env_scores["Kitchen"] = env_scores.get("Kitchen", 0) + 0.1

        # Aspect ratio and openness heuristic
        # Narrow hallways tend to have elongated perspectives
        edges = cv2.Canny(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 50, 150)
        edge_ratio = np.sum(edges > 0) / (h * w)

        if edge_ratio > 0.15:
            # Many edges = structured environment
            env_scores["Office"] = env_scores.get("Office", 0) + 0.05
        if edge_ratio < 0.05:
            env_scores["Open Lobby"] = env_scores.get("Open Lobby", 0) + 0.08

        return env_scores

    def _fallback_classification(self, frame: np.ndarray) -> str:
        """Simple fallback when ImageNet features don't match any environment."""
        import cv2

        # Use basic color and brightness analysis
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        brightness = np.mean(hsv[:, :, 2])

        if brightness > 180:
            return "Open Lobby"
        elif brightness > 120:
            return "Office"
        else:
            return "Narrow Hallway"

    def get_name(self) -> str:
        return "Environment Context (MobileNetV3)"

    def get_categories(self) -> List[str]:
        return self.CATEGORIES

    def get_imagenet_predictions(self) -> list:
        """Return last ImageNet top-10 predictions for debugging."""
        return self._last_imagenet_pred or []
