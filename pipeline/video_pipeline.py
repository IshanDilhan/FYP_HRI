"""
Video → MCN → Robot Policy Pipeline
======================================
Orchestrates the full end-to-end pipeline:
  1. Reads video frames (file or camera)
  2. Runs 4 upstream models in parallel on each frame
  3. Feeds results into the MCN temporal window
  4. MCN fuses and outputs scenario + behavioral policy
  5. Optionally visualizes results on the video
"""

import cv2
import json
import time
import sys
import os
import numpy as np
from typing import Dict, Optional, List

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcn.config import MCNConfig
from mcn.model import MultimodalCrossModalNetwork
from mcn.inference import MCNInferencePipeline


class VideoPipeline:
    """
    Full video-to-policy pipeline.

    Usage:
        pipeline = VideoPipeline()
        pipeline.run("input_video.mp4", output_path="output_annotated.mp4")
    """

    def __init__(
        self,
        mcn_checkpoint: Optional[str] = None,
        use_mtcnn: bool = False,
        device: str = None,
    ):
        """
        Initialize all models.

        Args:
            mcn_checkpoint: Path to trained MCN checkpoint. If None, uses
                           randomly initialized MCN (for demo purposes).
            use_mtcnn: Use MTCNN for face detection (more accurate, slower).
            device: 'cuda' or 'cpu'. Auto-detected if None.
        """
        print("[Pipeline] Initializing models...")

        # Initialize 4 upstream models
        print("  Loading Emotion Model (FER)...")
        from models.emotion_model import EmotionModel
        self.emotion_model = EmotionModel(use_mtcnn=use_mtcnn)

        print("  Loading Gesture Model (MediaPipe Hands)...")
        from models.gesture_model import GestureModel
        self.gesture_model = GestureModel()

        print("  Loading Motion Model (MediaPipe Pose)...")
        from models.motion_model import MotionModel
        self.motion_model = MotionModel()

        print("  Loading Context Model (MobileNetV3)...")
        from models.context_model import ContextModel
        self.context_model = ContextModel(device=device)

        # Initialize MCN fusion engine
        print("  Loading MCN Fusion Engine...")
        config = MCNConfig()
        if mcn_checkpoint and os.path.exists(mcn_checkpoint):
            self.mcn = MCNInferencePipeline.from_checkpoint(mcn_checkpoint, device)
        else:
            model = MultimodalCrossModalNetwork(config)
            self.mcn = MCNInferencePipeline(model, config, device)
            if mcn_checkpoint:
                print(f"  [WARNING] Checkpoint not found: {mcn_checkpoint}")
                print(f"  [WARNING] Using untrained MCN (predictions will be random)")
            else:
                print("  [INFO] No checkpoint provided, using untrained MCN")

        self._frame_count = 0
        self._fps_timer = time.time()
        self._current_fps = 0.0

        print("[Pipeline] All models loaded successfully!\n")

    def process_frame(self, frame: np.ndarray) -> Dict:
        """
        Process a single video frame through the full pipeline.

        Args:
            frame: BGR image from OpenCV.

        Returns:
            Dict with all model outputs + MCN policy.
        """
        self._frame_count += 1

        # Run 4 upstream models
        emotion_result = self.emotion_model.predict(frame)
        gesture_result = self.gesture_model.predict(frame)
        motion_result = self.motion_model.predict(frame)
        context_result = self.context_model.predict(frame)

        # Package as MCN input frame
        mcn_input = {
            "environment_context": context_result,
            "facial_affect_emotion": emotion_result,
            "skeletal_hand_gesture": gesture_result,
            "body_motion_vector": motion_result,
        }

        # Feed to MCN
        policy = self.mcn.process_frame(mcn_input, allow_warmup=True)

        # Compute FPS
        now = time.time()
        elapsed = now - self._fps_timer
        if elapsed > 0.5:
            self._current_fps = self._frame_count / elapsed
            self._frame_count = 0
            self._fps_timer = now

        return {
            "context": context_result,
            "emotion": emotion_result,
            "gesture": gesture_result,
            "motion": motion_result,
            "policy": policy,
            "fps": round(self._current_fps, 1),
        }

    def draw_overlay(self, frame: np.ndarray, result: Dict) -> np.ndarray:
        """Draw detection results and policy overlay on the frame."""
        overlay = frame.copy()
        h, w = overlay.shape[:2]

        # --- Top-left: Model outputs panel ---
        panel_h = 180
        cv2.rectangle(overlay, (0, 0), (400, panel_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, overlay)
        # Re-draw the panel area with transparency
        sub = overlay[0:panel_h, 0:400]
        black = np.zeros_like(sub)
        cv2.addWeighted(sub, 0.4, black, 0.6, 0, sub)
        overlay[0:panel_h, 0:400] = sub

        y = 22
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.55
        color_white = (255, 255, 255)
        color_green = (0, 255, 0)
        color_yellow = (0, 255, 255)
        color_cyan = (255, 255, 0)

        # Context
        ctx = result["context"]
        cv2.putText(overlay, f"Context: {ctx['state']} ({ctx['confidence']:.0%})",
                    (10, y), font, font_scale, color_cyan, 1)
        y += 28

        # Emotion
        emo = result["emotion"]
        cv2.putText(overlay, f"Emotion: {emo['state']} ({emo['confidence']:.0%})",
                    (10, y), font, font_scale, color_green, 1)
        y += 28

        # Gesture
        ges = result["gesture"]
        cv2.putText(overlay, f"Gesture: {ges['state']} ({ges['confidence']:.0%})",
                    (10, y), font, font_scale, color_yellow, 1)
        y += 28

        # Motion
        mot = result["motion"]
        cv2.putText(overlay, f"Motion:  {mot['state']} ({mot['confidence']:.0%})",
                    (10, y), font, font_scale, (180, 180, 255), 1)
        y += 28

        # FPS
        cv2.putText(overlay, f"FPS: {result['fps']}",
                    (10, y), font, font_scale, color_white, 1)

        # --- Bottom: Policy output ---
        policy = result.get("policy")
        if policy:
            # Policy panel at bottom
            panel_top = h - 100
            sub_bottom = overlay[panel_top:h, 0:w]
            black_bottom = np.zeros_like(sub_bottom)
            cv2.addWeighted(sub_bottom, 0.4, black_bottom, 0.6, 0, sub_bottom)
            overlay[panel_top:h, 0:w] = sub_bottom

            intent = policy.get("predicted_intent", "UNKNOWN")
            prob = policy.get("intent_probability", 0)
            scenario = policy.get("scenario_id", "")
            bp = policy.get("behavioral_policy", {})
            action = bp.get("proxemic_action", "")
            tone = bp.get("vocal_affect_tone", "")

            # Intent with color coding
            intent_color = self._get_intent_color(intent)

            cv2.putText(overlay, f"SCENARIO: {intent} ({prob:.0%})",
                        (10, panel_top + 25), font, 0.7, intent_color, 2)
            cv2.putText(overlay, f"ID: {scenario}  |  Action: {action}",
                        (10, panel_top + 55), font, 0.5, color_white, 1)
            cv2.putText(overlay, f"Tone: {tone}  |  Buffer: {bp.get('social_buffer_zone_radius', 0)}m",
                        (10, panel_top + 80), font, 0.5, color_white, 1)

        return overlay

    def _get_intent_color(self, intent: str) -> tuple:
        """Color-code intents for visual clarity."""
        colors = {
            "EMERGENCY": (0, 0, 255),          # Red
            "HOSTILE_CONFRONTATION": (0, 0, 200),  # Dark red
            "HELP_REQUEST": (0, 200, 255),     # Orange
            "DISTRESSED_STUDENT_QUERY": (0, 200, 255),
            "GIVE_WAY": (0, 255, 255),         # Yellow
            "GREETING": (0, 255, 0),           # Green
            "TASK_ASSIST": (255, 200, 0),      # Cyan
            "NEUTRAL_PASS": (200, 200, 200),   # Gray
            "UNKNOWN": (128, 128, 128),
        }
        return colors.get(intent, (255, 255, 255))

    def run(
        self,
        input_source: str = "0",
        output_path: Optional[str] = None,
        display: bool = True,
        max_frames: Optional[int] = None,
        print_json: bool = True,
    ):
        """
        Run the pipeline on a video file or camera.

        Args:
            input_source: Path to video file, or "0" for webcam.
            output_path: Path to save annotated output video. None = no save.
            display: Show live window with results.
            max_frames: Stop after N frames. None = process entire video.
            print_json: Print policy JSON to console for each frame.
        """
        # Open video source
        if input_source.isdigit():
            cap = cv2.VideoCapture(int(input_source))
            print(f"[Pipeline] Opened camera {input_source}")
        else:
            cap = cv2.VideoCapture(input_source)
            print(f"[Pipeline] Opened video: {input_source}")

        if not cap.isOpened():
            print(f"[ERROR] Could not open video source: {input_source}")
            return

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"[Pipeline] Video: {width}x{height} @ {fps:.0f}fps, {total_frames} frames")

        # Output video writer
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            print(f"[Pipeline] Saving output to: {output_path}")

        print(f"[Pipeline] Starting processing...\n")

        frame_idx = 0
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_idx += 1
                if max_frames and frame_idx > max_frames:
                    break

                # Process frame
                result = self.process_frame(frame)

                # Print policy JSON
                if print_json and result["policy"]:
                    policy_str = json.dumps(result["policy"], indent=None)
                    progress = f"[{frame_idx}/{total_frames}]" if total_frames > 0 else f"[{frame_idx}]"
                    sys.stdout.write(f"\r{progress} {result['policy']['predicted_intent']:30s} "
                                   f"({result['policy']['intent_probability']:.0%}) | "
                                   f"E:{result['emotion']['state']:10s} "
                                   f"G:{result['gesture']['state']:15s} "
                                   f"M:{result['motion']['state']:15s} "
                                   f"| {result['fps']:.0f}fps")
                    sys.stdout.flush()

                # Draw overlay
                annotated = self.draw_overlay(frame, result)

                # Save output
                if writer:
                    writer.write(annotated)

                # Display
                if display:
                    cv2.imshow("MCN Pipeline - Video Analysis", annotated)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q') or key == 27:  # q or ESC
                        print("\n[Pipeline] Stopped by user")
                        break

        except KeyboardInterrupt:
            print("\n[Pipeline] Interrupted")
        finally:
            cap.release()
            if writer:
                writer.release()
                print(f"\n[Pipeline] Output saved to: {output_path}")
            if display:
                cv2.destroyAllWindows()

        print(f"\n[Pipeline] Processed {frame_idx} frames")

        # Print final stats
        stats = self.mcn.get_stats()
        print(f"[Pipeline] MCN Stats: {json.dumps(stats, indent=2)}")
