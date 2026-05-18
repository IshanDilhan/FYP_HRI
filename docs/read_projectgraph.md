How to read this flow:
Raw Data Input: The system captures live video frames from the robot's camera.
Upstream Vision Models: The frame is split across 4 parallel models. Each model extracts a specific social cue (Emotion, Gesture, Motion, Context) and returns a categorical state alongside a confidence percentage.
MCN Fusion Engine:
Since human intent builds up over time, the cues are pushed into a Temporal Sliding Window (holding the last 1.2 seconds of history).
The data is transformed into numerical vectors (Embeddings).
The Transformer analyzes all modalities together, allowing it to weigh conflicting signals (like a person smiling but walking aggressively).
The network outputs the final Intent (e.g., HELP_REQUEST) and detects any Dissonance (conflicting body language).
Robotic Execution: The Policy Mapper turns the human intent into structured instructions the robot physically executes (like backing away, approaching slowly, or adopting a sympathetic voice).