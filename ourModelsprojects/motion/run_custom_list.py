import os
import subprocess
import sys

# ==========================================================
# MANUALLY ADD YOUR VIDEOS HERE
# Just put the filename (e.g., "5.mp4") if it's in testVideo
# ==========================================================
VIDEOS_TO_CHECK = [
    "20.mp4",
    "21.mp4",
    "22.mp4",
]
# ==========================================================

def run_custom():
    video_dir = "testVideo"
    script_name = "action_recognizer.py"

    print(f"--- Custom Video Batch Runner ---")
    print(f"Using Python: {sys.executable}")

    if not os.path.exists(video_dir):
        print(f"Error: Directory '{video_dir}' not found.")
        return

    # Filter to ensure we only try existing files
    valid_videos = []
    for v in VIDEOS_TO_CHECK:
        full_path = os.path.join(video_dir, v)
        if os.path.exists(full_path):
            valid_videos.append(full_path)
        else:
            print(f"[WARNING] File not found: {full_path}")

    if not valid_videos:
        print("No valid videos to process.")
        return

    print(f"Starting processing of {len(valid_videos)} videos...\n")

    for video_path in valid_videos:
        print(f">>> Processing: {video_path}")

        # Use the same command format that worked in run_single_video.py
        command = [sys.executable, script_name, "--video", video_path]

        try:
            # Shell=False is safer, and we use the current sys.executable
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error: Script failed for {video_path}")
        except KeyboardInterrupt:
            print("\nStopped by user.")
            break

    print("\nAll tasks finished.")

if __name__ == "__main__":
    run_custom()

