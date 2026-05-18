import os
import subprocess
import sys

def run_batch():
    video_dir = "testVideo"
    script_name = "action_recognizer.py"
    
    if not os.path.exists(video_dir):
        print(f"Error: Directory '{video_dir}' not found.")
        return

    # Filter for video files
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv')
    videos = [f for f in os.listdir(video_dir) if f.lower().endswith(video_extensions)]
    
    if not videos:
        print(f"No video files found in '{video_dir}'.")
        return

    # Sort videos numerically if possible, otherwise alphabetically
    try:
        videos.sort(key=lambda x: int(os.path.splitext(x)[0]))
    except ValueError:
        videos.sort()

    print(f"Found {len(videos)} videos. Starting processing...")

    for video in videos:
        video_path = os.path.join(video_dir, video)
        print(f"\n>>> Processing: {video_path}")
        
        # Construct the command
        command = [sys.executable, script_name, "--video", video_path]
        
        try:
            # Run the script and wait for it to finish
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error processing {video}: {e}")
        except KeyboardInterrupt:
            print("\nBatch processing interrupted by user.")
            break

    print("\nBatch processing complete.")

if __name__ == "__main__":
    run_batch()
