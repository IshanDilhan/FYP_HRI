import os
import subprocess
import sys

def run_all_and_save():
    video_dir = "testVideo"
    output_dir = "output"
    script_name = "action_recognizer.py"
    
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    if not os.path.exists(video_dir):
        print(f"Error: Directory '{video_dir}' not found.")
        return

    # 1. Get and filter video list
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv')
    videos = [f for f in os.listdir(video_dir) if f.lower().endswith(video_extensions)]
    
    if not videos:
        print(f"No video files found in '{video_dir}'.")
        return

    # Sort videos numerically if possible
    try:
        videos.sort(key=lambda x: int(os.path.splitext(x)[0]))
    except ValueError:
        videos.sort()

    print(f"--- Batch Video Action Recognizer ---")
    print(f"Found {len(videos)} videos in '{video_dir}'.")
    
    # Optional: ask if they want to run the action model (slower for batch)
    run_model = input("Run action model (ResNet50) for all? (y/N): ").strip().lower()
    
    print("\nStarting batch processing...")

    for video in videos:
        video_path = os.path.join(video_dir, video)
        name_only = os.path.splitext(video)[0]
        output_path = os.path.join(output_dir, f"output_{name_only}.avi")
        
        print(f"\n>>> Processing: {video}")
        print(f"    Saving to: {output_path}")
        
        # Construct the command
        command = [sys.executable, script_name, "--video", video_path, "--save", output_path]
        if run_model == 'y':
            command.append("--action-model")
        
        try:
            # Run the script and wait for it to finish
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error processing {video}: {e}")
        except KeyboardInterrupt:
            print("\nBatch processing interrupted by user.")
            break

    print("\nBatch processing complete. Outputs are in the 'output' folder.")

if __name__ == "__main__":
    run_all_and_save()
