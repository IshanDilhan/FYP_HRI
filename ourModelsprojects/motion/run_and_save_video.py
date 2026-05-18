import os
import subprocess
import sys

def run_and_save():
    video_dir = "testVideo"
    output_dir = "output"
    script_name = "action_recognizer.py"
    
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    print("--- Video Action Recognizer with Save ---")
    print(f"Default video folder: {video_dir}")
    print(f"Default output folder: {output_dir}")
    
    # 1. Get video list
    if not os.path.exists(video_dir):
        print(f"Error: Directory '{video_dir}' not found.")
        return

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

    print("\nAvailable videos:")
    for i, v in enumerate(videos):
        print(f"[{i+1}] {v}")
    
    # 2. Select Video
    try:
        val = input("\nEnter video number or filename (e.g. 1 or 1.mp4): ").strip()
        if not val:
            print("No input provided.")
            return
            
        target_video = ""
        if val.isdigit():
            idx = int(val) - 1
            if 0 <= idx < len(videos):
                target_video = os.path.join(video_dir, videos[idx])
            else:
                print("Invalid number.")
                return
        else:
            # Check if it's a filename in testVideo
            potential_path = os.path.join(video_dir, val)
            if os.path.isfile(potential_path):
                target_video = potential_path
            elif os.path.isfile(val):
                target_video = val
            else:
                print(f"Could not find video: {val}")
                return
    except ValueError:
        print("Invalid input.")
        return

    # 3. Get Output Filename
    base_name = os.path.basename(target_video)
    name_only = os.path.splitext(base_name)[0]
    default_output = f"output_{name_only}.avi"
    
    print(f"\nTarget video: {target_video}")
    output_choice = input(f"Enter output filename (default: {default_output}): ").strip()
    
    if not output_choice:
        output_choice = default_output
    
    # Ensure it's in the output directory if no path is provided
    if not os.path.isabs(output_choice) and os.sep not in output_choice:
        output_path = os.path.join(output_dir, output_choice)
    else:
        output_path = output_choice

    # 4. Execute
    print(f"\n>>> Running recognizer on: {target_video}")
    print(f">>> Saving output to: {output_path}")
    
    command = [sys.executable, script_name, "--video", target_video, "--save", output_path]
    
    # Optional: ask if they want to run the action model (slower)
    run_model = input("Run action model (ResNet50)? (y/N): ").strip().lower()
    if run_model == 'y':
        command.append("--action-model")

    try:
        subprocess.run(command, check=True)
        print(f"\nSuccessfully processed and saved to {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
    except KeyboardInterrupt:
        print("\nStopped by user.")

if __name__ == "__main__":
    run_and_save()
