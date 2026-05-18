import os
import subprocess
import sys

def select_and_run():
    video_dir = "testVideo"
    script_name = "action_recognizer.py"
    
    print("--- Single Video Action Recognizer ---")
    print(f"Using Python: {sys.executable}")
    print(f"Default folder: {video_dir}")
    print("Enter a specific video path, or press Enter to list files in 'testVideo':")
    choice = input("> ").strip()

    target_video = ""

    if choice and os.path.isfile(choice):
        target_video = choice
    else:
        # 2. List files if no valid path provided
        if not os.path.exists(video_dir):
            print(f"Error: Directory '{video_dir}' not found and no valid path entered.")
            return

        video_extensions = ('.mp4', '.avi', '.mov', '.mkv')
        videos = [f for f in os.listdir(video_dir) if f.lower().endswith(video_extensions)]
        
        if not videos:
            print(f"No video files found in '{video_dir}'.")
            return

        # Sort videos numerically
        try:
            videos.sort(key=lambda x: int(os.path.splitext(x)[0]))
        except ValueError:
            videos.sort()

        print("\nAvailable videos:")
        for i, v in enumerate(videos):
            print(f"[{i+1}] {v}")
        
        try:
            idx = int(input("\nSelect video number: ")) - 1
            if 0 <= idx < len(videos):
                target_video = os.path.join(video_dir, videos[idx])
            else:
                print("Invalid selection.")
                return
        except ValueError:
            print("Invalid input.")
            return

    # 3. Execute
    if target_video:
        print(f"\n>>> Running recognizer on: {target_video}")
        command = [sys.executable, script_name, "--video", target_video]
        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error: {e}")
        except KeyboardInterrupt:
            print("\nStopped by user.")

if __name__ == "__main__":
    select_and_run()
