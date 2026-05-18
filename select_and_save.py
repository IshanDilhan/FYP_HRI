"""
Interactive menu to select and run a test video, AND SAVE the output video.
Lists all videos in the testVideos folder and lets you choose one.
The output will be saved with '_MCN_result' appended to the filename.

Usage:
    cd D:\FYP_Tranformer
    .\env\Scripts\activate
    python select_and_save.py
"""

import os
import sys
import glob

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

VIDEO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "testVideos")
CHECKPOINT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints", "best_model.pt")
VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm")

def main():
    print("=" * 60)
    print("  MCN - Interactive Video Selector")
    print("=" * 60)

    # Find all videos
    videos = []
    for ext in VIDEO_EXTENSIONS:
        videos.extend(glob.glob(os.path.join(VIDEO_DIR, f"*{ext}")))
        videos.extend(glob.glob(os.path.join(VIDEO_DIR, f"*{ext.upper()}")))

    # Remove duplicates and sort
    videos = sorted(set(videos))

    if not videos:
        print(f"\nNo videos found in: {VIDEO_DIR}")
        print(f"Supported formats: {', '.join(VIDEO_EXTENSIONS)}")
        print(f"Please add video files to the testVideos folder and try again.")
        return

    # Display menu
    print(f"\nFound {len(videos)} video(s) in the 'testVideos' folder:\n")
    print("  [0]  Process ALL videos")
    
    for i, v in enumerate(videos, 1):
        name = os.path.basename(v)
        size_mb = os.path.getsize(v) / (1024 * 1024)
        print(f"  [{i}]  {name} ({size_mb:.1f} MB)")
    
    print("\n  [q]  Quit / Cancel\n")

    # Get user selection
    while True:
        choice = input("Select a number to run (e.g., 1): ").strip().lower()
        if choice in ('q', 'quit', 'exit'):
            print("Cancelled.")
            return
        if choice.isdigit() and 0 <= int(choice) <= len(videos):
            break
        print("Invalid selection. Please enter a valid number or 'q' to quit.")

    selection = int(choice)

    # Initialize pipeline
    print("\nLoading AI models (this takes ~10 seconds)...")
    from pipeline.video_pipeline import VideoPipeline

    use_checkpoint = None
    if os.path.exists(CHECKPOINT):
        use_checkpoint = CHECKPOINT
        print(f"Using trained MCN: {CHECKPOINT}")
    else:
        print("WARNING: MCN not trained yet. Run 'python -m mcn.train' first.")
        print("         Scenario predictions will be random.\n")

    pipeline = VideoPipeline(
        mcn_checkpoint=use_checkpoint,
        use_mtcnn=False,  # Faster face detection
    )

    if selection == 0:
        videos_to_run = videos
        print(f"\nProcessing ALL {len(videos)} videos...")
    else:
        videos_to_run = [videos[selection - 1]]
        print(f"\nProcessing selected video: {os.path.basename(videos_to_run[0])}")

    # Process selected video(s)
    for i, video_path in enumerate(videos_to_run, 1):
        name = os.path.basename(video_path)
        base, ext = os.path.splitext(name)
        output_path = os.path.join(VIDEO_DIR, f"{base}_MCN_result{ext}")

        print(f"\n{'=' * 60}")
        if selection == 0:
            print(f"Processing [{i}/{len(videos_to_run)}]: {name}")
        else:
            print(f"Processing: {name}")
        print(f"Output: {os.path.basename(output_path)}")
        print(f"{'=' * 60}")

        pipeline.mcn.reset()  # Reset MCN memory between videos

        pipeline.run(
            input_source=video_path,
            output_path=output_path,
            display=True,
            print_json=True,
        )

    print(f"\n{'=' * 60}")
    print(f"Done! Results saved in: {VIDEO_DIR}")
    print(f"{'=' * 60}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user.")
