"""
Process all videos in the testVideos folder.
Saves annotated results alongside the originals.

Usage:
    cd D:\FYP_Tranformer
    .\env\Scripts\activate
    python run_test_videos.py
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
    print("  MCN - Process All Test Videos")
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
        print(f"\nPlease add video files to the testVideos folder and try again.")
        return

    print(f"\nFound {len(videos)} video(s) in testVideos folder:")
    for i, v in enumerate(videos, 1):
        name = os.path.basename(v)
        size_mb = os.path.getsize(v) / (1024 * 1024)
        print(f"  {i}. {name} ({size_mb:.1f} MB)")

    # Initialize pipeline once (models are heavy to load)
    print(f"\nLoading AI models (this takes ~15 seconds on first run)...")
    from pipeline.video_pipeline import VideoPipeline

    # Check for trained MCN
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

    # Process each video
    for i, video_path in enumerate(videos, 1):
        name = os.path.basename(video_path)
        base, ext = os.path.splitext(name)
        output_path = os.path.join(VIDEO_DIR, f"{base}_MCN_result{ext}")

        print(f"\n{'=' * 60}")
        print(f"Processing [{i}/{len(videos)}]: {name}")
        print(f"Output: {os.path.basename(output_path)}")
        print(f"{'=' * 60}")

        pipeline.mcn.reset()  # Reset MCN window between videos

        pipeline.run(
            input_source=video_path,
            output_path=output_path,
            display=True,
            print_json=True,
        )

    print(f"\n{'=' * 60}")
    print(f"All {len(videos)} videos processed!")
    print(f"Results saved in: {VIDEO_DIR}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
