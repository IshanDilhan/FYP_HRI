"""
MCN Video Pipeline — CLI Entry Point
======================================
Run the full Video → 4 Models → MCN → Robot Policy pipeline.

Usage:
    # Process a video file
    python run_video.py --input video.mp4 --output result.mp4

    # Use webcam (live)
    python run_video.py --input 0

    # Process video without display (headless)
    python run_video.py --input video.mp4 --output result.mp4 --no-display

    # Use trained MCN model
    python run_video.py --input video.mp4 --checkpoint checkpoints/best_model.pt
"""

import argparse
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser(
        description="MCN Video Pipeline — Multimodal Intent Recognition",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_video.py --input video.mp4 --output result.mp4
  python run_video.py --input 0                              # webcam
  python run_video.py --input video.mp4 --checkpoint checkpoints/best_model.pt
        """,
    )
    parser.add_argument(
        "--input", "-i", type=str, default="0",
        help="Input video path or camera index (default: 0 for webcam)",
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="Output annotated video path (optional)",
    )
    parser.add_argument(
        "--checkpoint", "-c", type=str, default=None,
        help="Path to trained MCN checkpoint (.pt file)",
    )
    parser.add_argument(
        "--no-display", action="store_true",
        help="Run without displaying video window (headless mode)",
    )
    parser.add_argument(
        "--max-frames", type=int, default=None,
        help="Maximum frames to process (default: all)",
    )
    parser.add_argument(
        "--no-mtcnn", action="store_true",
        help="Use Haar Cascade instead of MTCNN for face detection (faster)",
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true",
        help="Suppress per-frame JSON output",
    )
    args = parser.parse_args()

    # Import pipeline (lazy to avoid loading models if --help)
    from pipeline.video_pipeline import VideoPipeline

    # Initialize
    pipeline = VideoPipeline(
        mcn_checkpoint=args.checkpoint,
        use_mtcnn=not args.no_mtcnn,
        device=None,
    )

    # Run
    pipeline.run(
        input_source=args.input,
        output_path=args.output,
        display=not args.no_display,
        max_frames=args.max_frames,
        print_json=not args.quiet,
    )


if __name__ == "__main__":
    main()
