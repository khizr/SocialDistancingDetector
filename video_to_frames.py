#!/usr/bin/env python3
"""
Short script to convert a video to frames.

Example: ./video_to_frames.py sample.mp4 converted
"""
import argparse
import detector

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert a video to its frames')
    parser.add_argument('video_path', help="Path to the video file")
    parser.add_argument('frames_dir', help="Path to the directory to save the frames")
    args = parser.parse_args()

    logger = detector.Logger('')
    detector.cache_video_to_frames(args.frames_dir, args.video_path,
        detector.get_framecount(args.video_path), logger)
