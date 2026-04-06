"""
video.py — Extract frames from a video file using OpenCV.
"""

import os
import cv2
from typing import Optional


def extract_frames(
    video_path: str,
    output_dir: str,
    fps: Optional[float] = None,
    prefix: str = "frame",
    verbose: bool = True,
) -> int:
    """
    Extract frames from a video file.

    Args:
        video_path:  path to the input video file.
        output_dir:  directory where frames will be saved as PNG.
        fps:         target sampling rate. None means every frame.
                     Use 1.0 for 1 frame/second (as in the paper).
        prefix:      filename prefix for saved frames.
        verbose:     print progress.

    Returns:
        Number of frames extracted.
    """
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps is None:
        step = 1
    else:
        step = max(1, int(round(src_fps / fps)))

    if verbose:
        print(f"[sigvideo] Video: {os.path.basename(video_path)}")
        print(f"[sigvideo] Source FPS: {src_fps:.2f}  |  Total frames: {total_frames}")
        print(f"[sigvideo] Extracting every {step} frame(s) → ~{src_fps/step:.2f} fps")

    frame_idx = 0
    saved = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % step == 0:
            fname = os.path.join(output_dir, f"{prefix}_{saved:06d}.png")
            cv2.imwrite(fname, frame)
            saved += 1
        frame_idx += 1

    cap.release()

    if verbose:
        print(f"[sigvideo] Saved {saved} frames to '{output_dir}'.")

    return saved
