"""
pipeline.py — Primary end-to-end pipeline: video in → summarized video out.

This is the main entry-point for sigvideo. It chains:
  1. extract_frames()        — sample the input video at a target fps
  2. summarize() / auto_length()  — select keyframes via Signature Transform
  3. write_summary_video()   — assemble selected frames into an output MP4
                               (uses ffmpeg H.264 when available, cv2 fallback)

Reference:
  @article{sigvideodecurto2023,
    title   = {Summarization of Videos with the Signature Transform},
    author  = {de Curtò, J. and de Zarzà, I. and Roig, G. and Calafate, C.T.},
    journal = {Electronics},
    volume  = {12}, number = {7}, pages = {1735}, year = {2023},
    doi     = {10.3390/electronics12071735}
  }
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .video import extract_frames
from .summarize import summarize, auto_length
from .core import SIG_ORDER, FRAME_SIZE


# ── ffmpeg helpers ────────────────────────────────────────────────────────────

def _ffmpeg_available() -> bool:
    """True if ffmpeg is on PATH."""
    return shutil.which("ffmpeg") is not None


def _write_with_ffmpeg(
    frame_list: List[np.ndarray],
    output_path: str,
    fps_output: float,
    width: int,
    height: int,
) -> bool:
    """
    Pipe raw BGR frames to ffmpeg → H.264 MP4.

    Returns True on success.  H.264 requires even dimensions; we pad if needed.
    """
    # H.264 requires even width and height
    w = width  + (width  % 2)
    h = height + (height % 2)

    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-s", f"{w}x{h}",
        "-pix_fmt", "bgr24",
        "-r", str(fps_output),
        "-i", "pipe:0",
        "-vcodec", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", "fast",
        "-crf", "23",
        "-movflags", "+faststart",
        output_path,
    ]
    try:
        proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL
        )
        for frame in frame_list:
            if frame.shape[0] != h or frame.shape[1] != w:
                frame = cv2.resize(frame, (w, h))
            proc.stdin.write(frame.tobytes())
        proc.stdin.close()
        proc.wait()
        return proc.returncode == 0
    except Exception:
        return False


# ── cv2 fallback helper ───────────────────────────────────────────────────────

def _write_with_cv2(
    frame_list: List[np.ndarray],
    output_path: str,
    fps_output: float,
    width: int,
    height: int,
) -> bool:
    """Write frames with cv2.VideoWriter (fallback when ffmpeg is absent)."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps_output, (width, height))
    if not writer.isOpened():
        return False
    for frame in frame_list:
        writer.write(frame)
    writer.release()
    return True


# ── public write function ─────────────────────────────────────────────────────

def write_summary_video(
    selected_frame_paths: List[str],
    output_path: str,
    fps_output: float = 2.0,
    preserve_timing: bool = False,
    source_fps: Optional[float] = None,
    source_frame_indices: Optional[List[int]] = None,
    add_timestamp: bool = True,
    verbose: bool = True,
) -> str:
    """
    Write selected keyframes into a summary MP4.

    Uses ffmpeg H.264 encoding when available (preferred — correct colorspace
    on all platforms).  Falls back to cv2.VideoWriter with mp4v otherwise.

    Args:
        selected_frame_paths:   ordered list of absolute frame paths.
        output_path:            destination MP4 file.
        fps_output:             playback fps of the summary (default 2 fps
                                → each keyframe shown for 0.5 s).
        preserve_timing:        hold each keyframe proportionally to its
                                temporal gap in the source video.
        source_fps:             source video fps (for preserve_timing).
        source_frame_indices:   source frame indices (for preserve_timing).
        add_timestamp:          overlay frame-number badge on each frame.
        verbose:                print progress.

    Returns:
        output_path
    """
    if not selected_frame_paths:
        raise ValueError("selected_frame_paths is empty.")

    first = cv2.imread(selected_frame_paths[0])
    if first is None:
        raise IOError(f"Cannot read frame: {selected_frame_paths[0]}")
    h, w = first.shape[:2]

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    n = len(selected_frame_paths)

    # ── frame hold durations ─────────────────────────────────────────────
    if (preserve_timing and source_fps and source_frame_indices
            and len(source_frame_indices) == n):
        indices = list(source_frame_indices) + [source_frame_indices[-1] + 1]
        frame_durations = [
            max(1, round((indices[k + 1] - indices[k]) / source_fps * fps_output))
            for k in range(n)
        ]
    else:
        frame_durations = [1] * n

    # ── build frame sequence with optional timestamp overlay ─────────────
    all_frames: List[np.ndarray] = []
    for k, (path, hold) in enumerate(zip(selected_frame_paths, frame_durations)):
        img = cv2.imread(path)
        if img is None:
            continue
        if add_timestamp:
            label   = f"{k + 1}/{n}  {os.path.basename(path)}"
            overlay = img.copy()
            cv2.rectangle(overlay, (0, h - 24), (w, h), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)
            cv2.putText(img, label, (6, h - 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1,
                        cv2.LINE_AA)
        for _ in range(hold):
            all_frames.append(img.copy())

    if not all_frames:
        raise IOError("No frames could be read from selected_frame_paths.")

    # ── encode ───────────────────────────────────────────────────────────
    written = len(all_frames)
    if _ffmpeg_available():
        ok = _write_with_ffmpeg(all_frames, output_path, fps_output, w, h)
        if not ok:
            # ffmpeg failed for some reason — fall back to cv2
            ok = _write_with_cv2(all_frames, output_path, fps_output, w, h)
    else:
        ok = _write_with_cv2(all_frames, output_path, fps_output, w, h)

    if not ok:
        raise IOError(f"Could not write output video: {output_path}")

    if verbose:
        duration = written / fps_output
        backend  = "ffmpeg" if _ffmpeg_available() else "cv2"
        print(f"[sigvideo] Summary video ({backend}): {written} frames "
              f"@ {fps_output} fps = {duration:.1f}s → '{output_path}'")
    return output_path


# ── main pipeline function ────────────────────────────────────────────────────

def summarize_video(
    input_video: str,
    output_video: str,
    # ── summary control ───────────────────────────────────────────────────
    summary_length: Optional[int] = None,
    auto: bool = True,
    length_range: Optional[List[int]] = None,
    n_candidates: int = 10,
    n_comparisons: int = 10,
    # ── frame extraction ──────────────────────────────────────────────────
    fps_extract: float = 1.0,
    # ── output video ──────────────────────────────────────────────────────
    fps_output: float = 2.0,
    preserve_timing: bool = False,
    add_timestamp: bool = True,
    # ── signature params ──────────────────────────────────────────────────
    sig_order: int = SIG_ORDER,
    sig_size: Tuple = FRAME_SIZE,
    # ── misc ──────────────────────────────────────────────────────────────
    keep_frames: Optional[str] = None,
    verbose: bool = True,
) -> Tuple[str, List[str], float, float]:
    """
    **Primary function** — one call, video in → summarized video out.

    Extracts frames, selects keyframes via Signature Transform, and writes
    a condensed summary MP4 (H.264 via ffmpeg, or mp4v fallback).

    Args:
        input_video:      source video file path.
        output_video:     destination summary MP4 path.
        summary_length:   number of keyframes. None → auto-detect.
        auto:             use auto_length() when summary_length is None.
        length_range:     lengths to search in auto mode.
        n_candidates:     candidate summaries evaluated (paper default: 10).
        n_comparisons:    inner RMSE comparison budget (paper default: 10).
        fps_extract:      frame sampling rate for extraction (default: 1 fps).
        fps_output:       playback fps of the output summary (default: 2 fps).
        preserve_timing:  hold each keyframe proportional to its source gap.
        add_timestamp:    overlay frame index on each output frame.
        sig_order:        signature truncation order.
                          2 → default, fast (~0.6 ms/frame, dim=4 160);
                          3 → paper exact (~70 ms/frame, dim=266 304).
        sig_size:         frame resize for signature (paper: 64×64).
        keep_frames:      save extracted frames here (else temp dir).
        verbose:          print progress.

    Returns:
        (output_video, selected_frame_names, rmse_mean, rmse_std)
    """
    if not os.path.isfile(input_video):
        raise FileNotFoundError(f"Input video not found: {input_video}")

    frames_dir = keep_frames if keep_frames else tempfile.mkdtemp(prefix="sigvideo_tmp_")
    cleanup    = keep_frames is None

    try:
        cap = cv2.VideoCapture(input_video)
        source_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        cap.release()

        n_extracted = extract_frames(
            input_video, frames_dir, fps=fps_extract, verbose=verbose,
        )
        if n_extracted == 0:
            raise RuntimeError(f"No frames extracted from '{input_video}'.")

        if summary_length is not None:
            selected, rmse_mean, rmse_std = summarize(
                frames_dir,
                summary_length=summary_length,
                n_candidates=n_candidates,
                n_comparisons=n_comparisons,
                order=sig_order,
                size=sig_size,
                verbose=verbose,
            )
        else:
            if auto:
                _, selected, rmse_mean, rmse_std = auto_length(
                    frames_dir,
                    length_range=length_range,
                    n_candidates=n_candidates,
                    n_comparisons=n_comparisons,
                    order=sig_order,
                    size=sig_size,
                    verbose=verbose,
                )
            else:
                raise ValueError("Provide summary_length or set auto=True.")

        selected_sorted = sorted(selected)
        selected_paths  = [os.path.join(frames_dir, f) for f in selected_sorted]

        all_fnames   = sorted(f for f in os.listdir(frames_dir) if f.endswith(".png"))
        fname_to_idx = {f: i for i, f in enumerate(all_fnames)}
        src_indices  = [fname_to_idx.get(f, k) for k, f in enumerate(selected_sorted)]

        write_summary_video(
            selected_paths,
            output_video,
            fps_output=fps_output,
            preserve_timing=preserve_timing,
            source_fps=source_fps / (source_fps / fps_extract),
            source_frame_indices=src_indices,
            add_timestamp=add_timestamp,
            verbose=verbose,
        )

    finally:
        if cleanup:
            shutil.rmtree(frames_dir, ignore_errors=True)

    return output_video, selected_sorted, rmse_mean, rmse_std
