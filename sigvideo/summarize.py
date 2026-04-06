"""
summarize.py — RMSE(S̄, S̄_umin)|n baseline summarizer.

Selects the uniform random sample that minimises std(RMSE(S̄, S̄*))
over n candidates, as described in §2.2 of the paper.
"""

import os
import random
import numpy as np
from typing import List, Optional, Tuple

from .core import (
    mean_signature,
    rmse,
    SIG_ORDER,
    FRAME_SIZE,
)


def _collect_frame_paths(frames_dir: str) -> List[str]:
    """Return sorted list of image paths in a directory."""
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}
    paths = [
        os.path.join(frames_dir, f)
        for f in sorted(os.listdir(frames_dir))
        if os.path.splitext(f)[1].lower() in exts
    ]
    return paths


def summarize(
    frames_dir: str,
    summary_length: int,
    n_candidates: int = 10,
    n_comparisons: int = 10,
    order: int = SIG_ORDER,
    size: Tuple = FRAME_SIZE,
    verbose: bool = True,
) -> Tuple[List[str], float, float]:
    """
    Select the best uniform random summary of `summary_length` frames
    from `frames_dir` using RMSE(S̄, S̄_umin)|n_candidates.

    Algorithm (§2.2):
      For each candidate uniform random sample S̄_u:
        Compute std(RMSE(S̄, S̄_u)) over n_comparisons random draws.
      Return the candidate with the minimum std.

    Args:
        frames_dir:      directory of extracted video frames.
        summary_length:  number of frames to include in the summary.
        n_candidates:    number of candidate summaries to evaluate (n in the paper).
        n_comparisons:   inner comparison budget (10 in the paper).
        order:           signature truncation order (3 in the paper).
        size:            frame resize resolution (64×64 in the paper).
        verbose:         print progress.

    Returns:
        best_frames:  sorted list of selected frame filenames (basenames).
        best_mean:    mean RMSE of the best candidate.
        best_std:     std  RMSE of the best candidate.
    """
    all_paths = _collect_frame_paths(frames_dir)
    if not all_paths:
        raise FileNotFoundError(f"No image files found in: {frames_dir}")
    if summary_length > len(all_paths):
        raise ValueError(
            f"summary_length ({summary_length}) exceeds total frames ({len(all_paths)})."
        )

    if verbose:
        print(f"[sigvideo] Found {len(all_paths)} frames in '{frames_dir}'.")
        print(f"[sigvideo] Evaluating {n_candidates} candidates of length {summary_length}...")

    candidate_paths_list = []
    std_devs = []
    means = []

    for i in range(n_candidates):
        candidate = random.sample(all_paths, summary_length)
        candidate_paths_list.append(candidate)

        # Fix this candidate as S̄_u; compare against n_comparisons random draws
        candidate_mean = mean_signature(candidate, order, size)
        if candidate_mean is None:
            std_devs.append(float("inf"))
            means.append(float("inf"))
            continue

        rmse_vals = []
        for _ in range(n_comparisons):
            draw = random.sample(all_paths, summary_length)
            draw_mean = mean_signature(draw, order, size)
            if draw_mean is not None:
                rmse_vals.append(rmse(draw_mean, candidate_mean))

        std_devs.append(float(np.std(rmse_vals)))
        means.append(float(np.mean(rmse_vals)))

        if verbose:
            print(f"  Candidate {i+1:2d}/{n_candidates}: std={std_devs[-1]:.1f}  mean={means[-1]:.1f}")

    best_idx = int(np.argmin(std_devs))
    best_paths = sorted(candidate_paths_list[best_idx])
    best_frames = [os.path.basename(p) for p in best_paths]

    if verbose:
        print(f"\n[sigvideo] Best candidate: #{best_idx+1}  std={std_devs[best_idx]:.1f}  mean={means[best_idx]:.1f}")

    return best_frames, means[best_idx], std_devs[best_idx]


def auto_length(
    frames_dir: str,
    length_range: Optional[List[int]] = None,
    n_candidates: int = 10,
    n_comparisons: int = 10,
    order: int = SIG_ORDER,
    size: Tuple = FRAME_SIZE,
    verbose: bool = True,
) -> Tuple[int, List[str], float, float]:
    """
    Estimate an appropriate summary length from `length_range` by selecting
    the length whose RMSE(S̄, S̄_umin)|n has the minimum std, as suggested
    in §2.2 (Table 2 analysis).

    Args:
        frames_dir:    directory of extracted video frames.
        length_range:  list of lengths to evaluate. Defaults to
                       [5%, 10%, 15%, 20%, 25%] of total frame count.

    Returns:
        best_length, best_frames, best_mean, best_std
    """
    all_paths = _collect_frame_paths(frames_dir)
    n = len(all_paths)
    if length_range is None:
        length_range = sorted(set(max(1, int(n * p)) for p in [0.05, 0.10, 0.15, 0.20, 0.25]))

    if verbose:
        print(f"[sigvideo] Auto-length search over: {length_range}")

    best_length = None
    best_frames = None
    best_std = float("inf")
    best_mean = float("inf")

    for length in length_range:
        frames, mean_v, std_v = summarize(
            frames_dir, length, n_candidates, n_comparisons, order, size, verbose=False
        )
        if verbose:
            print(f"  length={length:4d}: std={std_v:.1f}  mean={mean_v:.1f}")
        if std_v < best_std:
            best_std = std_v
            best_mean = mean_v
            best_length = length
            best_frames = frames

    if verbose:
        print(f"\n[sigvideo] Best length: {best_length}  std={best_std:.1f}  mean={best_mean:.1f}")

    return best_length, best_frames, best_mean, best_std
