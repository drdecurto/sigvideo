"""
core.py — Signature Transform metrics for video summarization.

Based on:
  @article{sigvideodecurto2023} — de Curtò & de Zarzà et al., Electronics 2023, 12:1735.
  https://doi.org/10.3390/electronics12071735
"""

import numpy as np
import iisignature
import cv2
import os
import random
from typing import List, Optional, Tuple

# Default parameters from the paper (§4, paragraph 3)
SIG_ORDER = 2
FRAME_SIZE = (64, 64)


def load_frame_as_sig(path: str, order: int = SIG_ORDER, size: Tuple = FRAME_SIZE) -> Optional[np.ndarray]:
    """Load a single image file and return its iisignature signature."""
    img = cv2.imread(path)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, size).astype(float)
    return iisignature.sig(img, order)


def mean_signature(paths: List[str], order: int = SIG_ORDER, size: Tuple = FRAME_SIZE) -> Optional[np.ndarray]:
    """Compute element-wise mean of signatures over a list of frame paths."""
    sigs = []
    for p in paths:
        s = load_frame_as_sig(p, order, size)
        if s is not None:
            sigs.append(s)
    if not sigs:
        return None
    return np.mean(sigs, axis=0)


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    """Root mean squared error between two signature vectors."""
    return float(np.sqrt(np.mean((a - b) ** 2)))


def mae(a: np.ndarray, b: np.ndarray) -> float:
    """Mean absolute error between two signature vectors."""
    return float(np.mean(np.abs(a - b)))


def rmse_signature_score(
    summary_paths: List[str],
    video_paths: List[str],
    n_comparisons: int = 10,
    order: int = SIG_ORDER,
    size: Tuple = FRAME_SIZE,
) -> Tuple[List[float], float, float]:
    """
    Compute RMSE(S̄, S̄*): score a summary against n random uniform samples.

    Lower std → the summary covers harmonic components well.

    Returns:
        rmse_values: list of 10 RMSE values
        mean: mean RMSE
        std: standard deviation of RMSE
    """
    summary_mean = mean_signature(summary_paths, order, size)
    if summary_mean is None:
        raise ValueError("Could not compute signatures for summary frames.")

    k = len(summary_paths)
    rmse_values = []
    for _ in range(n_comparisons):
        sample = random.sample(video_paths, min(k, len(video_paths)))
        sample_mean = mean_signature(sample, order, size)
        if sample_mean is not None:
            rmse_values.append(rmse(sample_mean, summary_mean))

    return rmse_values, float(np.mean(rmse_values)), float(np.std(rmse_values))


def rmse_baseline(
    video_paths: List[str],
    summary_length: int,
    n_comparisons: int = 10,
    order: int = SIG_ORDER,
    size: Tuple = FRAME_SIZE,
) -> Tuple[List[float], float, float]:
    """
    Compute RMSE(S̄, S̄): baseline between two random uniform samples of the same length.

    Used as a confidence interval / reference level.

    Returns:
        rmse_values, mean, std
    """
    k = summary_length
    rmse_values = []
    for _ in range(n_comparisons):
        s1 = random.sample(video_paths, min(k, len(video_paths)))
        s2 = random.sample(video_paths, min(k, len(video_paths)))
        m1 = mean_signature(s1, order, size)
        m2 = mean_signature(s2, order, size)
        if m1 is not None and m2 is not None:
            rmse_values.append(rmse(m1, m2))
    return rmse_values, float(np.mean(rmse_values)), float(np.std(rmse_values))
