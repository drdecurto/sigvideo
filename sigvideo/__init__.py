"""
sigvideo — Video Summarization with the Signature Transform.

Primary usage:
    from sigvideo import summarize_video
    summarize_video("lecture.mp4", "summary.mp4")

Reference:
    @article{sigvideodecurto2023,
      title   = {Summarization of Videos with the Signature Transform},
      author  = {de Curtò, J. and de Zarzà, I. and Roig, G. and Calafate, C.T.},
      journal = {Electronics},
      volume  = {12}, number = {7}, pages = {1735}, year = {2023},
      doi     = {10.3390/electronics12071735}
    }

License:
    MIT -- Copyright (c) 2023 J. de Curtò, I. de Zarzà
"""

__version__ = "0.3.0"
__authors__ = ["J. de Curtò", "I. de Zarzà"]
__license__ = "MIT"

# Primary API
from .pipeline import summarize_video, write_summary_video

# Frame-level API (secondary)
from .core import rmse_signature_score, rmse_baseline, mean_signature, rmse, mae
from .summarize import summarize, auto_length
from .video import extract_frames

__all__ = [
    # primary
    "summarize_video",
    "write_summary_video",
    # frame-level
    "summarize",
    "auto_length",
    "extract_frames",
    "rmse_signature_score",
    "rmse_baseline",
    "mean_signature",
    "rmse",
    "mae",
    # VLM (requires sigvideo[vlm])
    "summarize_vlm",
    "summarize_vlm_from_subtitles",
    "extract_noun_queries",
]


def __getattr__(name: str):
    _vlm = {"summarize_vlm", "summarize_vlm_from_subtitles", "extract_noun_queries"}
    if name in _vlm:
        from . import vlm as _v
        return getattr(_v, name)
    raise AttributeError(f"module 'sigvideo' has no attribute {name!r}")
