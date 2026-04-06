# sigvideo

[![PyPI version](https://img.shields.io/pypi/v/sigvideo?color=blue)](https://pypi.org/project/sigvideo/)
[![Python](https://img.shields.io/pypi/pyversions/sigvideo)](https://pypi.org/project/sigvideo/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/drdecurto/sigvideo/blob/main/sigvideo_colab_v6.ipynb)
[![DOI](https://img.shields.io/badge/DOI-10.3390%2Felectronics12071735-blue)](https://doi.org/10.3390/electronics12071735)

**Automatic video summarization via the Signature Transform.**

Given any video file, `sigvideo` selects the most representative keyframes and assembles them into a condensed summary video — no annotations, no training data, no GPU required.

```python
from sigvideo import summarize_video
summarize_video("lecture.mp4", "summary.mp4")
```

```bash
sigvideo lecture.mp4 summary.mp4
```

Based on the paper:

> **de Curtò, J.; de Zarzà, I.; Roig, G.; Calafate, C.T.**
> *Summarization of Videos with the Signature Transform.*
> Electronics **2023**, 12, 1735.
> https://doi.org/10.3390/electronics12071735

---

## How it works

The [Signature Transform](https://arxiv.org/abs/1405.4537) is a rough equivalent of the Fourier Transform for paths: instead of frequency it captures order and area (iterated integrals). Applied to video frames treated as a temporal path, it provides a compact harmonic descriptor of each frame.

`sigvideo` evaluates *n* candidate uniform random samples and selects the one whose element-wise mean signature has the **lowest standard deviation** when compared against repeated random draws from the full video — meaning it best preserves the harmonic components of the original. This baseline is called **RMSE(S̄, S̄_umin)|n** in the paper and achieves 100% positive cases on the 28-video benchmark without any human annotations or training.

---

## Installation

```bash
pip install sigvideo
```

**Optional — text-conditioned VLM summarization** (OWL-ViT via HuggingFace, requires PyTorch):

```bash
pip install sigvideo[vlm]
```

### Dependencies

| Package | Purpose | License |
|---------|---------|---------|
| `iisignature` | Signature Transform | MIT |
| `opencv-python-headless` | Frame I/O | Apache 2.0 |
| `numpy` | Numerical ops | BSD-3 |

`sigvideo` is MIT-licensed. All dependencies use permissive licenses — no copyleft, no additional obligations.

---

## Quick start

### Python

```python
from sigvideo import summarize_video

# One call — video in, summary video out
out, frames, rmse_mean, rmse_std = summarize_video(
    "input.mp4",
    "summary.mp4",
)
print(f"{len(frames)} keyframes selected")
```

```python
# Auto-detect best length, keep extracted frames
out, frames, _, _ = summarize_video(
    "input.mp4",
    "summary.mp4",
    fps_extract = 1.0,    # extract 1 frame/s
    fps_output  = 2.0,    # each keyframe shown 0.5 s
    keep_frames = "./frames",
)
```

```python
# Paper-exact reproduction (sig_order=3, ~100x slower)
summarize_video("input.mp4", "summary.mp4", sig_order=3)
```

### CLI

```bash
# Minimal
sigvideo input.mp4 summary.mp4

# With options
sigvideo input.mp4 summary.mp4 \
    --length 20        \   # fixed keyframe count (default: auto)
    --fps-extract 1    \   # extraction rate
    --fps-output 2     \   # output playback rate
    --sig-order 3      \   # paper-exact (default: 2, fast)
    --no-timestamp         # suppress frame badge overlay

# Score a summary against original frames
sigvideo score --frames-dir ./frames --summary-dir ./keyframes

# Text-conditioned summarization (requires sigvideo[vlm])
sigvideo vlm --frames-dir ./frames --subtitles transcript.txt
```

---

## Signature order

The signature truncation order controls the speed/quality tradeoff:

| `sig_order` | Time/frame | Signature dim | Use case |
|-------------|-----------|---------------|---------|
| **2** (default) | ~0.6 ms | 4,160 | Everyday use, long videos |
| 3 (paper) | ~70 ms | 266,304 | Paper-exact reproduction |

Both produce valid summaries. Order 2 is the practical default; order 3 reproduces the numbers in Tables 1–5 of the paper exactly.

---

## Full API

### `summarize_video(input_video, output_video, **kwargs)`

Primary function. Chains frame extraction → signature selection → H.264 video writing.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `summary_length` | `None` | Number of keyframes. `None` → auto-detect. |
| `n_candidates` | `10` | Candidate summaries evaluated (paper default). |
| `n_comparisons` | `10` | Inner RMSE comparison budget (paper default). |
| `fps_extract` | `1.0` | Frame sampling rate for extraction. |
| `fps_output` | `2.0` | Playback fps of output video. |
| `preserve_timing` | `False` | Hold each frame proportional to its source gap. |
| `add_timestamp` | `True` | Overlay frame index badge. |
| `sig_order` | `2` | Signature order (2=fast, 3=paper exact). |
| `sig_size` | `(64,64)` | Frame resize for signature computation. |
| `keep_frames` | `None` | Save extracted frames to this directory. |

Returns `(output_path, selected_frame_names, rmse_mean, rmse_std)`.

### `write_summary_video(paths, output_path, **kwargs)`

Assemble a list of frame paths into an MP4 directly (H.264 via ffmpeg when available).

### `summarize(frames_dir, summary_length, **kwargs)`

Frame-level: select keyframes from a directory of images, returns `(frame_names, rmse_mean, rmse_std)`.

### `auto_length(frames_dir, length_range=None, **kwargs)`

Sweep a range of lengths and return the one with lowest `std(RMSE)`.

### `rmse_signature_score(summary_paths, video_paths, n_comparisons=10)`

Score an existing summary: returns `(values, mean, std)` of RMSE(S̄, S̄*).

### `rmse_baseline(video_paths, summary_length, n_comparisons=10)`

Compute the RMSE(S̄, S̄) baseline (two random samples vs each other).

### `extract_frames(video_path, output_dir, fps=1.0)`

Extract frames from a video at a target fps, saves PNGs to `output_dir`.

---

## VLM text-conditioned pipeline

When `sigvideo[vlm]` is installed, frames can also be selected by matching
video content against noun queries extracted from a subtitle transcript:

```python
from sigvideo.vlm import summarize_vlm_from_subtitles

frames, queries = summarize_vlm_from_subtitles(
    frames_dir   = "./frames",
    subtitles_path = "transcript.txt",
    top_n_queries  = 20,
    score_threshold = 0.025,
)
```

Or from the CLI:

```bash
sigvideo vlm \
    --frames-dir ./frames \
    --subtitles  transcript.txt \
    --threshold  0.025 \
    --output     summary_vlm.txt
```

This uses [OWL-ViT](https://arxiv.org/abs/2205.06230) (Minderer et al., 2022) for
zero-shot text-conditioned object detection, following §3 of the paper.

---

## Metrics

The package exposes the three metrics defined in §2.1 of the paper:

| Metric | Meaning |
|--------|---------|
| `RMSE(S̄, S̄*)` | Error between summary spectrum and random uniform sample. Low std → good coverage. |
| `RMSE(S̄, S̄)` | Error between two random samples. Serves as confidence baseline. |
| `RMSE(S̄, S̄_umin)\|n` | The best candidate among *n* random samples by minimum std. |

A summary **passes** when `std(RMSE(S̄, S̄*)) ≤ std(RMSE(S̄, S̄))`.

---

## Citation

```bibtex
@article{sigvideodecurto2023,
  title   = {Summarization of Videos with the Signature Transform},
  author  = {de Curt{\`o}, J. and de Zarz{\`a}, I. and Roig, G. and Calafate, C.T.},
  journal = {Electronics},
  volume  = {12},
  number  = {7},
  pages   = {1735},
  year    = {2023},
  doi     = {10.3390/electronics12071735}
}
```

---

## License

MIT — Copyright (c) 2023 J. de Curtò, I. de Zarzà

See [LICENSE](LICENSE) for the full text.
