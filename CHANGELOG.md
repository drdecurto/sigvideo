# Changelog

All notable changes to `sigvideo` are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [0.3.0] — 2025-04-06

### Changed
- **Default `sig_order` changed from 3 to 2** across all functions and the CLI.
  Order 2 (~0.6 ms/frame) is ~100× faster than order 3 (~70 ms/frame) with
  comparable summarization quality for practical use.
  Pass `sig_order=3` explicitly to reproduce paper-exact results.

### Added
- `--sig-order` and `--sig-size` flags added to the CLI primary command and
  `sigvideo frames` subcommand.

---

## [0.2.2] — 2025-04-06

### Fixed
- **Green video bug**: `write_summary_video()` and the CLI now use an ffmpeg
  H.264 pipe backend (`rawvideo → libx264 yuv420p`) when ffmpeg is available,
  eliminating the colorspace corruption that occurred with cv2's `mp4v` codec
  on H.264-decoded source frames. Falls back to `cv2.VideoWriter` when ffmpeg
  is absent.
- **`NameError: Counter not defined`** in `sigvideo.vlm.extract_noun_queries()` —
  `from collections import Counter` was missing after a refactor.

### Changed
- Progress line now reports which backend was used:
  `[sigvideo] Summary video (ffmpeg): …` or `… (cv2): …`.

---

## [0.2.1] — 2025-04-06

### Added
- `summarize_video()` — primary end-to-end function: video in → summary video out.
- `write_summary_video()` — assemble frame paths into an MP4 directly.
- CLI primary syntax: `sigvideo INPUT.mp4 OUTPUT.mp4 [options]`.
- `preserve_timing` option: hold each keyframe proportional to its gap in the
  source video.
- `add_timestamp` overlay: frame index badge on each output frame.
- `keep_frames` parameter: save extracted frames to a named directory.
- `sigvideo vlm` CLI subcommand for text-conditioned summarization.

### Changed
- CLI redesigned: primary positional syntax replaces the old `summarize`
  subcommand as the entry point. Previous subcommands (`frames`, `score`, `vlm`)
  remain fully available.

---

## [0.2.0] — 2025-04-05

### Changed
- Package renamed from `sigsum` to `sigvideo`.
- License updated to MIT, copyright holders: J. de Curtò, I. de Zarzà.
- `pyproject.toml`: GitHub URL set to `github.com/drdecurto/sigvideo`.
- Citation key updated to `sigvideodecurto2023`.

---

## [0.1.0] — 2025-04-05

### Added
- Initial release.
- `summarize()` — RMSE(S̄, S̄_umin)|n baseline (NB01 from the paper).
- `auto_length()` — sweep a range of lengths, pick minimum-std candidate.
- `rmse_signature_score()` — RMSE(S̄, S̄*) scoring.
- `rmse_baseline()` — RMSE(S̄, S̄) confidence baseline.
- `mean_signature()` — element-wise mean of truncated signatures.
- `extract_frames()` — extract frames from a video at a target fps.
- `sigvideo.vlm` — optional OWL-ViT text-conditioned pipeline (§3 of the paper).
- CLI subcommands: `sigvideo summarize`, `sigvideo score`, `sigvideo vlm`.
- 33 unit tests covering metrics, summarizer, video extraction, pipeline,
  VLM NLP stage, and public API.
