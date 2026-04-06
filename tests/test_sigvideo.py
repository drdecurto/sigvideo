"""
tests/test_sigvideo.py — Unit tests for the sigvideo package.
"""

import os
import sys
import tempfile
import shutil
import numpy as np
import pytest
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import sigvideo
from sigvideo.core import rmse, mae, mean_signature, rmse_signature_score, rmse_baseline
from sigvideo.summarize import summarize, auto_length
from sigvideo.video import extract_frames


# ── fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def frames_dir():
    tmp = tempfile.mkdtemp(prefix="sigvideo_test_")
    np.random.seed(42)
    for i in range(40):
        img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        cv2.imwrite(os.path.join(tmp, f"frame_{i:04d}.png"), img)
    yield tmp
    shutil.rmtree(tmp, ignore_errors=True)


@pytest.fixture(scope="module")
def summary_dir(frames_dir):
    tmp = tempfile.mkdtemp(prefix="sigvideo_test_sum_")
    for i in range(0, 16, 2):
        shutil.copy(os.path.join(frames_dir, f"frame_{i:04d}.png"), tmp)
    yield tmp
    shutil.rmtree(tmp, ignore_errors=True)


@pytest.fixture(scope="module")
def synthetic_video(tmp_path_factory):
    """Create a tiny synthetic MP4 (20 frames, 5 fps)."""
    tmp = str(tmp_path_factory.mktemp("vid"))
    path = os.path.join(tmp, "test.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, 5.0, (64, 64))
    np.random.seed(7)
    for _ in range(20):
        frame = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        writer.write(frame)
    writer.release()
    yield path


# ── core metric tests ─────────────────────────────────────────────────────────

class TestMetrics:
    def test_rmse_identical(self):
        a = np.array([1.0, 2.0, 3.0])
        assert rmse(a, a) == pytest.approx(0.0)

    def test_rmse_known(self):
        assert rmse(np.zeros(2), np.ones(2)) == pytest.approx(1.0)

    def test_mae_known(self):
        assert mae(np.array([0.0, 2.0]), np.array([1.0, 1.0])) == pytest.approx(1.0)

    def test_rmse_symmetry(self):
        a = np.array([1.0, 3.0, 5.0])
        b = np.array([2.0, 4.0, 6.0])
        assert rmse(a, b) == pytest.approx(rmse(b, a))

    def test_mean_signature_shape(self, frames_dir):
        paths = [os.path.join(frames_dir, f)
                 for f in sorted(os.listdir(frames_dir))[:5]]
        sig = mean_signature(paths)
        assert sig is not None
        assert sig.ndim == 1
        assert sig.shape[0] > 0

    def test_mean_signature_single_frame(self, frames_dir):
        p = os.path.join(frames_dir, "frame_0000.png")
        sig = mean_signature([p])
        assert sig is not None

    def test_mean_signature_missing_file(self):
        result = mean_signature(["/nonexistent/path.png"])
        assert result is None

    def test_mean_signature_reproducible(self, frames_dir):
        paths = [os.path.join(frames_dir, f)
                 for f in sorted(os.listdir(frames_dir))[:3]]
        s1 = mean_signature(paths)
        s2 = mean_signature(paths)
        np.testing.assert_array_equal(s1, s2)


# ── scoring tests ─────────────────────────────────────────────────────────────

class TestScoring:
    def test_rmse_signature_score_shape(self, frames_dir, summary_dir):
        vp = [os.path.join(frames_dir, f) for f in sorted(os.listdir(frames_dir))]
        sp = [os.path.join(summary_dir, f) for f in sorted(os.listdir(summary_dir))]
        vals, mean_v, std_v = rmse_signature_score(sp, vp, n_comparisons=5)
        assert len(vals) == 5
        assert mean_v > 0
        assert std_v >= 0

    def test_rmse_baseline_shape(self, frames_dir):
        vp = [os.path.join(frames_dir, f) for f in sorted(os.listdir(frames_dir))]
        vals, mean_v, std_v = rmse_baseline(vp, summary_length=8, n_comparisons=5)
        assert len(vals) == 5
        assert mean_v > 0

    def test_rmse_signature_score_raises_on_bad_summary(self, frames_dir):
        vp = [os.path.join(frames_dir, f) for f in sorted(os.listdir(frames_dir))]
        with pytest.raises(ValueError):
            rmse_signature_score(["/bad/path.png"], vp, n_comparisons=2)


# ── summarizer tests ──────────────────────────────────────────────────────────

class TestSummarize:
    def test_correct_length(self, frames_dir):
        frames, mean_v, std_v = summarize(
            frames_dir, summary_length=8, n_candidates=3, n_comparisons=3, verbose=False
        )
        assert len(frames) == 8
        assert all(f.endswith(".png") for f in frames)

    def test_frames_exist_on_disk(self, frames_dir):
        frames, _, _ = summarize(
            frames_dir, summary_length=5, n_candidates=3, n_comparisons=3, verbose=False
        )
        for fname in frames:
            assert os.path.exists(os.path.join(frames_dir, fname))

    def test_no_duplicate_frames(self, frames_dir):
        frames, _, _ = summarize(
            frames_dir, summary_length=10, n_candidates=3, n_comparisons=3, verbose=False
        )
        assert len(frames) == len(set(frames))

    def test_length_exceeds_raises(self, frames_dir):
        with pytest.raises(ValueError):
            summarize(frames_dir, summary_length=9999, n_candidates=2,
                      n_comparisons=2, verbose=False)

    def test_empty_dir_raises(self):
        with tempfile.TemporaryDirectory() as d:
            with pytest.raises(FileNotFoundError):
                summarize(d, summary_length=5, verbose=False)

    def test_auto_length_picks_valid(self, frames_dir):
        best_len, best_frames, best_mean, best_std = auto_length(
            frames_dir, length_range=[5, 8, 12],
            n_candidates=3, n_comparisons=3, verbose=False,
        )
        assert best_len in [5, 8, 12]
        assert len(best_frames) == best_len
        assert best_std >= 0

    def test_auto_length_default_range(self, frames_dir):
        best_len, best_frames, _, _ = auto_length(
            frames_dir, n_candidates=3, n_comparisons=3, verbose=False
        )
        n = len(os.listdir(frames_dir))
        assert 1 <= best_len <= n


# ── video extraction tests ────────────────────────────────────────────────────

class TestVideoExtraction:
    def test_extract_creates_pngs(self, synthetic_video):
        with tempfile.TemporaryDirectory() as out:
            n = extract_frames(synthetic_video, out, fps=1.0, verbose=False)
            pngs = [f for f in os.listdir(out) if f.endswith(".png")]
            assert len(pngs) == n
            assert n > 0

    def test_extract_all_frames(self, synthetic_video):
        with tempfile.TemporaryDirectory() as out:
            n = extract_frames(synthetic_video, out, fps=None, verbose=False)
            assert n == 20  # 20 frames written by fixture

    def test_extract_bad_path_raises(self):
        with pytest.raises(IOError):
            extract_frames("/nonexistent/video.mp4", "/tmp", verbose=False)


# ── VLM NLP tests (no GPU/model weights needed) ───────────────────────────────

class TestVLMNLP:
    def test_extract_noun_queries_basic(self):
        try:
            from sigvideo.vlm import extract_noun_queries
        except ImportError:
            pytest.skip("sigvideo[vlm] not installed")
        text = "The catapult launches a projectile. The projectile flies through the air."
        try:
            queries = extract_noun_queries(text, top_n=5)
        except LookupError:
            pytest.skip("NLTK data not available in this environment")
        assert isinstance(queries, list)
        assert len(queries) <= 5
        assert all(isinstance(q, str) for q in queries)

    def test_load_subtitles(self):
        try:
            from sigvideo.vlm import load_subtitles
        except ImportError:
            pytest.skip("sigvideo[vlm] not installed")
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Hello world\nThis is a test\n")
            path = f.name
        try:
            text = load_subtitles(path)
            assert "Hello" in text
            assert "test" in text
        finally:
            os.unlink(path)


# ── public API ────────────────────────────────────────────────────────────────

class TestPublicAPI:
    def test_all_symbols_callable(self):
        for name in ["summarize", "auto_length", "extract_frames",
                     "rmse_signature_score", "rmse_baseline",
                     "mean_signature", "rmse", "mae"]:
            assert callable(getattr(sigvideo, name)), f"sigvideo.{name} not callable"

    def test_version_string(self):
        assert isinstance(sigvideo.__version__, str)
        assert sigvideo.__version__ == "0.3.0"

    def test_vlm_lazy_import_error(self):
        """Calling sigvideo.summarize_vlm without torch installed should raise ImportError."""
        try:
            import torch  # noqa
            pytest.skip("torch is installed; lazy-import error path not triggered")
        except ImportError:
            fn = sigvideo.summarize_vlm  # attribute access succeeds (lazy import)
            assert callable(fn)
            with pytest.raises(ImportError, match="pip install sigvideo"):
                fn(".", ["test"])


# ── pipeline (video → video) tests ───────────────────────────────────────────

class TestPipeline:
    """Tests for the primary summarize_video() end-to-end function."""

    @pytest.fixture(scope="class")
    def synthetic_video(self, tmp_path_factory):
        """Create a 20-frame 5fps synthetic MP4."""
        tmp   = str(tmp_path_factory.mktemp("pipeline_vid"))
        vpath = os.path.join(tmp, "test_input.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(vpath, fourcc, 5.0, (128, 128))
        np.random.seed(99)
        for _ in range(30):
            writer.write(np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8))
        writer.release()
        return vpath

    def test_output_file_created(self, synthetic_video, tmp_path):
        from sigvideo import summarize_video
        out = str(tmp_path / "summary.mp4")
        result_path, selected, mean_v, std_v = summarize_video(
            synthetic_video, out,
            summary_length=4,
            fps_extract=5.0,
            fps_output=2.0,
            n_candidates=3,
            n_comparisons=3,
            verbose=False,
        )
        assert os.path.isfile(out), "Output video not created"
        assert os.path.getsize(out) > 0, "Output video is empty"
        assert result_path == out

    def test_selected_frames_count(self, synthetic_video, tmp_path):
        from sigvideo import summarize_video
        out = str(tmp_path / "summary4.mp4")
        _, selected, _, _ = summarize_video(
            synthetic_video, out,
            summary_length=4,
            fps_extract=5.0,
            fps_output=2.0,
            n_candidates=3,
            n_comparisons=3,
            verbose=False,
        )
        assert len(selected) == 4
        assert len(set(selected)) == 4, "Duplicate frames selected"

    def test_output_is_shorter_than_input(self, synthetic_video, tmp_path):
        from sigvideo import summarize_video
        out = str(tmp_path / "summary_short.mp4")
        summarize_video(
            synthetic_video, out,
            summary_length=4,
            fps_extract=5.0,
            fps_output=2.0,
            n_candidates=3,
            n_comparisons=3,
            verbose=False,
        )
        in_size  = os.path.getsize(synthetic_video)
        out_size = os.path.getsize(out)
        # Summary must be a valid video (not empty); it may be larger or smaller
        # byte-wise depending on codec, but frame count must be fewer
        cap_in  = cv2.VideoCapture(synthetic_video)
        cap_out = cv2.VideoCapture(out)
        n_in  = int(cap_in.get(cv2.CAP_PROP_FRAME_COUNT))
        n_out = int(cap_out.get(cv2.CAP_PROP_FRAME_COUNT))
        cap_in.release(); cap_out.release()
        assert n_out < n_in, f"Summary ({n_out} frames) not shorter than input ({n_in} frames)"
        assert n_out > 0

    def test_auto_length(self, synthetic_video, tmp_path):
        from sigvideo import summarize_video
        out = str(tmp_path / "summary_auto.mp4")
        _, selected, mean_v, std_v = summarize_video(
            synthetic_video, out,
            summary_length=None,
            auto=True,
            fps_extract=5.0,
            fps_output=2.0,
            n_candidates=3,
            n_comparisons=3,
            verbose=False,
        )
        assert os.path.isfile(out)
        assert len(selected) > 0
        assert std_v >= 0

    def test_output_video_readable(self, synthetic_video, tmp_path):
        from sigvideo import summarize_video
        out = str(tmp_path / "summary_read.mp4")
        summarize_video(
            synthetic_video, out,
            summary_length=3,
            fps_extract=5.0,
            fps_output=2.0,
            n_candidates=3,
            n_comparisons=3,
            verbose=False,
        )
        cap = cv2.VideoCapture(out)
        assert cap.isOpened(), "Output video cannot be opened by cv2"
        ret, frame = cap.read()
        cap.release()
        assert ret, "Cannot read first frame from output video"
        assert frame is not None

    def test_keep_frames_dir(self, synthetic_video, tmp_path):
        from sigvideo import summarize_video
        out        = str(tmp_path / "summary_kf.mp4")
        keep_dir   = str(tmp_path / "kept_frames")
        summarize_video(
            synthetic_video, out,
            summary_length=3,
            fps_extract=5.0,
            fps_output=2.0,
            n_candidates=3,
            n_comparisons=3,
            keep_frames=keep_dir,
            verbose=False,
        )
        assert os.path.isdir(keep_dir)
        pngs = [f for f in os.listdir(keep_dir) if f.endswith(".png")]
        assert len(pngs) > 0, "No frames saved to keep_frames dir"

    def test_missing_input_raises(self, tmp_path):
        from sigvideo import summarize_video
        with pytest.raises(FileNotFoundError):
            summarize_video("/nonexistent/video.mp4", str(tmp_path / "out.mp4"),
                            summary_length=3, verbose=False)

    def test_write_summary_video_standalone(self, frames_dir, tmp_path):
        from sigvideo import write_summary_video
        paths = [os.path.join(frames_dir, f)
                 for f in sorted(os.listdir(frames_dir))[:5]]
        out = str(tmp_path / "standalone.mp4")
        result = write_summary_video(paths, out, fps_output=2.0,
                                     add_timestamp=True, verbose=False)
        assert result == out
        assert os.path.isfile(out) and os.path.getsize(out) > 0


class TestPublicAPIV2:
    def test_primary_exports(self):
        import sigvideo
        assert callable(sigvideo.summarize_video)
        assert callable(sigvideo.write_summary_video)
        assert sigvideo.__version__ == "0.3.0"
