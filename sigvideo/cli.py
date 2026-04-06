"""
cli.py — Command-line interface for sigvideo.

Primary usage (video → summarized video):
    sigvideo input.mp4 output_summary.mp4
    sigvideo input.mp4 output_summary.mp4 --length 30 --sig-order 2

Secondary subcommands:
    sigvideo frames  --frames-dir ./f --length 20 --output summary.txt
    sigvideo score   --frames-dir ./f --summary-dir ./s
    sigvideo vlm     --frames-dir ./f --subtitles transcript.txt
"""

import argparse
import os
import sys
import shutil
import tempfile


# ── primary: video → video ────────────────────────────────────────────────────

def cmd_video(args):
    from .pipeline import summarize_video

    sig_size = tuple(int(x) for x in args.sig_size.replace("x", ",").split(","))[:2]

    out, selected, mean_v, std_v = summarize_video(
        input_video=args.input,
        output_video=args.output,
        summary_length=args.length,
        auto=True,
        length_range=[int(x) for x in args.length_range.split(",")] if args.length_range else None,
        n_candidates=args.n_candidates,
        n_comparisons=args.n_comparisons,
        fps_extract=args.fps_extract,
        fps_output=args.fps_output,
        preserve_timing=args.preserve_timing,
        add_timestamp=not args.no_timestamp,
        sig_order=args.sig_order,
        sig_size=sig_size,
        keep_frames=args.keep_frames,
        verbose=not args.quiet,
    )
    if not args.quiet:
        print(f"\n[sigvideo] {len(selected)} keyframes → '{out}'")
        print(f"[sigvideo] RMSE  mean={mean_v:.0f}  std={std_v:.0f}")


# ── secondary: frame-level summarize ─────────────────────────────────────────

def cmd_frames(args):
    from .summarize import summarize, auto_length
    from .video import extract_frames

    frames_dir = args.frames_dir
    tmp_dir = None
    if args.video:
        tmp_dir = tempfile.mkdtemp(prefix="sigvideo_frames_")
        extract_frames(args.video, tmp_dir, fps=args.fps_extract, verbose=not args.quiet)
        frames_dir = tmp_dir

    try:
        if args.auto_length:
            lr = [int(x) for x in args.length_range.split(",")] if args.length_range else None
            _, best_frames, mean_v, std_v = auto_length(
                frames_dir, length_range=lr,
                n_candidates=args.n_candidates,
                n_comparisons=args.n_comparisons,
                order=args.sig_order,
                verbose=not args.quiet,
            )
        else:
            if args.length is None:
                print("[sigvideo] ERROR: --length required (or use --auto-length).", file=sys.stderr)
                sys.exit(1)
            best_frames, mean_v, std_v = summarize(
                frames_dir, summary_length=args.length,
                n_candidates=args.n_candidates,
                n_comparisons=args.n_comparisons,
                order=args.sig_order,
                verbose=not args.quiet,
            )
        _write_frame_output(best_frames, args.output, frames_dir, args.copy_frames, args.quiet)
    finally:
        if tmp_dir:
            shutil.rmtree(tmp_dir, ignore_errors=True)


# ── secondary: score ──────────────────────────────────────────────────────────

def cmd_score(args):
    from .core import rmse_signature_score, rmse_baseline

    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}

    def get_paths(d):
        return sorted(
            os.path.join(d, f) for f in os.listdir(d)
            if os.path.splitext(f)[1].lower() in exts
        )

    video_paths   = get_paths(args.frames_dir)
    summary_paths = get_paths(args.summary_dir)

    if not video_paths:
        print(f"[sigvideo] ERROR: no frames in '{args.frames_dir}'", file=sys.stderr); sys.exit(1)
    if not summary_paths:
        print(f"[sigvideo] ERROR: no frames in '{args.summary_dir}'", file=sys.stderr); sys.exit(1)

    print(f"[sigvideo] Scoring {len(summary_paths)} summary frames vs {len(video_paths)} source frames\u2026")
    _, mean_star, std_star = rmse_signature_score(summary_paths, video_paths,
                                                   n_comparisons=args.n_comparisons)
    _, base_mean, base_std = rmse_baseline(video_paths, len(summary_paths),
                                           n_comparisons=args.n_comparisons)

    print(f"\n  RMSE(S\u0304,S\u0304*)  mean={mean_star:.0f}  std={std_star:.0f}")
    print(f"  RMSE(S\u0304,S\u0304)   mean={base_mean:.0f}  std={base_std:.0f}")
    verdict = "\u2713 PASS" if std_star <= base_std else "\u2717 FAIL"
    print(f"\n  {verdict}")


# ── secondary: vlm ────────────────────────────────────────────────────────────

def cmd_vlm(args):
    try:
        from .vlm import summarize_vlm, summarize_vlm_from_subtitles
    except ImportError as e:
        print(f"[sigvideo] ERROR: {e}", file=sys.stderr); sys.exit(1)

    if args.queries:
        queries = [q.strip() for q in args.queries.split(",") if q.strip()]
        frames  = summarize_vlm(args.frames_dir, queries,
                                score_threshold=args.threshold,
                                device=args.device,
                                verbose=not args.quiet)
    elif args.subtitles:
        frames, _ = summarize_vlm_from_subtitles(
            args.frames_dir, args.subtitles,
            top_n_queries=args.top_n,
            score_threshold=args.threshold,
            device=args.device,
            verbose=not args.quiet,
        )
    else:
        print("[sigvideo] ERROR: --subtitles or --queries required.", file=sys.stderr); sys.exit(1)

    _write_frame_output(frames, args.output, args.frames_dir, args.copy_frames, args.quiet)


# ── helpers ───────────────────────────────────────────────────────────────────

def _write_frame_output(frames, output_file, frames_dir, copy_dir, quiet):
    if output_file:
        with open(output_file, "w") as f:
            for fname in frames:
                f.write(fname + "\n")
        if not quiet:
            print(f"[sigvideo] Frame list written to '{output_file}'.")
    else:
        for fname in frames:
            print(fname)

    if copy_dir and frames_dir:
        os.makedirs(copy_dir, exist_ok=True)
        for fname in frames:
            src = os.path.join(frames_dir, fname)
            dst = os.path.join(copy_dir, fname)
            if os.path.exists(src):
                shutil.copy2(src, dst)
        if not quiet:
            print(f"[sigvideo] Frames copied to '{copy_dir}'.")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    import sys as _sys

    parser = argparse.ArgumentParser(
        prog="sigvideo",
        description=(
            "Video summarization via the Signature Transform.\n"
            "  Primary:  sigvideo INPUT.mp4 OUTPUT_SUMMARY.mp4\n"
            "  Advanced: sigvideo {frames|score|vlm} ...\n\n"
            "Reference: de Curt\u00f2 & de Zarz\u00e0 et al., Electronics 2023, 12:1735."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command")

    # ── PRIMARY positional shortcut: sigvideo INPUT OUTPUT ─────────────────
    _argv = _sys.argv[1:]
    _is_video_shortcut = (
        len(_argv) >= 2
        and not _argv[0].startswith("-")
        and _argv[0] not in ("frames", "score", "vlm", "-h", "--help")
        and os.path.splitext(_argv[0])[1].lower() in
            {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ""}
    )

    if _is_video_shortcut:
        p = argparse.ArgumentParser(
            prog="sigvideo",
            description="sigvideo INPUT.mp4 OUTPUT_SUMMARY.mp4 [options]",
        )
        p.add_argument("input",  help="Input video file.")
        p.add_argument("output", help="Output summary video file.")
        p.add_argument("--length",          type=int,   default=None,
                       help="Number of keyframes. Default: auto-detect.")
        p.add_argument("--length-range",    metavar="L1,L2,...",
                       help="Lengths to search in auto mode.")
        p.add_argument("--fps-extract",     type=float, default=1.0,
                       help="Frame sampling rate (default: 1 fps).")
        p.add_argument("--fps-output",      type=float, default=2.0,
                       help="Playback fps of summary video (default: 2 fps).")
        p.add_argument("--preserve-timing", action="store_true",
                       help="Hold each keyframe proportionally to its source gap.")
        p.add_argument("--no-timestamp",    action="store_true",
                       help="Suppress frame-number overlay.")
        p.add_argument("--n-candidates",    type=int,   default=10)
        p.add_argument("--n-comparisons",   type=int,   default=10)
        p.add_argument("--sig-order",       type=int,   default=2,
                       help="Signature order: 2=default (fast), 3=paper exact (~100x slower).")
        p.add_argument("--sig-size",        metavar="WxH", default="64x64",
                       help="Frame resize for signature (default: 64x64).")
        p.add_argument("--keep-frames",     metavar="DIR",
                       help="Save extracted frames here (otherwise temp dir).")
        p.add_argument("--quiet",           action="store_true")
        args = p.parse_args()
        args.func = cmd_video
        args.func(args)
        return

    # ── subcommands ─────────────────────────────────────────────────────────

    # frames
    p_fr = subparsers.add_parser("frames",
        help="Select keyframes from a directory or video file (frame-list output).")
    src = p_fr.add_mutually_exclusive_group()
    src.add_argument("--frames-dir",    metavar="DIR")
    src.add_argument("--video",         metavar="FILE")
    p_fr.add_argument("--fps-extract",  type=float, default=1.0)
    p_fr.add_argument("--length",       type=int,   default=None)
    p_fr.add_argument("--auto-length",  action="store_true")
    p_fr.add_argument("--length-range", metavar="L1,L2,...")
    p_fr.add_argument("--n-candidates", type=int, default=10)
    p_fr.add_argument("--n-comparisons",type=int, default=10)
    p_fr.add_argument("--sig-order",    type=int, default=2,
                       help="Signature order: 2=default (fast), 3=paper exact.")
    p_fr.add_argument("--output",       metavar="FILE")
    p_fr.add_argument("--copy-frames",  metavar="DIR")
    p_fr.add_argument("--quiet",        action="store_true")
    p_fr.set_defaults(func=cmd_frames)

    # score
    p_sc = subparsers.add_parser("score",
        help="Score a summary directory against original frames.")
    p_sc.add_argument("--frames-dir",   required=True, metavar="DIR")
    p_sc.add_argument("--summary-dir",  required=True, metavar="DIR")
    p_sc.add_argument("--n-comparisons",type=int, default=10)
    p_sc.set_defaults(func=cmd_score)

    # vlm
    p_vl = subparsers.add_parser("vlm",
        help="Text-conditioned summarization via OWL-ViT (requires sigvideo[vlm]).")
    p_vl.add_argument("--frames-dir",  required=True, metavar="DIR")
    p_vl.add_argument("--subtitles",   metavar="FILE")
    p_vl.add_argument("--queries",     metavar="Q1,Q2,...")
    p_vl.add_argument("--top-n",       type=int,   default=20)
    p_vl.add_argument("--threshold",   type=float, default=0.025)
    p_vl.add_argument("--device",      metavar="DEVICE", default=None)
    p_vl.add_argument("--output",      metavar="FILE")
    p_vl.add_argument("--copy-frames", metavar="DIR")
    p_vl.add_argument("--quiet",       action="store_true")
    p_vl.set_defaults(func=cmd_vlm)

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        sys.exit(0)
    args.func(args)


if __name__ == "__main__":
    main()
