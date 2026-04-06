"""
vlm.py — Text-conditioned video summarization via OWL-ViT (NB03).

Requires the optional [vlm] extra:
    pip install sigvideo[vlm]

Based on §3 of the paper: zero-shot text-conditioned object detection
using the OWL-ViT model (Minderer et al., 2022) to select frames that
contain the most frequent nouns extracted from video subtitles/transcriptions.

Reference: https://doi.org/10.3390/electronics12071735
"""

from __future__ import annotations

import os
from typing import List, Optional, Tuple


def _check_vlm_deps() -> None:
    """Raise a clear error if optional VLM dependencies are missing."""
    missing = []
    for pkg in ("torch", "transformers", "PIL", "nltk"):
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        raise ImportError(
            f"Missing VLM dependencies: {missing}. "
            "Install them with:  pip install sigvideo[vlm]"
        )


def extract_noun_queries(text: str, top_n: int = 20) -> List[str]:
    """
    Extract the top-n most frequent singular nouns (NN) from a text
    using NLTK POS tagging, after removing English stop-words.
    This function only requires nltk (not torch or transformers).

    Args:
        text:   raw subtitle / transcript text.
        top_n:  maximum number of query words to return.

    Returns:
        list of noun strings to use as OWL-ViT text queries.
    """
    try:
        import nltk
        from nltk.corpus import stopwords
        from collections import Counter
    except ImportError:
        raise ImportError(
            "nltk is required: pip install nltk  (or pip install sigvideo[vlm])"
        )

    for resource in ("punkt", "stopwords", "averaged_perceptron_tagger",
                     "punkt_tab", "averaged_perceptron_tagger_eng"):
        try:
            nltk.download(resource, quiet=True)
        except Exception:
            pass

    stop = set(stopwords.words("english"))
    sentences = nltk.sent_tokenize(text)
    nouns = []
    for sent in sentences:
        words = [w for w in nltk.word_tokenize(sent) if w not in stop]
        for word, tag in nltk.pos_tag(words):
            if tag == "NN":
                nouns.append(word.lower())

    counter = Counter(nouns)
    return [w for w, _ in counter.most_common(top_n)]


def load_subtitles(path: str) -> str:
    """Load a plain-text subtitle / transcript file."""
    with open(path, encoding="utf-8") as f:
        return " ".join(f.read().split())


def _parse_owlvit_outputs(outputs, images, device):
    """
    Parse raw OWL-ViT model outputs into per-image detection dicts.

    Avoids processor.post_process / post_process_object_detection API
    differences across transformers versions by working directly with
    the two raw tensors that have always been part of the output:

      outputs.logits    : (B, n_patches, n_queries)  raw logits per patch
      outputs.pred_boxes: (B, n_patches, 4)          cx,cy,w,h in [0,1]

    Returns a list of dicts with keys: scores, labels, boxes (all on CPU).
    """
    import torch
    import torch.nn.functional as F

    results = []
    for i, img in enumerate(images):
        W, H = img.size
        scores_all       = F.sigmoid(outputs.logits[i])    # (n_patches, n_queries)
        max_scores, lbls = scores_all.max(dim=-1)          # best query per patch

        cx, cy, w, h = outputs.pred_boxes[i].unbind(-1)
        boxes = torch.stack([
            (cx - w / 2) * W, (cy - h / 2) * H,
            (cx + w / 2) * W, (cy + h / 2) * H,
        ], dim=-1)

        results.append({
            "scores": max_scores.cpu(),
            "labels": lbls.cpu(),
            "boxes":  boxes.cpu(),
        })
    return results


def summarize_vlm(
    frames_dir: str,
    queries: List[str],
    score_threshold: float = 0.025,
    mini_batch: int = 20,
    device: Optional[str] = None,
    verbose: bool = True,
) -> List[str]:
    """
    Select frames whose content matches any of `queries` using OWL-ViT
    zero-shot text-conditioned object detection.

    Args:
        frames_dir:       directory of extracted video frames (PNG/JPG).
        queries:          list of text queries (e.g. top-20 nouns from subtitles).
        score_threshold:  minimum detection confidence to include a frame (paper: 0.025).
        mini_batch:       number of images per OWL-ViT forward pass (paper: 20).
        device:           torch device string, e.g. "cuda" or "cpu". Auto-detected if None.
        verbose:          print per-frame progress.

    Returns:
        sorted list of selected frame filenames (basenames).
    """
    _check_vlm_deps()

    import torch
    from PIL import Image
    from transformers import OwlViTProcessor, OwlViTForObjectDetection

    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}
    fnames = sorted(f for f in os.listdir(frames_dir)
                    if os.path.splitext(f)[1].lower() in exts)
    if not fnames:
        raise FileNotFoundError(f"No image files found in: {frames_dir}")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if verbose:
        print(f"[sigvideo.vlm] Loading OWL-ViT on {device}…")

    processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
    model = OwlViTForObjectDetection.from_pretrained(
        "google/owlvit-base-patch32").to(device)
    model.eval()

    if verbose:
        print(f"[sigvideo.vlm] {len(fnames)} frames | "
              f"{len(queries)} queries | threshold={score_threshold}")

    images   = [Image.open(os.path.join(frames_dir, f)).convert("RGB") for f in fnames]
    selected = []

    for start in range(0, len(images), mini_batch):
        batch_imgs  = images[start: start + mini_batch]
        batch_names = fnames[start: start + mini_batch]
        n = len(batch_imgs)

        # Pass [queries] * n  — one query-list per image in the batch.
        # A flat list of Q queries with B images gives max_text_queries = Q//B,
        # which is wrong when Q != B.  The correct form duplicates the list:
        # [[q1,...,qQ]] * B  →  max_text_queries = Q  ✓
        inputs = processor(
            text=[queries] * n,
            images=batch_imgs,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        results = _parse_owlvit_outputs(outputs, batch_imgs, device)

        for idx, (res, fname) in enumerate(zip(results, batch_names)):
            detections = [
                (queries[int(lbl)], round(float(sc), 3))
                for sc, lbl in zip(res["scores"], res["labels"])
                if float(sc) >= score_threshold
            ]
            if detections:
                selected.append(fname)
            if verbose:
                status = f"  {len(detections)} detection(s)" if detections else ""
                print(f"  frame {start + idx:5d}/{len(images)}{status}")

    summary = sorted(set(selected))
    if verbose:
        print(f"\n[sigvideo.vlm] Selected {len(summary)} frames.")
    return summary


def summarize_vlm_from_subtitles(
    frames_dir: str,
    subtitles_path: str,
    top_n_queries: int = 20,
    score_threshold: float = 0.025,
    mini_batch: int = 20,
    device: Optional[str] = None,
    verbose: bool = True,
) -> Tuple[List[str], List[str]]:
    """
    End-to-end VLM summarization: load subtitles → extract nouns → run OWL-ViT.

    Returns:
        (selected_frames, text_queries)
    """
    text    = load_subtitles(subtitles_path)
    queries = extract_noun_queries(text, top_n=top_n_queries)
    if verbose:
        print(f"[sigvideo.vlm] Queries: {queries}")
    frames = summarize_vlm(
        frames_dir, queries, score_threshold, mini_batch, device, verbose
    )
    return frames, queries
