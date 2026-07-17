#!/usr/bin/env python3
"""Windowed (Gaussian-taper) distance matrix from per-token embeddings."""
import argparse
import glob
import os

import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import pdist, squareform


def compute_windows(L: int, window_size: int, overlap: int):
    if window_size <= 0:
        raise ValueError("window_size must be > 0")
    if overlap < 0 or overlap >= window_size:
        raise ValueError("overlap must satisfy 0 <= overlap < window_size")

    step = window_size - overlap
    windows = []
    start = 0
    while start + window_size <= L:
        windows.append((start, start + window_size))
        start += step
    if start < L:
        windows.append((start, L))  # tail window
    return windows


def gaussian_weights(width: int, sigma: float):
    if width <= 1:
        return np.ones(1, dtype=np.float64)
    k = np.arange(width, dtype=np.float64)
    center = (width - 1) / 2.0
    w = np.exp(-((k - center) ** 2) / (2.0 * sigma * sigma))
    s = w.sum()
    return w / s if s > 0 else np.full(width, 1.0 / width)


def load_embeddings(input_dir: str, layer: int):
    pt_files = sorted(glob.glob(os.path.join(input_dir, "**", "*.pt"), recursive=True))
    if not pt_files:
        raise FileNotFoundError(f"No .pt files found under {input_dir}")

    labels, mats = [], []
    for f in pt_files:
        data = torch.load(f, map_location="cpu")
        reps = data.get("full", {}).get("representations", {}).get(layer)
        if reps is None:
            raise KeyError(
                f"{f}: missing per-token representations for layer {layer}. "
                "Run extract.py with --include per_tok."
            )
        labels.append(data["label"])
        mats.append(reps.numpy())

    order = np.argsort(labels)
    labels = [labels[i] for i in order]
    mats = [mats[i] for i in order]

    lengths = {m.shape[0] for m in mats}
    if len(lengths) != 1:
        raise ValueError(f"Per-token sequences are not equal length (found {sorted(lengths)}); aligned input required.")

    emb = np.stack(mats, axis=0)  # (N, L, D)
    return labels, emb


def main():
    ap = argparse.ArgumentParser(description="Windowed (Gaussian-taper) distance matrix from per-token embeddings.")
    ap.add_argument("--input-dir", required=True, help="Directory of per-token .pt files (extract.py --include per_tok).")
    ap.add_argument("--output", required=True, help="Output distance-matrix CSV (pathogen-distance format).")
    ap.add_argument("--layer", type=int, default=33, help="Layer number to read (default: 33).")
    ap.add_argument("--window-size", type=int, default=1, help="Window size in residues (default: 1 = per-site).")
    ap.add_argument("--overlap", type=int, default=0, help="Overlap in residues (default: 0).")
    ap.add_argument("--sigma", type=float, default=-1.0, help="Gaussian sigma; if <= 0, defaults to window_size/4.")
    ap.add_argument("--metric", default="cosine", help="pdist metric (default: cosine).")
    ap.add_argument("--reduce", choices=["mean", "sum"], default="mean", help="Aggregate over windows (default: mean).")
    args = ap.parse_args()

    labels, emb = load_embeddings(args.input_dir, args.layer)
    N, L, D = emb.shape

    windows = compute_windows(L, args.window_size, args.overlap)
    sigma = args.sigma if args.sigma > 0 else args.window_size / 4.0

    acc = None
    for (s, e) in windows:
        w = gaussian_weights(e - s, sigma)
        W = np.tensordot(w, emb[:, s:e, :], axes=([0], [1]))  # (N, D)
        dist = squareform(pdist(W, metric=args.metric))       # (N, N)
        acc = dist if acc is None else acc + dist

    if args.reduce == "mean":
        acc = acc / float(len(windows))

    df = pd.DataFrame(acc, index=labels, columns=labels)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    df.to_csv(args.output)
    print(f"Wrote {N}x{N} windowed distance matrix ({len(windows)} windows, w={args.window_size}) -> {args.output}")


if __name__ == "__main__":
    main()
