#!/usr/bin/env python3
"""Sorted-coverage distance matrix from per-token embeddings."""
import argparse
import glob
import os

import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import pdist, squareform


def load_from_pt(input_dir: str, layer: int):
    pt_files = sorted(glob.glob(os.path.join(input_dir, "**", "*.pt"), recursive=True))
    if not pt_files:
        raise FileNotFoundError(f"No .pt files found under {input_dir}")

    labels, mats = [], []
    for f in pt_files:
        data = torch.load(f, map_location="cpu")
        reps = data.get("full", {}).get("representations", {}).get(layer)
        if reps is None:
            raise KeyError(f"{f}: missing per-token representations for layer {layer}.")
        labels.append(data["label"])
        mats.append(reps.numpy())

    order = np.argsort(labels)
    labels = [labels[i] for i in order]
    mats = [mats[i] for i in order]
    return labels, mats


def load_from_npz(path: str):
    data = np.load(path, allow_pickle=True)
    labels = [str(x) for x in data["labels"]]
    emb = data["embeddings"]  # (N, L, D)
    order = np.argsort(labels)
    labels = [labels[i] for i in order]
    mats = [emb[i] for i in order]
    return labels, mats


def load_embeddings(input_path: str, layer: int):
    if input_path.endswith(".npz"):
        labels, mats = load_from_npz(input_path)
    else:
        labels, mats = load_from_pt(input_path, layer)

    lengths = {m.shape[0] for m in mats}
    if len(lengths) != 1:
        raise ValueError(f"Per-token sequences are not equal length (found {sorted(lengths)}); aligned input required.")

    emb = np.stack(mats, axis=0)  # (N, L, D)
    return labels, emb


def select_sites(site_weights, coverage, direction, include_zero_weights):
    L = site_weights.shape[0]
    pos_mask = site_weights > 0.0
    total_pos_weight = float(site_weights[pos_mask].sum())
    if total_pos_weight <= 0:
        raise ValueError("No positive site weights found.")

    target_weight = float(coverage) * total_pos_weight

    if include_zero_weights:
        candidate = np.arange(L, dtype=int)
    else:
        candidate = np.where(pos_mask)[0].astype(int)

    if direction == "high":
        order = np.argsort(site_weights[candidate])[::-1]
    elif direction == "low":
        order = np.argsort(site_weights[candidate])
    else:
        raise ValueError("direction must be 'high' or 'low'")

    selected = []
    cum_w = 0.0
    for j in order:
        idx = int(candidate[j])
        w = float(site_weights[idx])
        if (w <= 0.0) and (not include_zero_weights):
            continue
        selected.append(idx)
        cum_w += max(w, 0.0)
        if cum_w >= target_weight:
            break

    if not selected:
        raise ValueError("No sites selected (check coverage/include_zero_weights/weights).")
    return selected


def aggregate(embeddings_nld, selected, site_weights, metric, normalize):
    agg = None
    if normalize == "weights":
        den = 0.0
        for idx in selected:
            w = max(float(site_weights[idx]), 0.0)
            dist = squareform(pdist(embeddings_nld[:, idx, :], metric=metric))
            agg = (w * dist) if agg is None else (agg + w * dist)
            den += w
        if den == 0.0:
            raise ValueError("Normalization denominator is zero (all selected weights <= 0).")
        agg = agg / den
    elif normalize == "count":
        for idx in selected:
            dist = squareform(pdist(embeddings_nld[:, idx, :], metric=metric))
            agg = dist if agg is None else (agg + dist)
        agg = agg / float(len(selected))
    else:
        raise ValueError("normalize must be 'weights' or 'count'")
    return agg


def main():
    ap = argparse.ArgumentParser(description="Sorted-coverage distance matrix from per-token embeddings.")
    ap.add_argument("--input", required=True, help="Per-token .pt directory, or an .npz with keys labels/embeddings.")
    ap.add_argument("--weights", required=True, help="Site weights .npy, length L.")
    ap.add_argument("--output", required=True, help="Output distance-matrix CSV (pathogen-distance format).")
    ap.add_argument("--layer", type=int, default=33, help="Layer number to read (default: 33).")
    ap.add_argument("--coverage", type=float, required=True, help="0..1")
    ap.add_argument("--metric", default="cosine")
    ap.add_argument("--direction", choices=["high", "low"], default="high")
    ap.add_argument("--normalize", choices=["weights", "count"], default="weights")
    ap.add_argument("--include-zero", action="store_true")
    args = ap.parse_args()

    labels, emb = load_embeddings(args.input, args.layer)
    N, L, D = emb.shape

    site_weights = np.load(args.weights)
    if site_weights.shape != (L,):
        raise ValueError(f"weights length {site_weights.shape} != L {L}")

    selected = select_sites(site_weights, args.coverage, args.direction, args.include_zero)
    dist = aggregate(emb, selected, site_weights, args.metric, args.normalize)

    df = pd.DataFrame(dist, index=labels, columns=labels)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    df.to_csv(args.output)
    print(f"Wrote {N}x{N} sorted-coverage distance matrix ({len(selected)}/{L} sites) -> {args.output}")


if __name__ == "__main__":
    main()
