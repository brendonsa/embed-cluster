#!/usr/bin/env python3
"""Mean-pool an already-extracted per-token MSA Pairformer npz into a sequence-level CSV."""
import argparse
import csv

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", "-i", required=True, help="features_pertok_msapairformer/embeddings.npz")
    ap.add_argument("--output", "-o", required=True)
    args = ap.parse_args()

    data = np.load(args.input, allow_pickle=True)
    labels = [str(x) for x in data["labels"]]
    emb = data["embeddings"]  # (N, L, D)

    pooled = emb.mean(axis=1)  # (N, D)
    norm = np.linalg.norm(pooled, axis=1, keepdims=True)
    pooled = pooled / np.clip(norm, 1e-12, None)

    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerow(["strain"] + [str(i) for i in range(pooled.shape[1])])
        for label, row in zip(labels, pooled):
            writer.writerow([label] + list(row))

    print(f"Pooled {len(labels)} sequences -> {args.output} (dim={pooled.shape[1]})")


if __name__ == "__main__":
    main()
