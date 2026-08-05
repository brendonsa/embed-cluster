#!/usr/bin/env python3
import argparse
import math
from collections import Counter

import numpy as np
from Bio import SeqIO


def load_aligned_seqs(fasta_path):
    seqs = [str(r.seq) for r in SeqIO.parse(fasta_path, "fasta")]
    if not seqs:
        raise ValueError(f"No sequences found in {fasta_path}")
    L = len(seqs[0])
    if any(len(s) != L for s in seqs):
        raise ValueError("FASTA is not aligned: sequences have different lengths.")
    return seqs


def weights_unit(seqs):
    L = len(seqs[0])
    return np.ones(L, dtype=float)


def weights_unique(seqs):
    L = len(seqs[0])
    w = np.empty(L, dtype=float)
    for i, col in enumerate(zip(*seqs)):
        w[i] = max(0, len(set(col)) - 1)
    return w


def weights_shannon(seqs, base=2.0, ignore_gaps=False, gap_chars="-."):
    L = len(seqs[0])

    if base == 2.0:
        log = math.log2
    elif base == math.e:
        log = math.log
    else:
        log = lambda x: math.log(x, base)

    w = np.zeros(L, dtype=float)
    gaps = set(gap_chars)

    for i, col in enumerate(zip(*seqs)):
        if ignore_gaps:
            col = [c for c in col if c not in gaps]
        if not col:
            w[i] = 0.0
            continue

        counts = Counter(col)
        n = sum(counts.values())
        H = 0.0
        for cnt in counts.values():
            p = cnt / n
            H -= p * log(p)
        w[i] = H

    return w


def main():
    ap = argparse.ArgumentParser(description="Compute per-column weights from an aligned FASTA.")
    ap.add_argument("--input", "-i", required=True, help="Aligned FASTA (all seqs same length)")
    ap.add_argument("--output", "-o", required=True, help="Output .npy path")
    ap.add_argument("--kind", required=True, choices=["unit", "shannon", "unique"])
    ap.add_argument("--base", type=float, default=2.0, help="Log base for shannon (default: 2)")
    ap.add_argument("--ignore-gaps", action="store_true", help="Ignore '-' and '.' for shannon")
    ap.add_argument("--gap-chars", default="-.", help="Gap characters when --ignore-gaps is set")
    args = ap.parse_args()

    seqs = load_aligned_seqs(args.input)

    if args.kind == "unit":
        w = weights_unit(seqs)
    elif args.kind == "unique":
        w = weights_unique(seqs)
    else:
        w = weights_shannon(seqs, base=args.base, ignore_gaps=args.ignore_gaps, gap_chars=args.gap_chars)

    np.save(args.output, w)


if __name__ == "__main__":
    main()
