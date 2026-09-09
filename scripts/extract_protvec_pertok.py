#!/usr/bin/env python3
"""Per-block ProtVec embeddings from an aligned FASTA, for windowed distance accumulation.

Each sequence is cut into non-overlapping 3-column blocks directly on the alignment
(so block boundaries line up across sequences), and each block's 3 characters are
looked up as a single ProtVec 3-gram. Blocks containing a gap fall back to <unk>,
same as unseen 3-grams do in the whole-sequence embedding.
"""
import argparse

import numpy as np
import pandas as pd
from Bio import SeqIO


def load_ngram_vectors(path):
    table = pd.read_csv(path, sep="\t", index_col=0)
    vectors = {ngram: row.to_numpy(dtype=np.float64) for ngram, row in table.iterrows()}
    return vectors, vectors["<unk>"]


def embed_blocks(seq, vectors, unk, n=3):
    seq = seq.upper()
    n_blocks = len(seq) // n
    return np.stack([
        vectors.get(seq[i * n:(i + 1) * n], unk)
        for i in range(n_blocks)
    ])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", "-i", required=True, help="Aligned FASTA")
    parser.add_argument("--vectors", required=True, help="protVec_100d_3grams.csv")
    parser.add_argument("--output", "-o", required=True)
    args = parser.parse_args()

    vectors, unk = load_ngram_vectors(args.vectors)

    labels, mats = [], []
    for record in SeqIO.parse(args.input, "fasta"):
        labels.append(record.id)
        mats.append(embed_blocks(str(record.seq), vectors, unk))

    lengths = {m.shape[0] for m in mats}
    if len(lengths) != 1:
        raise ValueError(f"Sequences are not equal length after alignment (found block counts {sorted(lengths)}).")

    order = np.argsort(labels)
    labels = np.asarray(labels, dtype=object)[order]
    embeddings = np.stack(mats, axis=0)[order]  # (N, n_blocks, D)

    np.savez_compressed(args.output, labels=labels, embeddings=embeddings)
    print(f"Saved {len(labels)} sequences x {embeddings.shape[1]} blocks -> {args.output}")


if __name__ == "__main__":
    main()
