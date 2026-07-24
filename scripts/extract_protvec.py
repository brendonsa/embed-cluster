#!/usr/bin/env python3
"""Whole-sequence ProtVec embeddings from pretrained 3-gram vectors (Asgari & Mofrad 2015)."""

import argparse

import numpy as np
import pandas as pd
from Bio import SeqIO


def load_ngram_vectors(path):
    table = pd.read_csv(path, sep="\t", index_col=0)
    vectors = {ngram: row.to_numpy(dtype=np.float64) for ngram, row in table.iterrows()}
    return vectors, vectors["<unk>"]


def embed_sequence(seq, vectors, unk, n=3):
    seq = seq.upper()
    total = np.zeros_like(unk)
    for offset in range(n):
        for i in range(offset, len(seq) - n + 1, n):
            total += vectors.get(seq[i:i + n], unk)
    return total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fasta", required=True)
    parser.add_argument("--vectors", required=True,
                         help="protVec_100d_3grams.csv (tab-separated, pretrained)")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    vectors, unk = load_ngram_vectors(args.vectors)

    rows = []
    for record in SeqIO.parse(args.fasta, "fasta"):
        seq = str(record.seq).replace("-", "").replace(".", "")
        emb = embed_sequence(seq, vectors, unk)
        rows.append({"label": record.id, **{f"emb_{i}": v for i, v in enumerate(emb)}})

    df = pd.DataFrame(rows).sort_values("label").reset_index(drop=True)
    df.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
