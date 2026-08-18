import argparse
import csv
import pathlib

import numpy as np
import pandas as pd
import torch
from Bio import SeqIO
import esm


MAX_LEN = 1023  # model caps total tokens (incl. BOS) at 1024


def load_msa(input_fasta):
    records = list(SeqIO.parse(input_fasta, "fasta"))
    labels = [r.id for r in records]
    seqs = [str(r.seq) for r in records]
    if seqs and len(seqs[0]) > MAX_LEN:
        print(f"[extract_msatransformer_pertok] truncating alignment {len(seqs[0])} -> {MAX_LEN} columns")
        seqs = [s[:MAX_LEN] for s in seqs]
    return labels, seqs


def process(input_fasta, output_path, gpu_id, pooling="mean"):
    pooling = pooling.lower()

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu_id}")
        print(f"Using GPU: {torch.cuda.get_device_name(gpu_id)}")
    else:
        device = torch.device("cpu")
        print("CUDA not available, using CPU")

    model, alphabet = esm.pretrained.esm_msa1b_t12_100M_UR50S()
    model = model.to(device).eval()
    batch_converter = alphabet.get_batch_converter()

    labels, seqs = load_msa(input_fasta)
    data = [(labels[i], seqs[i]) for i in range(len(labels))]

    _, _, batch_tokens = batch_converter([data])  # (1, N, L+1)
    batch_tokens = batch_tokens.to(device)

    with torch.no_grad():
        out = model(batch_tokens, repr_layers=[12])
    reps = out["representations"][12][0]  # (N, L+1, D)
    reps = reps[:, 1:, :]  # drop leading BOS token -> (N, L, D)

    labels_arr = np.asarray(labels, dtype=object)
    order = np.argsort(labels_arr)
    labels_sorted = labels_arr[order]
    reps = reps[order]

    output_file = pathlib.Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    if pooling == "concat":
        if output_file.suffix.lower() != ".npz":
            output_file = output_file.with_suffix(".npz")
        np.savez_compressed(output_file, labels=labels_sorted, embeddings=reps.cpu().numpy())
        print(f"Saved {len(labels_sorted)} raw embeddings -> {output_file} (shape={tuple(reps.shape)})")
        return

    pooled = reps.mean(dim=1)
    norm = pooled.norm(p=2, dim=1, keepdim=True)
    pooled = torch.where(norm > 0, pooled / norm, pooled)
    df = pd.DataFrame(pooled.cpu().numpy(), index=labels_sorted)
    df.to_csv(output_file, index_label="strain", quoting=csv.QUOTE_ALL)
    print(f"Saved {len(df)} sequence embeddings -> {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute MSA Transformer embeddings (pooled -> CSV; concat -> NPZ)."
    )
    parser.add_argument("--input", "-i", required=True)
    parser.add_argument("--output", "-o", required=True)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--pooling", type=str, default="mean", choices=["mean", "concat"])
    args = parser.parse_args()

    process(args.input, args.output, args.gpu_id, pooling=args.pooling)
