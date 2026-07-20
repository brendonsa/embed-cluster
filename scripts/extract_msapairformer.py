import argparse
import pathlib
import csv

from Bio import SeqIO
import torch
import pandas as pd
import numpy as np 

from MSA_Pairformer.model import MSAPairformer
from MSA_Pairformer.dataset import aa2tok_d, prepare_msa_masks


def tokenize_msa(records):
    """Tokenize an aligned MSA FASTA using aa2tok_d (keeps gaps)."""
    if not records:
        raise ValueError("No sequences found in input FASTA.")

    seqs = [str(r.seq) for r in records]
    lengths = {len(s) for s in seqs}
    if len(lengths) != 1:
        raise ValueError(f"All sequences in the MSA must have the same length, got lengths: {lengths}")

    L = lengths.pop()
    S = len(seqs)

    token_msa = torch.empty((S, L), dtype=torch.long)

    # Fallbacks for unknown and gap tokens
    def get_fallback(keys):
        for k in keys:
            if k in aa2tok_d:
                return aa2tok_d[k]
        # absolute fallback: first value in dict
        return next(iter(aa2tok_d.values()))

    unk_idx = get_fallback(["X", "UNK", "<unk>"])
    gap_idx = get_fallback(["-", ".", "*", "<gap>", "GAP"])

    for i, seq in enumerate(seqs):
        row_tokens = []
        for aa in seq:
            aa_u = aa.upper()
            if aa_u in aa2tok_d:
                row_tokens.append(aa2tok_d[aa_u])
            elif aa == "-":
                row_tokens.append(gap_idx)
            else:
                row_tokens.append(unk_idx)
        token_msa[i] = torch.tensor(row_tokens, dtype=torch.long)

    return token_msa


def embed_msa(records, model: MSAPairformer, device) -> torch.Tensor:
    """
    Compute per-sequence embeddings for an aligned MSA using MSA Pairformer.
    Returns a tensor of shape (num_seqs, D).
    """
    # Tokenize
    msa_tokenized_t = tokenize_msa(records)          # (S, L)
    S, L = msa_tokenized_t.shape

    # One-hot encode: (1, S, L, 28)
    num_tokens = len(aa2tok_d)
    msa_onehot_t = torch.nn.functional.one_hot(
        msa_tokenized_t, num_classes=num_tokens
    ).unsqueeze(0).float().to(device)

    # Masks
    mask, msa_mask, full_mask, pairwise_mask = prepare_msa_masks(
        msa_tokenized_t.unsqueeze(0)
    )
    mask = mask.to(device)
    msa_mask = msa_mask.to(device)
    full_mask = full_mask.to(device)
    pairwise_mask = pairwise_mask.to(device)

    # Forward pass
    with torch.no_grad():
        if device.type == "cuda":
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                res = model(
                    msa=msa_onehot_t.to(torch.bfloat16),
                    mask=mask,
                    msa_mask=msa_mask,
                    full_mask=full_mask,
                    pairwise_mask=pairwise_mask,
                    return_contacts=False,
                    query_only=False,
                    return_msa_repr_layer_idx=None,
                    return_pairwise_repr_layer_idx=None,
                )
        else:
            res = model(
                msa=msa_onehot_t.to(torch.float32),
                mask=mask,
                msa_mask=msa_mask,
                full_mask=full_mask,
                pairwise_mask=pairwise_mask,
                return_contacts=False,
                query_only=False,
                return_msa_repr_layer_idx=None,
                return_pairwise_repr_layer_idx=None,
            )

    # final_msa_repr: (1, S, L, D)
    final_msa_repr = res["final_msa_repr"].squeeze(0)  

    return final_msa_repr.cpu()


def process(input_fasta: str, output_path: str, gpu_id: int, pooling: str = "mean"):
    """Process aligned MSA FASTA → embeddings → CSV (pooled) or NPZ (concat)."""
    pooling = pooling.lower()

    # device management
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu_id}")
        print(f"Using GPU: {torch.cuda.get_device_name(gpu_id)}")
    else:
        device = torch.device("cpu")
        print("CUDA not available, using CPU")

    # load MSA Pairformer model
    print("Loading MSA Pairformer weights from Hugging Face cache...")
    model = MSAPairformer.from_pretrained(device=device)
    model.eval()

    # read MSA records (aligned FASTA)
    records = list(SeqIO.parse(input_fasta, "fasta"))
    if not records:
        raise ValueError(f"No sequences found in {input_fasta}")

    # compute embeddings for all sequences together
    emb_tensor = embed_msa(records, model, device)   # (S, L, D) on CPU, may be bf16

    # labels + deterministic order
    labels = np.asarray([r.id for r in records], dtype=object)
    order = np.argsort(labels)
    labels = labels[order]
    emb_tensor = emb_tensor[order, :, :]  # (S, L, D)

    output_file = pathlib.Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    if pooling == "concat":
        if output_file.suffix.lower() != ".npz":
            output_file = output_file.with_suffix(".npz")

        # Cast only for serialization (NumPy can't handle bfloat16)
        np.savez_compressed(
            output_file,
            labels=labels,
            embeddings=emb_tensor.to(torch.float32).numpy(),
        )
        print(f"Saved {len(labels)} raw embeddings → {output_file} (shape={emb_tensor.shape})")
        return

    # pooled -> CSV (N, D) : do pooling in float32 for numerical safety
    emb_f32 = emb_tensor.to(torch.float32)

    if pooling in ("mean", "avg", "average"):
        pooled = emb_f32.mean(dim=1)
    elif pooling == "max":
        pooled, _ = emb_f32.max(dim=1)
    elif pooling == "min":
        pooled, _ = emb_f32.min(dim=1)
    else:
        raise ValueError(f"Unknown pooling mode: {pooling}")

    # L2 normalize per sequence
    norms = pooled.norm(p=2, dim=1, keepdim=True)
    pooled = torch.where(norms > 0, pooled / norms, pooled)

    df = pd.DataFrame(pooled.numpy(), index=labels)
    df.to_csv(output_file, index_label="strain", quoting=csv.QUOTE_ALL)
    print(f"Saved {len(df)} sequence embeddings → {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute MSA Pairformer embeddings (pooled -> CSV; concat -> NPZ)."
    )
    parser.add_argument("--input", "-i", required=True, help="Input aligned MSA FASTA file")
    parser.add_argument("--output", "-o", required=True, help="Output CSV/NPZ path")
    parser.add_argument("--gpu-id", type=int, default=0, help="GPU device index (default: 0)")
    parser.add_argument(
        "--pooling",
        type=str,
        default="mean",
        choices=["mean", "min", "max", "concat", "avg", "average"],
        help="Pooling mode (default: mean). If concat, saves NPZ (N,L,D).",
    )
    args = parser.parse_args()

    process(args.input, args.output, args.gpu_id, pooling=args.pooling)
