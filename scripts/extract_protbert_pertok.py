#!/usr/bin/env python3
import argparse
import pathlib
import re

import torch
from Bio import SeqIO
from transformers import BertTokenizer, BertModel


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fasta", required=True, type=pathlib.Path)
    parser.add_argument("--output-dir", required=True, type=pathlib.Path)
    parser.add_argument("--model-name", default="Rostlab/prot_bert_bfd")
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = BertTokenizer.from_pretrained(args.model_name, do_lower_case=False)
    model = BertModel.from_pretrained(args.model_name).to(device)
    model.eval()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for record in SeqIO.parse(args.fasta, "fasta"):
        label = record.id
        # Keep one token per alignment column so positions stay aligned across strains.
        seq = str(record.seq).upper().replace("*", "X")
        seq = re.sub(r"[^ACDEFGHIKLMNPQRSTVWY]", "X", seq)
        L = len(seq)

        tokens = tokenizer(" ".join(seq), add_special_tokens=True, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**tokens)
            emb = out.last_hidden_state[0, 1:L + 1].clone().cpu()  # (L, D)

        result = {"label": label, "full": {"representations": {args.layer: emb}}}
        output_file = args.output_dir / f"{label}.pt"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        torch.save(result, output_file)


if __name__ == "__main__":
    main()
