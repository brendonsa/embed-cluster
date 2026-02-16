#!/usr/bin/env python3

import argparse
import pathlib
import re

import torch
from Bio import SeqIO
from transformers import T5Tokenizer, T5EncoderModel


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fasta",
        required=True,
        type=pathlib.Path,
        help="Input FASTA file"
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=pathlib.Path,
        help="Output directory for embeddings"
    )
    parser.add_argument(
        "--model-name",
        default="Rostlab/prot_t5_xl_uniref50",
        help="HuggingFace model name"
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device (e.g. cuda, cuda:0, cpu). Auto-detect if not set."
    )
    parser.add_argument(
        "--cpu_threads",
        type=int,
        default=0,
        help="Number of CPU threads for PyTorch (0 = all available cores)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == "cpu":
        import os
        cpu_threads = args.cpu_threads if args.cpu_threads > 0 else os.cpu_count()
        torch.set_num_threads(cpu_threads)
        torch.set_num_interop_threads(cpu_threads)
        print(f"Using {cpu_threads} CPU threads for inference")

    tokenizer = T5Tokenizer.from_pretrained(
        args.model_name, do_lower_case=False
    )
    model = T5EncoderModel.from_pretrained(args.model_name)
    model.to(device)
    model.eval()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for record in SeqIO.parse(args.fasta, "fasta"):
        label = record.id
        output_file = args.output_dir / f"{label}.pt"

        seq = str(record.seq).replace("*", "X")
        seq = re.sub(r"[UZOB]", "X", seq)
        seq = [" ".join(seq)]

        tokens = tokenizer.batch_encode_plus(
            seq,
            add_special_tokens=True,
            return_tensors="pt",
        )
        input_ids = tokens["input_ids"].to(device)
        attention_mask = tokens["attention_mask"].to(device)

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            embedding = (
                outputs.last_hidden_state[0, : len(record.seq)]
                .mean(dim=0)
                .clone()
                .cpu()
            )

        result = {
            "label": label,
            "full": {
                "mean_representations": {
                    0: embedding
                }
            }
        }
        output_file.parent.mkdir(parents=True, exist_ok=True)
        torch.save(result, output_file)


if __name__ == "__main__":
    main()
