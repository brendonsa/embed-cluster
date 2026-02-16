import argparse
import glob
import os
import torch
import numpy as np
import pandas as pd


def load_data(filename, layer_number=33, mode="mean"):
    data = torch.load(filename)

    records = []

    if mode == "windowed" and "windows" in data and data["windows"]:
        for idx, window in enumerate(data["windows"]):
            emb_dict = window.get("mean_representations", {})
            emb = emb_dict.get(layer_number)
            if emb is not None:
                row = {
                    "label": data["label"],
                    "start": window["start"],
                    "end": window["end"],
                    **{f"emb_{i}": v for i, v in enumerate(emb.numpy())}
                }
                records.append(row)
    else:
        full_data = data.get("full", {})

        if mode == "bos":
            emb = full_data.get("bos_representations", {}).get(layer_number)
            if emb is None:
                emb = data.get("bos_representations", {}).get(layer_number)
        elif mode == "mean":
            emb = full_data.get("mean_representations", {}).get(layer_number)
            if emb is None:
                emb = data.get("mean_representations", {}).get(layer_number)
        else:
            emb = None

        if emb is not None:
            row = {"label": data["label"], **
                   {f"emb_{i}": v for i, v in enumerate(emb.numpy())}}
            records.append(row)

    return records


def collect_all_data(root_name, layer_number=33, mode="mean"):
    pt_files = glob.glob(os.path.join(root_name, "**", "*.pt"), recursive=True)
    data = []
    for file in pt_files:
        try:
            records = load_data(file, layer_number, mode)
            data.extend(records)
        except Exception as e:
            print(f"Failed to process {file}: {e}")
    return data


def save_to_csv(data, output_file):
    df = pd.DataFrame(data)
    if "label" in df.columns:
        df = df.sort_values("label").reset_index(drop=True)
    df.to_csv(output_file, index=False)


def main():
    parser = argparse.ArgumentParser(
        description="Process and join .pt files into a CSV.")
    parser.add_argument("input_dir", type=str,
                        help="Input directory containing .pt files.")
    parser.add_argument("output_csv", type=str, help="Output CSV file.")
    parser.add_argument("--layer", type=int, default=33,
                        help="Layer number to extract embeddings from (default: 33).")
    parser.add_argument("--mode", choices=["bos", "mean", "windowed"], default="mean",
                        help="Which embedding mode to extract: bos, mean, or windowed.")
    args = parser.parse_args()

    print(f"Loading data from {args.input_dir} using mode: {args.mode}...")
    data = collect_all_data(
        args.input_dir, layer_number=args.layer, mode=args.mode)

    print(f"Saving data to {args.output_csv}...")
    save_to_csv(data, args.output_csv)
    print("Done.")


if __name__ == "__main__":
    main()
