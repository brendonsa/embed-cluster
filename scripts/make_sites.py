#!/usr/bin/env python3
"""Combine antigenic site definitions from JSON files into a single sites.txt.

Each JSON file has the format:
    {"map": {"HA1": {"<site>": 1, ...}}, ...}

where sites are 1-based HA1 positions. The output is 0-based indices
offset by the signal peptide length (default 16).
"""
import argparse
import json
import pathlib


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "json_files", type=pathlib.Path, nargs="+",
        help="Antigenic site JSON files to combine",
    )
    parser.add_argument(
        "-o", "--output", type=pathlib.Path, required=True,
        help="Output sites.txt file",
    )
    parser.add_argument(
        "--sigpep_length", type=int, default=16,
        help="Signal peptide length to offset HA1 positions (default: 16)",
    )
    args = parser.parse_args()

    all_sites = set()
    for path in args.json_files:
        with open(path) as f:
            data = json.load(f)
        ha1_sites = data.get("map", {}).get("HA1", {})
        for site_str in ha1_sites:
            all_sites.add(int(site_str))

    # Convert 1-based HA1 positions to 0-based indices with SigPep offset
    indices = sorted((site - 1) + args.sigpep_length for site in all_sites)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        f.write("\n".join(str(i) for i in indices) + "\n")

    print(f"Combined {len(args.json_files)} files -> {len(indices)} unique sites")
    print(f"Written to {args.output}")


if __name__ == "__main__":
    main()
