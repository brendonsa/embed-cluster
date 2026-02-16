#!/usr/bin/env python3

import argparse
from Bio import SeqIO
from collections import defaultdict


def join_fasta_files(fasta_files, output_file):
    sequences = defaultdict(dict)  # Dictionary to store sequences by label

    # Iterate over each FASTA file to read sequences
    for fasta_file in fasta_files:
        print(f"Processing {fasta_file}")
        with open(fasta_file, "r") as handle:
            gene_name = fasta_file.split('/')[-1].split('.')[0]
            for record in SeqIO.parse(handle, "fasta"):
                label = record.id
                sequences[label][gene_name] = str(record.seq)

    # Write the combined sequences to the output FASTA file
    with open(output_file, "w") as output_handle:
        for label, gene_seqs in sequences.items():
            combined_seq = "".join(gene_seqs[gene] for gene in gene_seqs)
            output_handle.write(f">{label}\n{combined_seq}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Join multiple FASTA files based on labels.")
    parser.add_argument("fasta_files", nargs="+",
                        help="List of FASTA files to be joined")
    parser.add_argument("output_file", type=str, help="Output FASTA file")

    args = parser.parse_args()
    join_fasta_files(args.fasta_files, args.output_file)


if __name__ == "__main__":
    main()
