from Bio import SeqIO
import argparse

def filter_fasta(input_fasta, output_fasta):
    with open(input_fasta, "r") as input_handle, open(output_fasta, "w") as output_handle:
        for record in SeqIO.parse(input_handle, "fasta"):
            if not record.id.startswith("NODE_"):
                SeqIO.write(record, output_handle, "fasta")

def main():
    parser = argparse.ArgumentParser(description="Filter out sequences with IDs starting with NODE_ from a FASTA file.")
    parser.add_argument("input_fasta", type=str, help="Input FASTA file")
    parser.add_argument("output_fasta", type=str, help="Output FASTA file with filtered sequences")

    args = parser.parse_args()
    filter_fasta(args.input_fasta, args.output_fasta)

if __name__ == "__main__":
    main()