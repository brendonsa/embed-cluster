#!/usr/bin/env python3
import argparse
import numpy as np
from collections import Counter
def translate_nt(nt_seq):
    from Bio.Seq import Seq
    return str(Seq(nt_seq.replace("-", "")).translate()).rstrip("*")

def read_fasta(path):
    seqs = {}
    current = None
    with open(path) as fh:
        for line in fh:
            line = line.rstrip()
            if line.startswith(">"):
                current = line[1:].split()[0]
                seqs[current] = []
            elif current is not None:
                seqs[current].append(line)
    return {k: "".join(v) for k, v in seqs.items()}


def read_table(path, clade_col="clade_membership"):
    import csv
    clades = {}
    phylo  = {}
    with open(path) as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            strain = row.get("strain") or row.get("name")
            clade  = row.get(clade_col, "").strip()
            if not strain:
                continue
            if clade:
                clades[strain] = clade
            bl  = row.get("branch_length", "")
            div = row.get("divergence", "")
            entry = {}
            if bl:
                try:
                    entry["branch_length"] = float(bl)
                except ValueError:
                    pass
            if div:
                try:
                    entry["divergence"] = float(div)
                except ValueError:
                    pass
            if entry:
                phylo[strain] = entry
    return clades, phylo

def to_matrix(seqs):
    return np.frombuffer(
        b"".join(s.encode() for s in seqs), dtype=np.uint8
    ).reshape(len(seqs), -1)


def pairwise_hamming(mat):
    n = mat.shape[0]
    dists = []
    L = mat.shape[1]
    for i in range(n):
        diff = (mat[i+1:] != mat[i]).sum(axis=1)
        dists.append(diff)
    flat = np.concatenate(dists).astype(float) / L
    return flat


def simpson_diversity(counts):
    n_vec = np.array(list(counts.values()), dtype=float)
    N = n_vec.sum()
    if N < 2:
        return float("nan")
    return 1.0 - (n_vec * (n_vec - 1)).sum() / (N * (N - 1))

def variable_sites(mat):
    return int((mat.min(axis=0) != mat.max(axis=0)).sum())


def mutations_per_seq(mat, reference):
    return (mat != reference[None, :]).sum(axis=1).astype(float)


def report(label, seqs, clades, phylo, reference=None, aa_seqs=None, aa_reference=None):
    strains = [s for s in seqs if s in clades]
    missing = [s for s in seqs if s not in clades]

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  Sequences in alignment : {len(seqs)}")
    print(f"  With clade assignment  : {len(strains)}")
    if missing:
        print(f"  Without clade (skipped): {len(missing)}")

    clade_counts = Counter(clades[s] for s in strains)
    n_clades = len(clade_counts)
    N = len(strains)

    print(f"\n  Clades ({n_clades} total, N={N})")
    print(f"  {'-'*40}")
    for clade, count in sorted(clade_counts.items(), key=lambda x: -x[1]):
        bar = "#" * int(30 * count / N)
        print(f"  {clade:<25} {count:>5}  {bar}")

    simpson = simpson_diversity(clade_counts)
    print(f"\n  Simpson diversity index: {simpson:.4f}")

    # Hamming distances
    seq_list   = [seqs[s] for s in strains]
    clade_list = [clades[s] for s in strains]

    L = len(seq_list[0])
    print(f"\n  Sequence length (aligned): {L} nt")

    mat = to_matrix(seq_list)
    all_d = pairwise_hamming(mat)
    print("\n  Pairwise Hamming distances (normalised)")
    print(f"  {'Overall':<20}  mean={all_d.mean():.4f}  "
          f"std={all_d.std():.4f}  "
          f"min={all_d.min():.4f}  max={all_d.max():.4f}")

    # Within-clade
    within = []
    for clade in clade_counts:
        idx = [i for i, c in enumerate(clade_list) if c == clade]
        if len(idx) < 2:
            continue
        sub = mat[idx]
        d = pairwise_hamming(sub)
        within.append(d)
    within_all = np.concatenate(within)
    print(f"  {'Within-clade':<20}  mean={within_all.mean():.4f}  "
          f"std={within_all.std():.4f}  "
          f"min={within_all.min():.4f}  max={within_all.max():.4f}")

    # Between-clade
    between = []
    clades_list = sorted(clade_counts.keys())
    for i, ca in enumerate(clades_list):
        idx_a = [j for j, c in enumerate(clade_list) if c == ca]
        for cb in clades_list[i+1:]:
            idx_b = [j for j, c in enumerate(clade_list) if c == cb]
            diff = (mat[idx_a][:, None, :] != mat[idx_b][None, :, :]).sum(axis=2)
            between.append((diff.astype(float) / L).ravel())
    between_all = np.concatenate(between)
    print(f"  {'Between-clade':<20}  mean={between_all.mean():.4f}  "
          f"std={between_all.std():.4f}  "
          f"min={between_all.min():.4f}  max={between_all.max():.4f}")

    # Mutations
    vs = variable_sites(mat)
    print("\n  Mutations (vs reference)")
    print(f"  NT variable sites      : {vs} / {L}  ({100*vs/L:.1f}%)")
    if reference is not None:
        nt_muts = mutations_per_seq(mat, reference)
        print(f"  NT per-sequence        : mean={nt_muts.mean():.1f}  "
              f"std={nt_muts.std():.1f}  "
              f"min={nt_muts.min():.0f}  max={nt_muts.max():.0f}")

        if aa_seqs is not None and aa_reference is not None:
            aa_strains = [s for s in strains if s in aa_seqs]
            n_missing_aa = len(strains) - len(aa_strains)
            if n_missing_aa:
                print(f"  ({n_missing_aa} strains missing from AA alignment, skipped)")
            aa_list = [aa_seqs[s] for s in aa_strains]
            aa_mat  = to_matrix(aa_list)
            aa_muts = mutations_per_seq(aa_mat, aa_reference)
            nt_muts_matched = mutations_per_seq(mat[[strains.index(s) for s in aa_strains]], reference)
            syn = nt_muts_matched - aa_muts
            print(f"  AA per-sequence        : mean={aa_muts.mean():.1f}  "
                  f"std={aa_muts.std():.1f}  "
                  f"min={aa_muts.min():.0f}  max={aa_muts.max():.0f}")
            print(f"  Synonymous (NT - AA)   : mean={syn.mean():.1f}  "
                  f"std={syn.std():.1f}  "
                  f"min={syn.min():.0f}  max={syn.max():.0f}")
            total_nt = nt_muts_matched.sum()
            total_syn = syn.sum()
            if total_nt > 0:
                print(f"  Synonymous fraction    : {100*total_syn/total_nt:.1f}%")
    else:
        print("  Per-sequence           : (skipped — no --reference provided)")

    # Phylogenetic distances
    print("\n  Phylogenetic distances")
    n_phylo = sum(1 for s in strains if s in phylo)
    print(f"  Strains with phylo data: {n_phylo} / {len(strains)}")
    if n_phylo == 0:
        print("  (no match — check that strain names in the FASTA and table agree)")
    else:
        bl_vals  = [phylo[s]["branch_length"] for s in strains if s in phylo and "branch_length" in phylo[s]]
        div_vals = [phylo[s]["divergence"]    for s in strains if s in phylo and "divergence"    in phylo[s]]
        if bl_vals:
            bl = np.array(bl_vals)
            print(f"  Branch length          : mean={bl.mean():.6f}  "
                  f"std={bl.std():.6f}  "
                  f"min={bl.min():.6f}  max={bl.max():.6f}")
        if div_vals:
            div = np.array(div_vals)
            print(f"  Divergence from root   : mean={div.mean():.6f}  "
                  f"std={div.std():.6f}  "
                  f"min={div.min():.6f}  max={div.max():.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--alignment",  required=True,
                        help="Aligned FASTA (results/aligned_sequences.fasta)")
    parser.add_argument("--table",      required=True,
                        help="Table TSV with clade_membership column "
                             "(results/table.tsv)")
    parser.add_argument("--label",      default="Split",
                        help="Label printed in the report header")
    parser.add_argument("--clade-col",  default="clade_membership",
                        help="Column name for clade labels in the table")
    parser.add_argument("--reference", default=None,
                        help="Reference NT FASTA (config/reference_h3n2_ha.fasta).")
    parser.add_argument("--aa-alignment", default=None,
                        help="AA alignment FASTA "
                             "(results/translations/alignment_filtered_All.fasta).")
    args = parser.parse_args()

    seqs          = read_fasta(args.alignment)
    clades, phylo = read_table(args.table, clade_col=args.clade_col)

    reference = aa_reference = aa_seqs = None

    if args.reference:
        ref_seqs = read_fasta(args.reference)
        if not ref_seqs:
            print(f"WARNING: no sequences found in {args.reference}", flush=True)
        else:
            nt_ref_seq = next(iter(ref_seqs.values()))
            reference  = to_matrix([nt_ref_seq])[0]
            if args.aa_alignment:
                aa_translated = translate_nt(nt_ref_seq)
                aa_reference  = to_matrix([aa_translated])[0]

    if args.aa_alignment:
        aa_seqs = read_fasta(args.aa_alignment)

    report(args.label, seqs, clades, phylo,
           reference=reference, aa_seqs=aa_seqs, aa_reference=aa_reference)
