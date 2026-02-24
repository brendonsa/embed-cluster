# Clustering

Embedding-based clustering of seasonal influenza H3N2 HA sequences using protein language models (ESM-2, ProtBERT, ProtT5, CARP) and traditional dimensionality reduction methods (PCA, t-SNE, UMAP, MDS), with HDBSCAN clustering evaluated against phylogenetic clades.

## Prerequisites

- [Conda](https://docs.conda.io/) (or [Mamba](https://mamba.readthedocs.io/))
- [Snakemake](https://snakemake.readthedocs.io/) (v8.14)
- GPU recommended for PLM feature extraction (CUDA 12.4)

## Setup

Create the conda environment from the provided yaml:

```bash
conda env create -f envs/clustering.yaml
```

Or, if you prefer mamba:

```bash
mamba env create -f envs/clustering.yaml
```

## Project structure

```
clustering/
├── compare_flu/                # Training pipeline (2016-2018, RANDOM_SEED=314159)
│   ├── Snakefile
│   └── profile/config.yaml
├── compare_flu2018-2020/       # Test pipeline (2018-2020, RANDOM_SEED=314159)
│   ├── Snakefile
│   └── profile/config.yaml
├── compare_flu_seed{2,3,4}/           # Robustness replicates — same pipeline as
├── compare_flu2018-2020_seed{2,3,4}/  # above but with different subsampling seeds:
│                                      #   seed2 = 271828 (e)
│                                      #   seed3 = 161803 (φ)
│                                      #   seed4 = 141421 (√2)
│                                      # These replicates use optimal clustering
│                                      # parameters from the original training run.
├── config/                     # Reference sequences, clades, and augur configs
├── data/                       # Input FASTA (NCBI H3N2 HA)
├── envs/                       # Conda environment definition
├── scripts/                    # Python scripts for extraction, reduction, clustering
└── simulations/                # Simulation configs and parameter files
```

## Running

### Training pipeline (2016-2018)

```bash
cd compare_flu
snakemake --profile profile
```

This will:
1. Filter and align H3N2 HA sequences (2016-2018)
2. Build a phylogenetic tree and assign clades
3. Extract embeddings using ESM-2 (t33-650M, t36-3B, t48-15B), ProtBERT, ProtT5, and CARP
4. Reduce embeddings with PCA, t-SNE, UMAP, and MDS (multiple metrics and component counts)
5. Cluster with HDBSCAN across distance thresholds and evaluate against clade assignments
6. Export results as an Auspice JSON for visualization

### Test pipeline (2018-2020)

The test pipeline requires the training pipeline to be completed first, as it uses the optimal clustering parameters found during training.

```bash
cd compare_flu2018-2020
snakemake --profile profile
```

This runs the same embedding and reduction steps on 2018-2020 sequences, clusters using the optimal parameters from training, and produces a `best_per_model_HDBSCAN.csv` comparing train vs. test performance for each model.

### Running specific rules

To run only a specific step, specify the target:

```bash
snakemake --profile profile results/embed_prot5.csv
```

### GPU vs CPU

Most PLM extraction rules request a GPU (`resources: gpu=1`). The t48-15B model falls back to CPU with multi-threading if no GPU is available. Adjust `CPU_THREADS` at the top of each Snakefile as needed.

## Acknowledgments

Rules, scripts, data and configurations adapted from [blab/cartography](https://github.com/blab/cartography)

Last commit of the version used: 3fcac4ef2a73e5929ea018f11555ec84b88371dc

Copyright (c) 2019 Bedford Lab.

> Nanduri, S., Black, A., Bedford, T., & Huddleston, J. (2024). Dimensionality reduction distills complex evolutionary relationships in seasonal influenza and SARS-CoV-2. *Virus Evolution*, veae087. https://doi.org/10.1093/ve/veae087

### Modifications done
The [blab/cartography](https://github.com/blab/cartography) results we report in our paper can be different than the ones reported in the [blab/cartography](https://doi.org/10.1093/ve/veae087).

The pipeline was not modified if not needed, to keep the baseline results as comparable as possible. But some modifications had to be applied.

In the original pipeline the tree generating ran on multiple threads, thus making the tree generation non-deterministic (even when fixing a seed). We fixed the seed, set the thread number to 1, and to make labels more reliable we ran the algorithm with a bootstrap option.

The value of undefined cluster from HDBSCAN was not set to correct one. We set it to correct one, now it shows the possible outlier numbers.


