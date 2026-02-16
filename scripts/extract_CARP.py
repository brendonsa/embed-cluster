import argparse
from pathlib import Path

import torch
import torch.cuda
from tqdm import tqdm
import numpy as np
import pandas as pd

from sequence_models.utils import parse_fasta
from sequence_models.pretrained import load_model_and_alphabet

parser = argparse.ArgumentParser()
parser.add_argument('model')
parser.add_argument('in_fpath')
parser.add_argument('out_dir')
parser.add_argument('--repr_layers', nargs='*',
                    default=[-1])          # e.g., -1 33
# mean, per_tok, logp
parser.add_argument('--include', nargs='*', default=['mean'])
parser.add_argument('--device', default=None)
parser.add_argument('--batchsize', default=1, type=int)
parser.add_argument('--cpu_threads', type=int, default=0,
                    help='Number of CPU threads for PyTorch (0 = all available cores)')
args = parser.parse_args()

print('Loading model...')
model, collater = load_model_and_alphabet(args.model)
model.eval()

# device
if args.device is None:
    device = torch.device(
        'cuda:0') if torch.cuda.is_available() else torch.device('cpu')
else:
    device = torch.device(args.device)

if device.type == 'cpu':
    import os
    cpu_threads = args.cpu_threads if args.cpu_threads > 0 else os.cpu_count()
    torch.set_num_threads(cpu_threads)
    torch.set_num_interop_threads(cpu_threads)
    print(f"Using {cpu_threads} CPU threads for inference")

model = model.to(device)

# data
print('Loading data...')
seqs, names = parse_fasta(args.in_fpath, return_names=True)
ells = [len(s) for s in seqs]
seqs = [[s] for s in seqs]  # collater expects list of lists
n_total = len(seqs)

# repr layers and logits flag (kept for compatibility)
repr_layers = []
logits = False
for r in args.repr_layers:
    if r == 'logits':
        logits = True
    else:
        repr_layers.append(int(r))

include_mean = 'mean' in args.include
include_per_tok = 'per_tok' in args.include
include_logp = 'logp' in args.include

# output dir
out_dir = Path(args.out_dir)
out_dir.mkdir(parents=True, exist_ok=True)

# optional logp CSV
if include_logp:
    logps = np.empty(len(seqs))

with torch.no_grad(), tqdm(total=n_total) as pbar:
    for i in range(0, n_total, args.batchsize):
        start = i
        end = min(start + args.batchsize, n_total)
        bs = seqs[start:end]
        bn = names[start:end]
        bl = ells[start:end]

        batch = collater(bs)
        x = batch[0].to(device)

        results = model(x, repr_layers=repr_layers, logits=logits)

        reps_by_layer = results.get(
            'representations', {})  # {layer: [B, L, D]}

        for j, (name, ell) in enumerate(zip(bn, bl)):
            result = {"label": name}

            if reps_by_layer:
                seq_token_reps = {
                    int(layer): reps_by_layer[layer][j, :ell].detach().cpu().clone()
                    for layer in reps_by_layer.keys()
                }

                if include_per_tok:
                    # match ESM: top-level key "representations"
                    result["representations"] = seq_token_reps

                if include_mean:
                    # match ESM: top-level key "mean_representations"
                    result["mean_representations"] = {
                        layer: t.mean(dim=0).clone() for layer, t in seq_token_reps.items()
                    }

            if logits:
                seq_logits = results['logits'][j, :ell].detach().cpu()
                if include_logp:
                    src = x[j, :ell].detach().cpu()
                    log_probs = seq_logits.log_softmax(dim=-1)
                    token_lp = log_probs.gather(-1,
                                                src.view(-1, 1)).squeeze(-1)
                    avg_lp = float(token_lp.mean().item())
                    result["logp"] = avg_lp
                    logps[start + j] = avg_lp

            out_file = (out_dir / name).with_suffix(".pt")
            out_file.parent.mkdir(parents=True, exist_ok=True)
            torch.save(result, out_file)

        pbar.update(len(bs))

if include_logp:
    df = pd.DataFrame({"name": names, "sequence": [
                      s[0] for s in seqs], "logp": logps})
    out_fpath = out_dir / f"{args.model}_logp.csv"
    print(f"Writing results to {out_fpath}")
    df.to_csv(out_fpath, index=False)
