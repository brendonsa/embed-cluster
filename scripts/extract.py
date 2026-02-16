import argparse
import pathlib
import torch
from esm import Alphabet, FastaBatchedDataset, ProteinBertModel, pretrained, MSATransformer


def create_parser():
    parser = argparse.ArgumentParser(
        description="Extract per-token representations with optional windowed processing for long sequences."
    )
    parser.add_argument("model_location", type=str,
                        help="Model file or pretrained model name")
    parser.add_argument("fasta_file", type=pathlib.Path,
                        help="FASTA file to extract from")
    parser.add_argument("output_dir", type=pathlib.Path,
                        help="Directory to save output files")
    parser.add_argument("--toks_per_batch", type=int, default=4096)
    parser.add_argument("--repr_layers", type=int, nargs="+", default=[-1])
    parser.add_argument("--include", type=str, nargs="+",
                        choices=["mean", "per_tok", "bos", "contacts", "attentions",
                                 "attentionmean", "sitemean"], required=True)
    parser.add_argument("--sites_file", type=pathlib.Path, default=None,
                        help="File with site indices (0-based, one per line or comma-separated) for site-mean mode")
    parser.add_argument("--window_size", type=int, default=None,
                        help="Window size for windowed extraction (optional)")
    parser.add_argument("--stride", type=int, default=None,
                        help="Stride for windowed extraction (optional)")
    parser.add_argument("--nogpu", action="store_true")
    parser.add_argument("--cuda_device", type=int, default=0)
    parser.add_argument("--cpu_threads", type=int, default=0,
                        help="Number of CPU threads for PyTorch (0 = all available cores)")
    parser.add_argument("--num_workers", type=int, default=0,
                        help="Number of DataLoader worker processes")
    return parser


def sliding_windows(tensor, window_size, stride):
    seq_len = tensor.size(0)
    for start in range(0, seq_len - window_size + 1, stride):
        end = start + window_size
        yield tensor[start:end], start, end


def run(args):
    model, alphabet = pretrained.load_model_and_alphabet(args.model_location)
    model.eval()

    if isinstance(model, MSATransformer):
        raise ValueError("MSA Transformer is not supported in this script.")

    if torch.cuda.is_available() and not args.nogpu:
        device = torch.device(f"cuda:{args.cuda_device}")
        model = model.to(device)
        print(f"Transferred model to GPU device {args.cuda_device}")
    else:
        device = torch.device("cpu")
        import os
        cpu_threads = args.cpu_threads if args.cpu_threads > 0 else os.cpu_count()
        torch.set_num_threads(cpu_threads)
        torch.set_num_interop_threads(cpu_threads)
        print(f"Using {cpu_threads} CPU threads for inference")

    toks_per_batch = args.toks_per_batch
    if "attentions" in args.include or "attentionmean" in args.include:
        toks_per_batch = max(1, toks_per_batch // 4)
        print(
            f"Attentions requested: reducing toks_per_batch to {toks_per_batch} to fit in memory")

    dataset = FastaBatchedDataset.from_file(args.fasta_file)
    batches = dataset.get_batch_indices(
        toks_per_batch, extra_toks_per_seq=1)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=alphabet.get_batch_converter(),
        batch_sampler=batches,
        num_workers=args.num_workers,
    )

    print(f"Read {args.fasta_file} with {len(dataset)} sequences")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if "sitemean" in args.include:
        if args.sites_file is None:
            raise ValueError("--sites_file is required when using site-mean")
        with open(args.sites_file) as f:
            text = f.read().strip()
        sites = [int(s.strip()) for s in text.replace('\n', ',').split(',') if s.strip()]
        print(f"Site-informed mode: using {len(sites)} sites: {sites}")

    return_contacts = "contacts" in args.include
    return_attentions = "attentions" in args.include or "attentionmean" in args.include
    assert all(-(model.num_layers + 1) <= i <=
               model.num_layers for i in args.repr_layers)
    repr_layers = [(i + model.num_layers + 1) %
                   (model.num_layers + 1) for i in args.repr_layers]

    with torch.inference_mode():
        for batch_idx, (labels, strs, toks) in enumerate(data_loader):
            print(
                f"Processing batch {batch_idx + 1}/{len(batches)} ({toks.size(0)} sequences)")

            strs = [s.replace('*', 'X') for s in strs]
            toks = toks.to(device=device, non_blocking=True)

            out = model(toks, repr_layers=repr_layers,
                        return_contacts=return_contacts,
                        need_head_weights=return_attentions)
            representations = {
                layer: t.to(device="cpu").detach()
                for layer, t in out["representations"].items()
            }
            if return_contacts:
                contacts = out["contacts"].to(device="cpu").detach()
            if return_attentions:
                attn_all = out["attentions"].to(device="cpu").detach()
                attentions = {
                    layer: attn_all[:, layer - 1]
                    for layer in repr_layers if layer > 0
                }

            for i, label in enumerate(labels):
                seq_len = len(strs[i])
                token_repr = {
                    layer: t[i, 1:seq_len + 1]
                    for layer, t in representations.items()
                }

                result = {"label": label}

                full_result = {}

                if "per_tok" in args.include:
                    full_result["representations"] = {
                        layer: token_repr[layer].clone()
                        for layer in repr_layers
                    }

                if "mean" in args.include:
                    full_result["mean_representations"] = {
                        layer: token_repr[layer].mean(0).clone()
                        for layer in repr_layers
                    }

                if "attentionmean" in args.include:
                    full_result["attention_mean_representations"] = {}
                    for layer in repr_layers:
                        if layer in attentions:
                            # attentions[layer][i] shape: (heads, seq_len, seq_len)
                            attn = attentions[layer][i, :, 1:seq_len + 1, 1:seq_len + 1]
                            # Average across heads and source positions to get per-token importance
                            weights = attn.mean(dim=0).mean(dim=0)  # (seq_len,)
                            weights = weights / weights.sum()
                            full_result["attention_mean_representations"][layer] = (
                                (token_repr[layer] * weights.unsqueeze(-1)).sum(0).clone()
                            )

                if "sitemean" in args.include:
                    site_indices = [s for s in sites if s < seq_len]
                    if len(site_indices) == 0:
                        print(f"Warning: no valid sites for {label} (seq_len={seq_len}), falling back to full mean")
                        full_result["site_mean_representations"] = {
                            layer: token_repr[layer].mean(0).clone()
                            for layer in repr_layers
                        }
                    else:
                        idx = torch.tensor(site_indices, dtype=torch.long)
                        full_result["site_mean_representations"] = {
                            layer: token_repr[layer][idx].mean(0).clone()
                            for layer in repr_layers
                        }

                if "bos" in args.include:
                    full_result["bos_representations"] = {
                        layer: representations[layer][i, 0].clone()
                        for layer in repr_layers
                    }

                if return_contacts:
                    full_result["contacts"] = contacts[i,
                                                       :seq_len, :seq_len].clone()

                if return_attentions:
                    full_result["attentions"] = {
                        layer: attentions[layer][i, :,
                                                 1:seq_len + 1, 1:seq_len + 1].clone()
                        for layer in repr_layers if layer in attentions
                    }

                result["full"] = full_result

                # Optionally extract windowed representations if window_size and stride are provided
                if args.window_size is not None and args.stride is not None:
                    result["windows"] = []
                    for window_tensor, start, end in sliding_windows(
                        token_repr[repr_layers[0]
                                   ], args.window_size, args.stride
                    ):
                        window_result = {"start": start, "end": end}

                        if "per_tok" in args.include:
                            window_result["representations"] = {
                                layer: token_repr[layer][start:end].clone()
                                for layer in repr_layers
                            }

                        if "mean" in args.include:
                            window_result["mean_representations"] = {
                                layer: token_repr[layer][start:end].mean(
                                    0).clone()
                                for layer in repr_layers
                            }

                        if "attentionmean" in args.include:
                            for layer in repr_layers:
                                if layer in attentions:
                                    attn = attentions[layer][i, :, start + 1:end + 1, start + 1:end + 1]
                                    weights = attn.mean(dim=0).mean(dim=0)
                                    weights = weights / weights.sum()
                                    window_result.setdefault("attention_mean_representations", {})[layer] = (
                                        (token_repr[layer][start:end] * weights.unsqueeze(-1)).sum(0).clone()
                                    )

                        if "sitemean" in args.include:
                            window_sites = [s - start for s in sites if start <= s < end]
                            if len(window_sites) == 0:
                                window_result["site_mean_representations"] = {
                                    layer: token_repr[layer][start:end].mean(0).clone()
                                    for layer in repr_layers
                                }
                            else:
                                idx = torch.tensor(window_sites, dtype=torch.long)
                                window_result["site_mean_representations"] = {
                                    layer: token_repr[layer][start:end][idx].mean(0).clone()
                                    for layer in repr_layers
                                }

                        if "bos" in args.include:
                            window_result["bos_representations"] = {
                                layer: representations[layer][i, 0].clone()
                                for layer in repr_layers
                            }

                        if return_contacts:
                            window_result["contacts"] = contacts[i,
                                                                 start:end, start:end].clone()

                        if return_attentions:
                            window_result["attentions"] = {
                                layer: attentions[layer][i, :, start +
                                                         1:end + 1, start + 1:end + 1].clone()
                                for layer in repr_layers if layer in attentions
                            }

                        result["windows"].append(window_result)

                output_file = args.output_dir / f"{label}.pt"
                output_file.parent.mkdir(parents=True, exist_ok=True)
                torch.save(result, output_file)


def main():
    parser = create_parser()
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
