"""For each model group, select the best method (lowest normalized_vi) from
the training (compare_flu) results, then join with matching validation
(compare_flu2018-2020) results to produce a single table with train/validation
columns side by side.
"""
import argparse
import pandas as pd


def method_to_model(method, models):
    """Map a method string to its model group."""
    for model in models:
        if model in method:
            return model
    return "genetic"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--train",
        required=True,
        help="full_HDBSCAN_metadata.csv from training (compare_flu)",
    )
    parser.add_argument(
        "--validation",
        required=True,
        help="full_HDBSCAN_metadata.csv from validation (compare_flu2018-2020)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="list of model names to group by (e.g. t33-650M t36-3B protbert prot5 CARP)",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="output CSV with best method per model, train + validation columns",
    )
    args = parser.parse_args()

    train_df = pd.read_csv(args.train)
    val_df = pd.read_csv(args.validation)

    # Assign model group to each row.
    train_df["model"] = train_df["method"].apply(
        lambda m: method_to_model(m, args.models)
    )

    # The table is already sorted by normalized_vi (ascending).
    # Take the first (best) row per model group.
    best_train = train_df.groupby("model", sort=False).first().reset_index()
    best_methods = best_train["method"].tolist()

    # Filter validation to matching methods.
    best_val = val_df[val_df["method"].isin(best_methods)].copy()

    # Prefix columns (except method) with train_ / validation_.
    merge_cols = ["method"]
    train_renamed = best_train.rename(
        columns={c: f"train_{c}" for c in best_train.columns if c not in merge_cols + ["model"]},
    )
    val_renamed = best_val.rename(
        columns={c: f"validation_{c}" for c in best_val.columns if c not in merge_cols},
    )

    merged = train_renamed.merge(val_renamed, on="method", how="left")

    # Reorder: model, method, vi columns, other scoring metrics, then counts/misc.
    first_cols = ["model", "method"]
    vi_cols = [c for c in ["validation_normalized_vi", "train_normalized_vi"] if c in merged.columns]
    score_metric_names = [
        "adjusted_rand_score",
        "normalized_mutual_info_score",
        "homogeneity_completeness_v_measure",
        "adjusted_mutual_info_score",
    ]
    score_cols = [
        f"{prefix}_{m}"
        for prefix in ["validation", "train"]
        for m in score_metric_names
        if f"{prefix}_{m}" in merged.columns
    ]
    placed = set(first_cols + vi_cols + score_cols)
    remaining = sorted(c for c in merged.columns if c not in placed)
    merged = merged[first_cols + vi_cols + score_cols + remaining]

    merged.to_csv(args.output, index=False)
    print(f"Wrote {len(merged)} rows ({len(best_methods)} best methods) to {args.output}")
