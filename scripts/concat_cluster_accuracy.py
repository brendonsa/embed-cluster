import pandas as pd

df = pd.concat(
    [pd.read_csv(f) for f in snakemake.input.accuracies],
    ignore_index=True,
)
df = df.sort_values(by=[snakemake.params.column])
cols = [snakemake.params.column] + df.columns.drop(snakemake.params.column).tolist()
df[cols].to_csv(snakemake.output.metadata, index=False)
