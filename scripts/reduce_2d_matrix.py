import argparse
import pandas as pd
from sklearn.manifold import TSNE, MDS
import umap
import numpy as np


def reduce_dimensions(input_file, method, output_file, n_components=2, random_seed=None):
    # Load the precomputed distance matrix (first column = labels)
    data = pd.read_csv(input_file, index_col=0)
    labels = data.index.to_series().reset_index(drop=True)
    distances = data.to_numpy()

    if method == 't-sne':
        if n_components > 2:
            reducer = TSNE(n_components=n_components, metric='precomputed', method='exact',
                           init='random', random_state=random_seed)
        else:
            reducer = TSNE(n_components=n_components, metric='precomputed',
                           init='random', random_state=random_seed)
    elif method == 'umap':
        reducer = umap.UMAP(n_components=n_components, metric='precomputed', random_state=random_seed)
    elif method == 'mds':
        reducer = MDS(n_components=n_components, dissimilarity='precomputed', random_state=random_seed)
    else:
        raise ValueError(f"Unsupported method for a distance matrix: {method}")

    embeddings = reducer.fit_transform(distances)

    reduced_data = pd.DataFrame(embeddings, columns=[f'{method}{i}' for i in range(n_components)])
    reduced_data.insert(0, 'label', labels)

    reduced_data.to_csv(output_file, index=False)
    print(f"Reduced data saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reduce a precomputed distance matrix using t-SNE, UMAP, or MDS")
    parser.add_argument("input_file", help="Input distance-matrix CSV (square, index_col=0)")
    parser.add_argument("method", choices=["t-sne", "umap", "mds"], help="Dimensionality reduction method")
    parser.add_argument("output_file", help="Output CSV file")
    parser.add_argument("--n_components", default=2, type=int)
    parser.add_argument("--random-seed", default=None, type=int)

    args = parser.parse_args()

    reduce_dimensions(args.input_file, args.method, args.output_file, args.n_components, args.random_seed)
