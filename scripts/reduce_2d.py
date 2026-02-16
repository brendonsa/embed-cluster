import argparse
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, MDS
import umap
import numpy as np

def reduce_dimensions(input_file, method, output_file, metric='euclidean', n_components = 2, random_seed=None):
    # Load the CSV file
    data = pd.read_csv(input_file)

    # Separate labels (first column) and features (remaining columns)
    labels = data.iloc[:, 0]
    features = data.iloc[:, 1:]

    # Check the method and apply dimensionality reduction
    if method == 'pca':
        reducer = PCA(n_components=n_components, random_state=random_seed)
    elif method == 't-sne':
        if n_components >2:
            reducer = TSNE(n_components=n_components, metric=metric, method='exact', random_state=random_seed)
        else:
            reducer = TSNE(n_components=n_components, metric=metric, random_state=random_seed)
    elif method == 'umap':
        reducer = umap.UMAP(n_components=n_components, metric=metric, random_state=random_seed)
    elif method == 'mds':
        reducer = MDS(n_components=n_components, random_state=random_seed)
    else:
        raise ValueError(f"Unsupported method: {method}")

    # Fit and transform the features to 2D
    embeddings = reducer.fit_transform(features)

    # Create a new DataFrame with the labels and reduced dimensions
    reduced_data = pd.DataFrame(embeddings, columns=[f'{method}{i}' for i in range(n_components)])
    reduced_data.insert(0, 'label', labels)

    # Save the new DataFrame to a CSV file
    reduced_data.to_csv(output_file, index=False)
    print(f"Reduced data saved to {output_file}")

if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Reduce dimensions using PCA, t-SNE, or UMAP")
    parser.add_argument("input_file", help="Input CSV file")
    parser.add_argument("method", choices=["pca", "t-sne", "umap","mds"], help="Dimensionality reduction method")
    parser.add_argument("output_file", help="Output CSV file")
    parser.add_argument("--metric", default="euclidean", required=False, choices=["euclidean","cosine"])
    parser.add_argument("--n_components", default=2, type=int)
    parser.add_argument("--random-seed", default=None, type=int)

    args = parser.parse_args()

    # Call the function to reduce dimensions
    reduce_dimensions(args.input_file, args.method, args.output_file, args.metric, args.n_components, args.random_seed)