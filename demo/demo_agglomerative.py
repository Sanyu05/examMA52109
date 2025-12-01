"""
Demo script for agglomerative hierarchical clustering using the new
agglomerative_clustering module in cluster_maker.

This script loads the difficult_dataset.csv file, visualises the raw data,
runs hierarchical clustering, and plots the result.
"""

from __future__ import annotations

import os
import pandas as pd
import matplotlib.pyplot as plt

from cluster_maker import (
    select_features,
    standardise_features,
    agglomerative_clustering,
    plot_clusters_2d,
)

OUTPUT_DIR = "demo_output_agglomerative"


def plot_raw_data(X):
    fig, ax = plt.subplots()
    ax.scatter(X[:, 0], X[:, 1], alpha=0.7)
    ax.set_title("Raw Difficult Dataset")
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    fig.tight_layout()
    return fig


def main():

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load data
    data_path = "data/difficult_dataset.csv"
    df = pd.read_csv(data_path)

    # Select numeric columns
    numeric_cols = [
        col for col in df.columns
        if pd.api.types.is_numeric_dtype(df[col])
    ]
    feature_cols = numeric_cols

    # Extract & standardise features
    X_df = select_features(df, feature_cols)
    X = standardise_features(X_df.to_numpy())

    # Raw data plot
    fig_raw = plot_raw_data(X)
    fig_raw.savefig(os.path.join(OUTPUT_DIR, "raw_data.png"), dpi=150)
    plt.close(fig_raw)

    # Choose number of clusters (usually 2 or 3 for difficult datasets)
    chosen_k = 3

    # Run agglomerative clustering
    labels, _ = agglomerative_clustering(X, n_clusters=chosen_k, linkage="ward")

    # Cluster plot (centroids=None is fine)
    fig_cluster, _ = plot_clusters_2d(
        X,
        labels,
        centroids=None,
        title=f"Agglomerative Clustering (k={chosen_k})"
    )
    fig_cluster.savefig(os.path.join(OUTPUT_DIR, f"clusters_k{chosen_k}.png"), dpi=150)
    plt.close(fig_cluster)

    print("\nAgglomerative clustering complete.")
    print("Plots saved in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
