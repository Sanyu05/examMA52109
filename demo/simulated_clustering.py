"""
Simulated dataset clustering demo for MA52109.

This script demonstrates a convincing clustering analysis by:
- plotting the raw data
- standardising features
- computing an elbow curve
- selecting an appropriate k value
- running clustering
- producing clear visualisations of the cluster structure
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from cluster_maker import (
    select_features,
    standardise_features,
    run_clustering,
    elbow_curve,
    plot_elbow,
)


OUTPUT_DIR = "demo_output_simulated"


def plot_raw_data(X, title="Raw Data (First Two Features)"):
    """Simple scatter plot of the raw data before clustering."""
    fig, ax = plt.subplots()
    ax.scatter(X[:, 0], X[:, 1], alpha=0.7)
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def main():

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load dataset
    data_path = "data/simulated_data.csv"
    df = pd.read_csv(data_path)

    # Select numeric columns
    numeric_cols = [
        col for col in df.columns
        if pd.api.types.is_numeric_dtype(df[col])
    ]
    feature_cols = numeric_cols

    # Convert to NumPy
    X_df = select_features(df, feature_cols)
    X = X_df.to_numpy()

    # --- 1. Plot the raw data ---
    fig_raw = plot_raw_data(X, title="Raw Simulated Data (Unstandardised)")
    fig_raw.savefig(os.path.join(OUTPUT_DIR, "raw_data.png"), dpi=150)
    plt.close(fig_raw)

    # Standardise
    X_std = standardise_features(X)

    # --- 2. Elbow Analysis ---
    k_values = list(range(1, 9))
    inertias = elbow_curve(X_std, k_values, random_state=42)
    inertia_list = [inertias[k] for k in k_values]

    fig_elbow, _ = plot_elbow(k_values, inertia_list, title="Elbow Curve (Simulated Data)")
    fig_elbow.savefig(os.path.join(OUTPUT_DIR, "elbow_curve.png"), dpi=150)
    plt.close(fig_elbow)

    # Choose k based on the elbow (usually 3 for this dataset)
    chosen_k = 3

    # --- 3. Final Clustering ---
    result = run_clustering(
        input_path=data_path,
        feature_cols=feature_cols,
        algorithm="kmeans",
        k=chosen_k,
        standardise=True,
        random_state=42,
        output_path=os.path.join(OUTPUT_DIR, f"simulated_clustered_k{chosen_k}.csv"),
        compute_elbow=False,
    )

    # Save cluster plot
    fig_cluster = result["fig_cluster"]
    fig_cluster.savefig(os.path.join(OUTPUT_DIR, f"clusters_k{chosen_k}.png"), dpi=150)
    plt.close(fig_cluster)

    # Print metrics
    print("\nFinal Clustering Metrics:")
    for key, value in result["metrics"].items():
        print(f"  {key}: {value}")

    print("\nAnalysis complete. All plots saved in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
