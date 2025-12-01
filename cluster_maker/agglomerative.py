###
## cluster_maker
## Agglomerative clustering module
## Designed to integrate with the rest of the package
###

from __future__ import annotations

from typing import Tuple, Optional
import numpy as np
from sklearn.cluster import AgglomerativeClustering


def agglomerative_clustering(
    X: np.ndarray,
    n_clusters: int = 3,
    linkage: str = "ward",
) -> Tuple[np.ndarray, None]:
    """
    Perform hierarchical agglomerative clustering using scikit-learn.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
    n_clusters : int
        Number of clusters to form.
    linkage : str
        Linkage criterion: "ward", "single", "complete", "average".

    Returns
    -------
    labels : ndarray of shape (n_samples,)
        Cluster assignment for each sample.

    centroids : None
        Agglomerative clustering does not compute centroids, so this is None.
        This keeps compatibility with run_clustering and the plotting functions.
    """
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a NumPy array.")
    if n_clusters <= 1:
        raise ValueError("n_clusters must be at least 2.")

    model = AgglomerativeClustering(
        n_clusters=n_clusters,
        linkage=linkage,
    )
    labels = model.fit_predict(X)

    # No centroids in hierarchical clustering → return None
    return labels, None
