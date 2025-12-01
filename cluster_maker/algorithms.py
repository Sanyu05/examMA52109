### 
## cluster_maker
## James Foadi - University of Bath
## November 2025
###

from __future__ import annotations

from typing import Tuple, Optional

import numpy as np
from sklearn.cluster import KMeans


def init_centroids(
    X: np.ndarray,
    k: int,
    random_state: Optional[int] = None,
) -> np.ndarray:
    """
    Initialise centroids by randomly sampling points from X without replacement.
    """
    if k <= 0:
        raise ValueError("k must be a positive integer.")

    n_samples = X.shape[0]
    if k > n_samples:
        raise ValueError("k cannot be larger than the number of samples.")

    rng = np.random.RandomState(random_state)
    indices = rng.choice(n_samples, size=k, replace=False)
    return X[indices]


def assign_clusters(X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """
    Assign each sample to the nearest centroid (Euclidean distance).
    """
    # X: (n_samples, n_features)
    # centroids: (k, n_features)
    diff = X[:, np.newaxis, :] - centroids[np.newaxis, :, :]
    distances = np.linalg.norm(diff, axis=2)  # (n_samples, k)

    # FIX: choose nearest centroid, not furthest
    labels = np.argmin(distances, axis=1)
    return labels


def update_centroids(
    X: np.ndarray,
    labels: np.ndarray,
    k: int,
    random_state: Optional[int] = None,
) -> np.ndarray:
    """
    Update centroids by taking the mean of points in each cluster.
    If a cluster becomes empty, re-initialise its centroid randomly from X.
    """
    n_features = X.shape[1]
    new_centroids = np.zeros((k, n_features), dtype=float)
    rng = np.random.RandomState(random_state)

    for cluster_id in range(k):
        mask = labels == cluster_id
        if not np.any(mask):
            idx = rng.randint(0, X.shape[0])
            new_centroids[cluster_id] = X[idx]
        else:
            new_centroids[cluster_id] = X[mask].mean(axis=0)

    return new_centroids


def kmeans(
    X: np.ndarray,
    k: int,
    max_iter: int = 300,
    tol: float = 1e-4,
    random_state: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simple manual K-means implementation.
    """
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a NumPy array.")

    centroids = init_centroids(X, k, random_state=random_state)

    for _ in range(max_iter):
        labels = assign_clusters(X, centroids)
        new_centroids = update_centroids(X, labels, k, random_state=random_state)
        shift = np.linalg.norm(new_centroids - centroids)
        centroids = new_centroids
        if shift < tol:
            break

    labels = assign_clusters(X, centroids)
    return labels, centroids


def sklearn_kmeans(
    X: np.ndarray,
    k: int,
    random_state: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Thin wrapper around scikit-learn's KMeans.
    """
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a NumPy array.")

    model = KMeans(
        n_clusters=k,
        random_state=random_state,
        n_init=10,
    )
    model.fit(X)
    labels = model.labels_
    centroids = model.cluster_centers_
    return labels, centroids
