# Task 2 – Explanation of Bug Fix and Package Overview

## 1. What was wrong with the original demo script?

The demo script in `demo/cluster_plot.py` was intended to run K-means clustering
for four different values of *k*: 2, 3, 4, and 5.

However, the script contained an incorrect line:

```python
k = min(k, 3)
````

This forced `k` never to exceed 3, meaning:

* k = 2 → ran correctly
* k = 3 → ran correctly
* k = 4 → actually ran with k = 3
* k = 5 → also ran with k = 3

As a result, the inertia and silhouette values for k = 3, 4, and 5 were identical.
The script was not performing clustering for the intended number of clusters.

## 2. How the bug was fixed

The fix was simply to remove the cap on k and pass the correct value to the
clustering function:

```python
k=k,
```

Now each iteration correctly runs K-means with the intended number of clusters.

## 3. What the corrected demo script now does

After fixing the bug, the script now:

1. Loads the input CSV file.
2. Automatically selects the first two numeric columns as features.
3. Runs K-means clustering for k = 2, 3, 4, and 5.
4. Saves:

   * A PNG cluster plot for each k
   * A CSV file with cluster labels for each k
   * A summary CSV of all metrics (inertia and silhouette)
5. Prints the clustering metrics to the terminal.
6. Stores everything in the `demo_output/` directory.

The script now behaves exactly as intended.

## 4. Overview of the `cluster_maker` package

`cluster_maker` is a small, modular package designed for teaching and practicing
data clustering workflows. It provides tools for generating data, preparing it
for analysis, applying clustering algorithms, evaluating the results, and
visualising the output. The package is organised into clear components:

### **Data generation (`dataframe_builder`)**
Provides functions to:
- Define synthetic cluster centres.
- Simulate datasets around those centres using Gaussian noise.

This makes it easy to create controlled data for testing and demonstrations.

### **Preprocessing (`preprocessing`)**
Includes tools for:
- Selecting specific feature columns from a DataFrame.
- Ensuring those features are numeric.
- Standardising data to zero mean and unit variance.

These steps prepare raw data so clustering algorithms behave correctly.

### **Clustering algorithms (`algorithms`)**
Contains:
- A simple, manual implementation of K-means.
- A wrapper around scikit-learn’s KMeans.
- Helper functions for centroid initialisation, assignment, and updates.

This module provides the core clustering functionality.

### **Evaluation (`evaluation`)**
Computes:
- Inertia (within-cluster sum of squares).
- Silhouette scores.
- Elbow curves for choosing a suitable number of clusters.

These metrics help assess clustering quality.

### **Plotting (`plotting_clustered`)**
Generates:
- 2D scatter plots showing labelled clusters and centroids.
- Elbow plots showing inertia vs. k.

These visualisations help interpret cluster structure.

### **High-level interface (`interface`)**
The `run_clustering()` function combines all stages:
1. Load data
2. Preprocess features  
3. Run a chosen clustering algorithm  
4. Compute evaluation metrics  
5. Produce plots  
6. Optionally export labelled data  

This gives users a simple, end-to-end clustering workflow in a single call.

Overall, the package provides a complete, educational pipeline for clustering:
from raw data to processed results, evaluation, and clear visualisation.


