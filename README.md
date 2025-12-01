# cluster_maker
`cluster_maker` is a small educational Python package for simulating clustered
datasets and running clustering analyses with a simple, user-friendly interface.

It is designed for practicals and exams where students are given an incomplete
or faulty version of the package and asked to debug or extend it.

## Allowed libraries
The package only uses:

- Python standard library  
- NumPy  
- pandas  
- matplotlib  
- SciPy  
- scikit-learn  

No other third-party libraries are required.

## Main features
- Define a **seed DataFrame** describing cluster centres  
- Simulate clustered data around these centres  
- Compute basic **descriptive statistics** and **correlations**  
- Preprocess data: feature selection and standardisation  
- Run clustering with:
  - a simple **manual K-means** implementation  
  - a scikit-learn **KMeans** wrapper  
- Evaluate clustering with:
  - **inertia** (within-cluster sum of squares)  
  - **silhouette score**  
  - **elbow curve** for K selection  
- Plot:
  - 2D cluster scatter with optional centroids  
  - elbow curve  
- High-level **`run_clustering`** interface  
- Demo scripts and unit tests

## Package root directory structure
- `cluster_maker/`
  - `dataframe_builder.py` – build seed DataFrame and simulate clustered data  
  - `data_analyser.py` – descriptive statistics and correlation  
  - `data_exporter.py` – CSV and formatted text export  
  - `preprocessing.py` – feature selection and standardisation  
  - `algorithms.py` – manual K-means and scikit-learn KMeans wrapper  
  - `evaluation.py` – inertia, silhouette, elbow curve  
  - `plotting_clustered.py` – 2D cluster plots and elbow plots  
  - `interface.py` – high-level `run_clustering` function  
- `demo/` – example scripts  
- `data/` - csv data file used by the example scripts
- `tests/` – basic unit tests using the standard library `unittest`

## Installation (local use)
From the root directory of the project, run:

```bash
pip install -e .
```

This installs the package in editable mode, meaning you can modify the files
and re-run tests or demos without reinstalling.

## Notes on pyproject.toml and the *.egg-info directory
This project includes a small file named pyproject.toml.
You do not need to open or edit it. Its only purpose is to tell Python/pip
that this folder is a valid installable package. Without it, the command
pip install -e . would fail.

When you run the installation command, pip automatically creates a directory
called something like:

`cluster_maker.egg-info/`

This folder contains package metadata used internally by Python (file lists,
version information, etc.). It is generated automatically and should not be
edited.

# 📘 MA52109 Mock Practical – Student Submission Overview

This repository contains my completed submission for the MA52109: Programming for Data Science mock practical.  
The practical consisted of five tasks focused on debugging, extending, and demonstrating the functionality of the `cluster_maker` package.

Below is a summary of how each task was completed and where the relevant work is located in this repository.

---

## ✅ Task 1 — Fixing the clustering algorithms (30 marks)

The provided test suite (`tests/test_algorithms.py`) revealed several issues in the `algorithms.py` module.  
I corrected:

- incorrect centroid initialisation (`k+1` instead of `k`)
- incorrect validation (`k > n_samples`)
- incorrect cluster assignment (using `argmax` instead of `argmin`)
- consistency issues with labels and centroids

All fixes were applied **without modifying the test file**, and all tests now pass

---

## ✅ Task 2 — Debugging the demo script and explanation (20 marks)

The original demo script capped the number of clusters using:

```python
k = min(k, 3)
```

This caused incorrect behaviour when running for `k = 4` and `k = 5`.

I fixed this by correctly passing:

```python
k = k
```

The script now performs clustering for all intended values (2, 3, 4, 5) and saves the correct plots and metrics.

A full explanation of the bug and its fix is documented in
**`EXPLANATION.md`** (located at the project root).

---

## ✅ Task 3 — New preprocessing tests (15 marks)

I added a new test file:

```
tests/test_preprocessing.py
```

containing exactly three meaningful unit tests that check:

1. missing feature columns
2. non-numeric feature handling
3. correctness of standardisation (zero mean, unit variance)

Each test includes a comment explaining the real-world issue it detects.

---

## ✅ Task 4 — Clustering on simulated data (15 marks)

I created a new demo script:

```
demo/simulated_clustering.py
```

The script:

* loads `simulated_data.csv`
* displays a raw scatter plot of the dataset
* standardises the features
* computes an elbow curve (k = 1–8)
* selects a plausible `k = 3`
* runs clustering using only `cluster_maker` tools
* saves clear and meaningful visualisations

All outputs are stored in:

```
demo_output_simulated/
```

---

## ✅ Task 5 — Agglomerative clustering extension (20 marks)

I implemented a new module:

```
cluster_maker/agglomerative.py
```

following the design of the existing package.
It uses `sklearn.cluster.AgglomerativeClustering` and returns `(labels, None)` for compatibility.

I also created a new demo script:

```
demo/demo_agglomerative.py
```

which performs hierarchical clustering on `difficult_dataset.csv` and produces raw and clustered plots demonstrating the algorithm’s effectiveness.

Outputs are stored in:

```
demo_output_agglomerative/
```

---

## 📂 Directory Summary

```
cluster_maker/             Package source code
demo/                      Demo scripts for tasks 2, 4, and 5
tests/                     Automated test suite
data/                      Provided datasets
demo_output*/              Generated clustering outputs
EXPLANATION.md             Required written explanation (Task 2)
README.md                  Main package documentation
```

---


