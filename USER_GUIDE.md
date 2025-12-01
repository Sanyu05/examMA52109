# User Guide for Running `cluster_maker` Demos and Tests

This guide is written for users who may not be familiar with the internal structure of the
`cluster_maker` package but want to run the available demo scripts and test files successfully.
It focuses on clarity, ease of use, and a logical flow of instructions.

---

## 📦 Requirements

To use this project, you only need:

- Python 3.10+  
- The libraries listed in `pyproject.toml` (NumPy, pandas, matplotlib, scikit-learn)

Install the package in editable mode from the project root:

```bash
pip install -e .
```

---

## ▶ Running the Automated Test Suite

To verify the correctness of the clustering algorithms and preprocessing tools, run:

```bash
python -m unittest discover
```

This will automatically detect and execute all test files in the `tests/` directory.

**Expected behaviour**:

* Tests run without errors
* Output shows “OK”

---

## ▶ Running the Demo Scripts

Three demo scripts are provided to illustrate the functionality of the package.

### 1. **Basic K-means demo**

File: `demo/cluster_plot.py`

Usage:

```bash
python -m demo.cluster_plot data/demo_data.csv
```

What it does:

* Automatically selects two numeric columns
* Runs K-means for k = 2, 3, 4, 5
* Saves cluster plots and metrics to `demo_output/`

---

### 2. **Simulated dataset clustering**

File: `demo/simulated_clustering.py`

Usage:

```bash
python -m demo.simulated_clustering
```

What it does:

* Loads `simulated_data.csv`
* Plots raw data
* Performs elbow analysis
* Chooses a reasonable number of clusters
* Saves plots and labelled data to `demo_output_simulated/`

---

### 3. **Agglomerative clustering demo**

File: `demo/demo_agglomerative.py`

Usage:

```bash
python -m demo.demo_agglomerative
```

What it does:

* Loads `difficult_dataset.csv`
* Plots the raw structure of the dataset
* Applies hierarchical clustering
* Saves results to `demo_output_agglomerative/`

---

## 📁 Navigating the Package

Here is a simple explanation of what each major module does:

* **dataframe_builder.py** — Create seed cluster structures and simulate data
* **preprocessing.py** — Feature selection and standardisation
* **algorithms.py** — Manual K-means implementation + sklearn wrapper
* **agglomerative.py** — Agglomerative clustering
* **evaluation.py** — Inertia, silhouette scores, elbow curve
* **plotting_clustered.py** — 2D cluster plots and elbow plots
* **interface.py** — High-level `run_clustering` function
* **demo/** — All runnable scripts
* **tests/** — Automated validation tests

---

## 🧭 Tips for Users

* All scripts print clear, minimal information to avoid confusion.
* Output directories are created automatically — no manual setup needed.
* If a script requires an input CSV, an error message will appear if the file is missing.
* Numeric data is checked automatically; non-numeric columns are rejected.
* You will never need to understand the internal file structure to run the demos.

---


