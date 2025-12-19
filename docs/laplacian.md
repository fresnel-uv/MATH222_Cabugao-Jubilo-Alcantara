# GraphLaplacianGenerator Documentation

## Overview

`GraphLaplacianGenerator` is a Python class for computing graph Laplacians from similarity or adjacency matrices. It supports **standard**, **symmetric normalized**, and **random-walk normalized** Laplacians. The class also provides plotting of eigenvalue spectra and saving outputs to disk.

Outputs are automatically stored under `outputs/laplacians/<name>/`.

---

## Installation / Setup

Ensure the following packages are installed:

```bash
pip install numpy matplotlib networkx
```

Make sure `utils_spectral` is available as it provides utility functions:

```python
from utils import utils_spectral as utils
```

---

## Class: `GraphLaplacianGenerator`

### Constructor

```python
GraphLaplacianGenerator(
    sim_metric="cosine",
    graph_method="knn",
    k=10,
    threshold=0.5,
    show_info=True,
    show_plots=True,
    save_dir=OUTPUT_ROOT
)
```

**Parameters:**

| Parameter      | Type  | Default     | Description                                         |
| -------------- | ----- | ----------- | --------------------------------------------------- |
| `sim_metric`   | str   | "cosine"    | Similarity metric: "cosine" or "correlation"        |
| `graph_method` | str   | "knn"       | Graph construction method: "knn" or "threshold"     |
| `k`            | int   | 10          | Number of neighbors for `knn`                       |
| `threshold`    | float | 0.5         | Threshold for adjacency if using "threshold" method |
| `show_info`    | bool  | True        | Print Laplacian and degree summaries                |
| `show_plots`   | bool  | True        | Plot eigenvalue spectra                             |
| `save_dir`     | Path  | OUTPUT_ROOT | Directory to save outputs                           |

---

### Method: `generate`

```python
S, A, (D, L, L_sym, L_rw) = generate(X=None, A=None, name="laplacian_run")
```

**Description:**

* Computes similarity, adjacency, degree, and Laplacians.
* Supports synthetic fallback if no `X` or `A` is provided.
* Saves outputs to disk under `outputs/laplacians/<name>/`.

**Parameters:**

| Parameter | Type       | Default         | Description                             |
| --------- | ---------- | --------------- | --------------------------------------- |
| `X`       | np.ndarray | None            | Data matrix to compute similarity from  |
| `A`       | np.ndarray | None            | Adjacency matrix (overrides similarity) |
| `name`    | str        | "laplacian_run" | Subfolder name for saving outputs       |

**Returns:**

* `S` : similarity matrix
* `A` : adjacency matrix
* `(D, L, L_sym, L_rw)` : degree matrix and three Laplacians (standard, symmetric normalized, random-walk normalized)

---

### Example Usage

```python
from core.laplacian import GraphLaplacianGenerator

# Instantiate generator
gl = GraphLaplacianGenerator(sim_metric="cosine", graph_method="knn")

# Compute Laplacians from synthetic data
S, A, (D, L, L_sym, L_rw) = gl.generate()
```

### Notes

* `show_info=True` prints shapes and minimum eigenvalue of `L`.
* `show_plots=True` plots eigenvalue spectra of the three Laplacians.
* Files saved include `similarity.npy`, `adjacency.npy`, `degree.npy`, `L.npy`, `L_sym.npy`, `L_rw.npy`.
* Compatible with both Jupyter notebooks and script execution.
