# GraphGenerator

`GraphGenerator` is a utility class for generating graphs from data matrices. It can compute similarity matrices, adjacency matrices, degree matrices, and visualize graphs. It also supports saving outputs, including matrices, graphs, and metadata.

---

## Table of Contents

1. [Installation](#installation)
2. [Class Overview](#class-overview)
3. [Constructor Parameters](#constructor-parameters)
4. [Main Methods](#main-methods)

   * [`generate_graph`](#generate_graph)
5. [Helper Methods](#helper-methods)
6. [Saving Outputs](#saving-outputs)
7. [Examples](#examples)

---

## Installation

Ensure you have the following dependencies installed:

```bash
pip install numpy matplotlib networkx
```

You also need a `utils_spectral` module that provides similarity functions:

* `cosine_similarity`
* `correlation_similarity`
* `jaccard_similarity_binary`
* `adjacency_from_similarity`
* `degree_matrix`

---

## Class Overview

```python
class GraphGenerator:
    def __init__(...)
    def generate_graph(self, data: Dict[str, np.ndarray], dataset_name: str = "run") -> Dict[str, np.ndarray]:
    ...
```

`GraphGenerator` provides a pipeline to:

1. Compute similarity from data.
2. Build adjacency and degree matrices.
3. Convert adjacency to a NetworkX graph.
4. Optionally visualize the graph and matrices.
5. Save all outputs in a structured directory.

---

## Constructor Parameters

| Parameter      | Type    | Default       | Description                                                                  |
| -------------- | ------- | ------------- | ---------------------------------------------------------------------------- |
| `sim_metric`   | `str`   | `"cosine"`    | Similarity metric (`"cosine"`, `"correlation"`, `"jaccard_binary"`).         |
| `graph_method` | `str`   | `"threshold"` | Graph construction method (`"threshold"` or `"knn"`).                        |
| `threshold`    | `float` | `0.5`         | Threshold for adjacency if using `threshold` method.                         |
| `k`            | `int`   | `10`          | Number of neighbors for `knn` method.                                        |
| `show_info`    | `bool`  | `True`        | Print graph summary information.                                             |
| `show_plot`    | `bool`  | `True`        | Show the graph plot.                                                         |
| `show_heatmap` | `bool`  | `False`       | Show heatmaps for similarity and adjacency matrices.                         |
| `save_dir`     | `str`   | `None`        | Directory path to save outputs. Creates structured subfolders automatically. |

---

## Main Methods

### `generate_graph`

```python
def generate_graph(self, data: Dict[str, np.ndarray], dataset_name: str = "run") -> Dict[str, np.ndarray]:
```

Generates similarity, adjacency, and degree matrices from input data and returns a dictionary with:

* `'S'`: similarity matrix
* `'A'`: adjacency matrix
* `'D'`: degree matrix
* `'G'`: NetworkX graph object

**Parameters:**

* `data`: dictionary with keys `'matrix'`, `'X'`, or `'A'`.
* `dataset_name`: name used for saving outputs.

**Returns:**

* Dictionary of computed matrices and graph object.

---

## Helper Methods

* `_compute_similarity(X)`: Computes similarity matrix according to `sim_metric`.
* `_summarize_graph(G, A, D)`: Prints summary statistics of the graph.
* `_visualize_graph(G)`: Plots the NetworkX graph.
* `_plot_heatmap(M, title)`: Plots a heatmap of a matrix.
* `_save_outputs(result, dataset_name)`: Saves matrices, graph, and metadata to disk.

---

## Saving Outputs

If `save_dir` is provided, `generate_graph` automatically saves:

* `similarity.npy`
* `adjacency.npy`
* `degree.npy`
* `graph.graphml`
* `metadata.json`

All outputs are saved under:

```
<save_dir>/<dataset_name>/run_<timestamp>/
```

Where `<timestamp>` is in `YYYYMMDD_HHMMSS` format.

---

## Example Usage

```python
from core.graph_generation import GraphGenerator

gg = GraphGenerator(sim_metric="cosine", graph_method="threshold", threshold=0.5, save_dir="outputs")
data = {'matrix': np.random.rand(60, 10)}
result = gg.generate_graph(data, dataset_name="synthetic_test")

# Access matrices
S = result['S']
A = result['A']
D = result['D']
G = result['G']
```

This will also save all outputs and optionally plot the graph and heatmaps.
