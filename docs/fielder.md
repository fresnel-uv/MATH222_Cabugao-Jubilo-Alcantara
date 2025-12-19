# Fiedler module — Quick Guide

This document gives the essentials to use the `FiedlerAnalysis` and `LaplacianEmbeddings` classes from `core/fielder.py`.

## Purpose
- Compute Fiedler (second eigen) vectors for three Laplacian variants (standard, symmetric, random-walk).
- Save per-graph Fiedler vectors and a summary CSV; produce simple 1D/2D embeddings and plots.

## Quick start

Minimal example (run after you have `laplacian_results` and `graph_results` available):

```python
from core.fielder import FiedlerAnalysis, LaplacianEmbeddings

# Run Fiedler analysis (saves .npy files and summary CSV)
fa = FiedlerAnalysis(save_dir="outputs/fiedler", show_info=True, show_plots=False)
fied_results, df_fiedler = fa.analyze_all_laplacians(laplacian_results, graph_results, name_prefix="fiedler_demo")

# Compute and optionally plot embeddings for each label
le = LaplacianEmbeddings(show_plots=True)
for label, lapgroup in fied_results.items():
    embeddings = le.compute_and_plot(lapgroup)
```

## Inputs
- `laplacian_results` (dict): mapping label -> dict with keys `L`, `L_sym`, `L_rw` (numpy arrays)
- `graph_results` (dict): mapping label -> dict containing `"G"` (a networkx Graph) so node/edge counts can be reported

Label format expected by the module: e.g. `"cosine__knn_k2"` or `"jaccard_binary__threshold_t0.4"`.

## Outputs
- Return from `FiedlerAnalysis.analyze_all_laplacians`:
  - `final_output`: dict[label] -> {"Standard","Symmetric","RandomWalk"} each with keys:
    - `L`: Laplacian matrix
    - `fiedler`: numpy array of Fiedler vector
    - `labels_2way`: median-thresholded 0/1 partition
  - `df`: pandas DataFrame summary indexed by (sim_metric, method, params_str)
- Files saved under `save_dir` (default `outputs/fiedler/`):
  - `outputs/fiedler/{name_prefix}_{label}/fiedler_standard.npy` (and `_symmetric.npy`, `_randomwalk.npy`)
  - `outputs/fiedler/{name_prefix}_summary.csv`

`LaplacianEmbeddings.compute_and_plot` returns an `embeddings` dict: `{1: fiedler_1d, 2: emb2d}` and (optionally) shows plots.

## Implementation notes
- Fiedler vectors are computed via `utils.fiedler_vector(L)` from `utils/utils_spectral.py`.
- 2D embeddings are produced by building `A_sim = I - L` and using `sklearn.manifold.SpectralEmbedding(affinity='precomputed')`.

## Tips & troubleshooting
- Disconnected graphs: check connectivity (`nx.is_connected(G)`) — multiple zero eigenvalues can affect λ₂.
- Large graphs: `SpectralEmbedding` can be slow and memory-heavy; consider subsampling or approximate methods.
- If embeddings look degenerate, inspect `L` values — `A_sim = I - L` assumes L is a valid Laplacian in expected ranges.
- To override median split, compute your own partition from the returned `fiedler` vector.

## Dependencies
- numpy, pandas, matplotlib, scikit-learn, networkx, and the project's `utils` module.

## Where to find saved outputs
- Fiedler vectors and summary CSV: `outputs/fiedler/`
- Embedding output directory created by `LaplacianEmbeddings`: `outputs/embeddings/` (embeddings are returned in-memory by default)

---
Created to be brief and copy/paste friendly. For an example runner script or unit-test scaffold, open an issue or request in the repo.
