# Graph Analysis Notebook — Overview

This notebook demonstrates a complete workflow for generating, analyzing, and evaluating graphs created from synthetic block-structured data. It is organized to be reproducible and modular, with separate components for data preparation, graph construction, spectral analysis, and quantitative evaluation.

## Purpose
- Create synthetic voting-like data with known block structure.
- Build graphs using multiple similarity metrics and graph construction methods (k-NN, threshold).
- Compare graph topologies via a grid search over parameters.
- Compute Laplacians, Fiedler vectors, and spectral embeddings to study cluster structure.
- Evaluate clustering quality with multiple graph-level metrics.

## High-level workflow
1. Data preparation
    - Use DataPreparer to synthesize a dataset with multiple blocks and samples.
2. Graph generation and grid search
    - Use GraphGenerator to build graphs for combinations of similarity metrics and parameters.
    - Collect adjacency matrices, similarity matrices, and graph objects in `results`.
    - Summarize graph-level statistics in `summary_df`.
3. Laplacian analysis
    - Generate Laplacian variants for each graph and save results in `lap_outputs`.
4. Fiedler & embeddings
    - Run Fiedler analysis and 1D/2D Laplacian embeddings to visualize separation and ordering.
5. Spectral embedding and clustering
    - Compute eigenvectors, eigenvalues, and cluster labels using SpectralEmbeddingAnalysis.
6. Evaluation
    - Use GraphClusteringEvaluator to compute spectral gap, silhouette, modularity, conductance, etc., and aggregate results in `df_metrics`.

## Key variables (present in the notebook)
- `data` (dict): synthetic dataset with keys `matrix`, `meta`, `nodes`. Example shape: (180, 60).
- `dp` (DataPreparer): configured to produce the synthetic dataset.
- `gg` (GraphGenerator): configured for similarity/graph methods and grid search.
- `results` (dict): per-configuration outputs containing matrices (`S`, `A`, `D`), networkx `G`, and computed metrics.
- `summary_df` (DataFrame): tabular summary of graph statistics for all grid-search configurations.

## How to use this notebook
- Execute cells top-to-bottom to reproduce the workflow. Most objects produced by earlier cells are reused later (e.g., `results`, `lap_outputs`, `df_explore`).
- Toggle plotting flags (e.g., `show_plots`) on generators/analyses to inspect intermediate visualizations.
- Adjust grid-search parameters in the GraphGenerator call to explore alternative graph constructions.

## Outputs & artifacts
- Summary tables (`summary_df`, `df_explore`, `df_metrics`) for quick comparison of graph variants.
- Saved laplacian matrices, Fiedler outputs, and embedding plots (configurable save directories).
- Diagnostic plots for cluster structure and embedding quality.

## Next steps / suggestions
- Inspect `summary_df` to pick promising graph configurations for deeper analysis.
- Compare evaluation metrics across similarity metrics and graph methods to select the most robust construction.
- Extend the synthetic scenarios (vary block counts/sizes or add noise) to test method sensitivity.