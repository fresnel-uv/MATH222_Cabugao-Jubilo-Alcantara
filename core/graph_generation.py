import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple, Any
from datetime import datetime
import json
import pandas as pd
from utils import utils_spectral as utils

class GraphGenerator:
    """
    GraphGenerator:
    - generate_graph(...) -> single graph build
    - run_grid_search(...) -> iterate over similarity metrics, methods, parameter lists,
                               compute minimal core metrics, optionally return DataFrame, save outputs, and plot grouped subplots.
    """

    def __init__(
        self,
        sim_metric: str = "cosine",
        graph_method: str = "threshold",
        threshold: float = 0.5,
        k: int = 10,
        show_info: bool = True,
        show_plot: bool = True,
        show_heatmap: bool = False,
        save_dir: Optional[str] = None,
    ):
        self.sim_metric = sim_metric
        self.graph_method = graph_method
        self.threshold = threshold
        self.k = k
        self.show_info = show_info
        self.show_plot = show_plot
        self.show_heatmap = show_heatmap
        self.save_dir = Path(save_dir) if save_dir else None

        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------
    # Main Interface
    # ---------------------------------------------------
    def generate_graph(self, data: Dict[str, np.ndarray], dataset_name: str = "run") -> Dict[str, np.ndarray]:
        """
        Generates similarity (S), adjacency (A), degree (D) matrices and networkx graph (G).
        Accepts DataPreparer output or raw dict containing 'matrix'/'X'/'A'.
        """
        # Adapt DataPreparer output
        if "matrix" in data:
            X = data["matrix"]
            S = self._compute_similarity(X) if self.graph_method in ["threshold", "knn"] else X.copy()
        elif "X" in data:
            X = data["X"]
            S = self._compute_similarity(X)
        elif "A" in data:
            S = data["A"].copy()
        else:
            raise ValueError("Data must contain 'matrix', 'X', or 'A'.")

        # Build adjacency and degree matrices
        A = utils.adjacency_from_similarity(S, method=self.graph_method, threshold=self.threshold, k=self.k)
        D = utils.degree_matrix(A)
        G = nx.from_numpy_array(A)

        result = {"S": S, "A": A, "D": D, "G": G}

        # Optional visualization / info
        if self.show_info:
            self._summarize_graph(G, A, D)
        if self.show_heatmap:
            self._plot_heatmap(S, title="Similarity Matrix (S)")
            self._plot_heatmap(A, title="Adjacency Matrix (A)")
        if self.show_plot:
            self._visualize_graph(G)

        if self.save_dir:
            self._save_outputs(result, dataset_name=dataset_name)

        return result

    # ---------------------------------------------------
    # Grid search with optional DataFrame
    # ---------------------------------------------------
    def run_grid_search(
        self,
        data: Dict[str, np.ndarray],
        similarity_metrics: Sequence[str] = ("cosine", "correlation", "jaccard_binary"),
        methods: Sequence[str] = ("knn", "threshold"),
        k_values: Sequence[int] = (5, 10, 15),
        threshold_values: Sequence[float] = (0.2, 0.3, 0.4, 0.5),
        dataset_name: str = "grid_search",
        show_subplots: bool = True,
        return_dataframe: bool = True
    ) -> Any:
        """
        Full grid search across similarity metrics, graph methods, and parameters.
        Returns either:
          - results dict only
          - results dict + pandas DataFrame (if return_dataframe=True)
        """
        results: Dict[str, Dict[str, Any]] = {}
        summary_rows = []

        for sim in similarity_metrics:
            for method in methods:
                param_iter = [(('k', k),) for k in k_values] if method == "knn" else [(('t', t),) for t in threshold_values]

                for params in param_iter:
                    if method == "knn":
                        k = params[0][1]
                        label = f"{sim}__knn_k{k}"
                        gg = GraphGenerator(sim_metric=sim, graph_method="knn", k=k, threshold=self.threshold,
                                            show_info=False, show_plot=False, show_heatmap=False, save_dir=self.save_dir)
                    else:
                        t = params[0][1]
                        label = f"{sim}__threshold_t{t}"
                        gg = GraphGenerator(sim_metric=sim, graph_method="threshold", threshold=t, k=self.k,
                                            show_info=False, show_plot=False, show_heatmap=False, save_dir=self.save_dir)

                    print(f"\n=== Running grid item: {label} ===")
                    saved_dataset_name = f"{dataset_name}/{label}"
                    res = gg.generate_graph(data, dataset_name=saved_dataset_name)
                    metrics = self._compute_minimal_core_metrics(res["G"], res["A"])
                    res["metrics"] = metrics
                    results[label] = res

                    row = {"label": label, "sim_metric": sim, "method": method, "params": {p[0]: p[1] for p in params}, **metrics}
                    summary_rows.append(row)

        summary_df = pd.DataFrame(summary_rows) if return_dataframe else None
        if return_dataframe and not summary_df.empty:
            summary_df["params_str"] = summary_df["params"].apply(lambda x: json.dumps(x) if isinstance(x, dict) else str(x))
            summary_df.set_index(["sim_metric", "method", "params_str"], inplace=True)
            if self.save_dir:
                csv_path = self.save_dir / f"{dataset_name}_summary.csv"
                summary_df.to_csv(csv_path)
                print(f"[INFO] Summary dataframe saved to {csv_path.resolve()}")

        if show_subplots:
            self._plot_all_grid_results(results)

        return (results, summary_df) if return_dataframe else results

    # ---------------------------------------------------
    # Minimal Core Metrics
    # ---------------------------------------------------
    def _compute_minimal_core_metrics(self, G: nx.Graph, A: np.ndarray) -> Dict[str, Any]:
        metrics: Dict[str, Any] = {}
        n_components = nx.number_connected_components(G) if G.number_of_nodes() > 0 else 0
        metrics["nodes"] = G.number_of_nodes()
        metrics["edges"] = G.number_of_edges()
        metrics["number_connected_components"] = int(n_components)
        metrics["size_giant_component"] = int(max(len(c) for c in nx.connected_components(G)) if n_components > 0 else 0)
        metrics["density"] = float(nx.density(G))
        degrees = np.array([d for _, d in G.degree()], dtype=float)
        metrics["avg_degree"] = float(np.mean(degrees)) if degrees.size > 0 else 0.0
        metrics["max_degree"] = float(np.max(degrees)) if degrees.size > 0 else 0.0
        if degrees.size > 0 and degrees.sum() > 0:
            uniq, counts = np.unique(degrees, return_counts=True)
            probs = counts.astype(float) / counts.sum()
            metrics["degree_entropy"] = float(-np.sum(probs * np.log2(probs + 1e-12)))
        else:
            metrics["degree_entropy"] = 0.0
        metrics["global_clustering"] = float(nx.transitivity(G))
        metrics["avg_local_clustering"] = float(nx.average_clustering(G) if G.number_of_nodes() > 0 else 0.0)
        if metrics["size_giant_component"] <= 1:
            diam = 0
        else:
            largest_cc_nodes = max(nx.connected_components(G), key=len)
            subG = G.subgraph(largest_cc_nodes)
            try:
                diam = nx.diameter(subG)
            except Exception:
                try:
                    diam = max(nx.eccentricity(subG).values())
                except Exception:
                    diam = float('nan')
        metrics["diameter"] = int(diam) if isinstance(diam, (int, np.integer)) else diam
        return metrics

    # ---------------------------------------------------
    # Plotting & Visualization Helpers (unchanged)
    # ---------------------------------------------------
    def _plot_all_grid_results(self, results: Dict[str, Dict[str, Any]]):
        grouped = {}
        for label, res in results.items():
            sim, rest = label.split('__', 1) if '__' in label else ('unknown', label)
            method = 'knn' if rest.startswith('knn_') else ('threshold' if rest.startswith('threshold_') else 'other')
            grouped.setdefault(sim, {}).setdefault(method, []).append((label, res))
        for sim, methods_dict in grouped.items():
            for method_name, items in methods_dict.items():
                if items:
                    title = f"{sim} — {method_name.upper()} Graphs"
                    self._plot_category_subplots(items, title)

    def _plot_category_subplots(self, items: Sequence[Tuple[str, Dict[str, Any]]], title: str, max_cols: int = 3):
        n = len(items)
        cols = min(n, max_cols)
        rows = (n + cols - 1) // cols
        plt.figure(figsize=(6 * cols, 5 * rows))
        for i, (label, res) in enumerate(items, start=1):
            G = res['G']
            pos = nx.spring_layout(G, seed=42)
            plt.subplot(rows, cols, i)
            nx.draw_networkx_nodes(G, pos, node_size=40, node_color='skyblue', alpha=0.8)
            nx.draw_networkx_edges(G, pos, width=0.3, alpha=0.5)
            plt.title(label)
            plt.axis('off')
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        plt.show()

    def _compute_similarity(self, X: np.ndarray) -> np.ndarray:
        if self.sim_metric == 'cosine':
            return utils.cosine_similarity(X)
        elif self.sim_metric == 'correlation':
            return utils.correlation_similarity(X)
        elif self.sim_metric == 'jaccard_binary':
            Xbin = (X > X.mean()).astype(int)
            return utils.jaccard_similarity_binary(Xbin)
        else:
            raise ValueError(f"Unsupported similarity metric: {self.sim_metric}")

    def _summarize_graph(self, G: nx.Graph, A: np.ndarray, D: np.ndarray):
        n, m = G.number_of_nodes(), G.number_of_edges()
        density = nx.density(G)
        degrees = np.sum(A, axis=1)
        print(f"\n[GRAPH SUMMARY] Nodes: {n}, Edges: {m}, Density: {density:.4f}, Avg Degree: {degrees.mean():.2f}, Max Degree: {degrees.max():.0f}")

    def _visualize_graph(self, G: nx.Graph):
        plt.figure(figsize=(6,5))
        pos = nx.spring_layout(G, seed=42)
        nx.draw_networkx_nodes(G, pos, node_size=60, node_color='skyblue', alpha=0.7)
        nx.draw_networkx_edges(G, pos, width=0.5, alpha=0.5)
        plt.title('Graph Visualization')
        plt.axis('off')
        plt.show()

    def _plot_heatmap(self, M: np.ndarray, title: str):
        plt.figure(figsize=(5,4))
        plt.imshow(M, cmap='viridis', interpolation='nearest')
        plt.title(title)
        plt.colorbar(label='Value')
        plt.tight_layout()
        plt.show()

    def _save_outputs(self, result: Dict[str, np.ndarray], dataset_name: str):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_path = self.save_dir / dataset_name / f'run_{timestamp}'
        save_path.mkdir(parents=True, exist_ok=True)
        np.save(save_path / 'similarity.npy', result['S'])
        np.save(save_path / 'adjacency.npy', result['A'])
        np.save(save_path / 'degree.npy', result['D'])
        nx.write_graphml(result['G'], save_path / 'graph.graphml')
        metadata = {'dataset': dataset_name, 'graph_method': self.graph_method, 'similarity_metric': self.sim_metric, 'threshold': self.threshold, 'k': self.k}
        with open(save_path / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"\n[INFO] Saved graph data and metadata to {save_path.resolve()}")

    # ---------------------------------------------------
    # Plot parameter–metric curves for all sim metrics & methods
    # ---------------------------------------------------
    def plot_parameter_metrics(self, summary_df):
        """
        Produces line plots of graph metrics vs. graph parameters (k or threshold),
        for ALL combinations of (similarity metric × graph construction method).
        
        Expects the summary_df produced by run_grid_search().
        """

        import matplotlib.pyplot as plt
        import json

        if summary_df is None or summary_df.empty:
            print("[WARN] Empty summary_df. Nothing to plot.")
            return

        # These are the metrics computed in _compute_minimal_core_metrics()
        metric_names = [
            "nodes",
            "edges",
            "density",
            "avg_degree",
            "number_connected_components",
            "size_giant_component",
            "degree_entropy",
            "global_clustering",
            "diameter",
        ]

        # The MultiIndex has (sim_metric, method, params_str)
        sim_metrics = summary_df.index.get_level_values(0).unique()
        methods = summary_df.index.get_level_values(1).unique()

        for sim in sim_metrics:
            for method in methods:

                # Try to extract the subset for this combination
                try:
                    df_sub = summary_df.loc[sim, method]
                except KeyError:
                    # This (sim, method) pair does not exist
                    continue

                if df_sub.empty:
                    continue

                # Extract parameters from params_str (JSON inside string)
                params = []
                for p in df_sub.index:
                    p_dict = json.loads(p)
                    # KNN uses 'k', threshold uses 't'
                    if "k" in p_dict:
                        params.append(p_dict["k"])
                    elif "t" in p_dict:
                        params.append(p_dict["t"])
                    else:
                        params.append(None)

                df_plot = df_sub.copy()
                df_plot["param"] = params

                # Drop rows with None parameter
                df_plot = df_plot.dropna(subset=["param"])

                # Sort by parameter for clean plotting
                df_plot = df_plot.sort_values(by="param")

                if df_plot.empty:
                    continue

                # Create the figure
                n_metrics = len(metric_names)
                cols = 3
                rows = (n_metrics + cols - 1) // cols

                fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows))
                axes = axes.ravel()

                for i, metric in enumerate(metric_names):
                    ax = axes[i]
                    try:
                        ax.plot(df_plot["param"], df_plot[metric], marker="o", linewidth=2)
                        ax.set_title(metric.replace("_", " ").title(), fontsize=12)
                        ax.set_xlabel("k" if method == "knn" else "threshold")
                        ax.set_ylabel(metric)
                        ax.grid(True, alpha=0.3)
                    except KeyError:
                        ax.set_title(f"{metric} not found", fontsize=12)

                # Hide unused subplots if any
                for j in range(i + 1, len(axes)):
                    axes[j].axis("off")

                fig.suptitle(
                    f"{sim.upper()} — {method.upper()} Graph Metrics",
                    fontsize=18,
                    y=1.02
                )
                fig.tight_layout()

                # Save the figure if save_dir is set
                if self.save_dir:
                    out_dir = self.save_dir / "parameter_metric_plots"
                    out_dir.mkdir(parents=True, exist_ok=True)
                    fname = f"{sim}_{method}_metric_curves.png".replace("/", "_")
                    fig.savefig(out_dir / fname, dpi=200, bbox_inches="tight")
                    print(f"[INFO] Saved figure: {out_dir / fname}")

                plt.show()