import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import networkx as nx
import utils.utils_spectral as utils

# Default outputs folder
OUTPUT_ROOT = Path.cwd() / "outputs" / "spectral_embedding"
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


class SpectralEmbeddingAnalysis:
    """
    Computes spectral embeddings using L, L_sym, L_rw, clusters them,
    plots embeddings side-by-side with the original graph colored by cluster labels,
    and saves outputs.

    Compatible with:
        - LaplacianGenerator outputs (L, L_sym, L_rw)
        - GraphGenerator (producing A)
        - Raw data X (from which adjacency + Laplacians are built)
    """

    def __init__(
        self,
        sim_metric: str = "cosine",
        graph_method: str = "knn",
        k_neighbors: int = 10,
        show_info: bool = True,
        show_plots: bool = True,
        save_dir: Path = OUTPUT_ROOT,
    ):
        self.sim_metric = sim_metric
        self.graph_method = graph_method
        self.k_neighbors = k_neighbors
        self.show_info = show_info
        self.show_plots = show_plots
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # MAIN PIPELINE
    # --------------------------------------------------------
    def run(
        self,
        X=None,
        A=None,
        L=None,
        L_sym=None,
        L_rw=None,
        k: int = 2,
        skip_first: bool = True,
        seed: int = 123,
        name: str = "spectral_run",
    ):
        """
        Flexible spectral embedding pipeline.

        Priority:
            1. Laplacians supplied → use directly
            2. Adjacency A supplied → compute Laplacians
            3. Raw data X → compute similarity → adjacency → Laplacians

        Returns dictionary of embeddings, eigenvalues, and clustering labels.
        """
        # 1. Use supplied Laplacians
        if L is not None or L_sym is not None or L_rw is not None:
            lap_dict = {kname: kmat for kname, kmat in {"L": L, "L_sym": L_sym, "L_rw": L_rw}.items() if kmat is not None}
            S = None
        # 2. Use adjacency
        elif A is not None:
            S = None
            D, L, L_sym, L_rw = utils.laplacians(A)
            lap_dict = {"L": L, "L_sym": L_sym, "L_rw": L_rw}
        # 3. Full pipeline
        else:
            if X is None:
                X, _ = utils.make_synthetic_performance()
            S = self._compute_similarity(X)
            A = utils.adjacency_from_similarity(S, method=self.graph_method, k=self.k_neighbors)
            D, L, L_sym, L_rw = utils.laplacians(A)
            lap_dict = {"L": L, "L_sym": L_sym, "L_rw": L_rw}

        # Prepare output dictionary
        results = {"S": S, "A": A, "D": D}

        # Build networkx graph for plotting
        if A is not None:
            G = nx.from_numpy_array(A)
            pos = nx.spring_layout(G, seed=seed)
        else:
            G = pos = None

        # Loop over Laplacian types
        for lap_name, lap_matrix in lap_dict.items():
            U, vals = utils.spectral_embedding(lap_matrix, k=k, skip_first=skip_first)
            labels = utils.kmeans_cluster(U, k=k, seed=seed)

            results[lap_name] = {"U": U, "vals": vals, "labels": labels}

            if self.show_info:
                self._print_info(lap_name, U, vals, labels)

            if self.show_plots:
                self._plot_graph_vs_embedding(G, pos, U, labels, lap_name)

            self._save_outputs(lap_name, S, A, lap_matrix, U, vals, labels, name)

        return results

    # --------------------------------------------------------
    # HELPERS
    # --------------------------------------------------------
    def _compute_similarity(self, X):
        if self.sim_metric == "cosine":
            return utils.cosine_similarity(X)
        elif self.sim_metric == "correlation":
            return utils.correlation_similarity(X)
        elif self.sim_metric == "jaccard_binary":
            Xbin = (X > X.mean()).astype(int)
            return utils.jaccard_similarity_binary(Xbin)
        else:
            raise ValueError(f"Unknown similarity metric: {self.sim_metric}")

    def _print_info(self, lap_name, U, vals, labels):
        print(f"[SPECTRAL EMBEDDING] Laplacian: {lap_name}")
        print("Embedding shape:", U.shape)
        print("Eigenvalues (first 10):", np.round(vals[:10], 4))
        print("Cluster counts:", np.bincount(labels))

    def _plot_graph_vs_embedding(self, G, pos, U, labels, lap_name):
        """Plot original graph with nodes colored by cluster labels vs spectral embedding"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Original graph with cluster colors
        if G is not None and pos is not None:
            nx.draw(G, pos=pos, ax=axes[0], node_color=labels, cmap='tab10',
                    node_size=60, edge_color='gray', alpha=0.7)
            axes[0].set_title("Graph Colored by Cluster Labels")
            axes[0].axis("off")
        else:
            axes[0].text(0.5, 0.5, 'No Graph Available', ha='center', va='center')
            axes[0].axis('off')

        # Spectral embedding
        if U.shape[1] >= 2:
            axes[1].scatter(U[:, 0], U[:, 1], c=labels, cmap='tab10', s=60)
        else:
            axes[1].scatter(range(len(labels)), U[:, 0], c=labels, cmap='tab10', s=60)
        axes[1].set_title(f"Spectral Embedding ({lap_name})")
        axes[1].grid(True)

        plt.tight_layout()
        plt.show()

    def _save_outputs(self, lap_name, S, A, L, U, vals, labels, run_name):
        out_dir = self.save_dir / run_name / lap_name
        out_dir.mkdir(parents=True, exist_ok=True)

        np.save(out_dir / "similarity.npy", S)
        np.save(out_dir / "adjacency.npy", A)
        np.save(out_dir / f"{lap_name}.npy", L)
        np.save(out_dir / "embedding.npy", U)
        np.save(out_dir / "eigenvalues.npy", vals)
        np.save(out_dir / "labels.npy", labels)

        print(f"[INFO] Saved outputs for {lap_name} → {out_dir.resolve()}")

    def plot_spectral_gap_all(lap_outputs):
        """
        Plot adjacency spectral gap and Laplacian spectral gaps
        for all similarity metrics and methods.
        
        Parameters
        ----------
        lap_outputs : dict
            Keys like 'cosine__knn_k5', 'correlation__threshold_t0.3', etc.
            Each value must contain 'A', 'L', 'L_sym', 'L_rw'.
        """
        lap_types = ['Adjacency', 'L', 'L_sym', 'L_rw']
        sim_metrics = ['cosine', 'correlation', 'jaccard_binary']
        methods = ['knn', 'threshold']

        for sim in sim_metrics:
            for method in methods:
                # Filter relevant labels
                relevant_labels = [lbl for lbl in lap_outputs.keys() if lbl.startswith(f"{sim}__{method}")]
                if not relevant_labels:
                    continue

                param_values = []
                gaps = {lt: [] for lt in lap_types}

                for lbl in relevant_labels:
                    # Extract parameter value
                    param = np.nan
                    if '_k' in lbl:
                        try:
                            param = float(''.join(filter(lambda x: x.isdigit() or x=='.', lbl.split('_k')[1])))
                        except:
                            pass
                    elif '_t' in lbl:
                        try:
                            param = float(''.join(filter(lambda x: x.isdigit() or x=='.', lbl.split('_t')[1])))
                        except:
                            pass
                    param_values.append(param)

                    # Adjacency spectral gap: λ_max - λ_2
                    A = lap_outputs[lbl]['A']
                    eigs_A = np.sort(np.linalg.eigvalsh(A))[::-1]
                    gaps['Adjacency'].append(eigs_A[0] - eigs_A[1] if len(eigs_A) >= 2 else np.nan)

                    # Laplacian spectral gaps: λ₂ - λ₁
                    for lt in ['L', 'L_sym', 'L_rw']:
                        Lmat = lap_outputs[lbl][lt]
                        eigs_L = np.sort(np.linalg.eigvalsh(Lmat))
                        gaps[lt].append(eigs_L[1] - eigs_L[0] if len(eigs_L) >= 2 else np.nan)

                # Sort by parameter
                sorted_idx = np.argsort(param_values)
                param_sorted = np.array(param_values)[sorted_idx]
                gaps_sorted = {lt: np.array(gaps[lt])[sorted_idx] for lt in lap_types}

                # Plot all gaps in a single figure
                plt.figure(figsize=(8, 5))
                for lt in lap_types:
                    plt.plot(param_sorted, gaps_sorted[lt], marker='o', linestyle='-', label=lt)
                plt.title(f"Spectral Gaps ({sim} — {method})")
                plt.xlabel("Parameter (k or ε)")
                plt.ylabel("Spectral Gap")
                plt.grid(True)
                plt.legend()
                plt.tight_layout()
                plt.show()

