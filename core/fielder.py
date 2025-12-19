import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import SpectralEmbedding
from pathlib import Path
from utils import utils_spectral as utils

class FiedlerAnalysis:

    def __init__(self, show_info=True, show_plots=True, save_dir: Path = None):
        self.show_info = show_info
        self.show_plots = show_plots

        if save_dir is None:
            try:
                PACKAGE_ROOT = Path(__file__).resolve().parents[1]
            except NameError:
                PACKAGE_ROOT = Path.cwd().resolve()
            save_dir = PACKAGE_ROOT / "outputs" / "fiedler"

        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def analyze_all_laplacians(
        self,
        laplacian_results: dict,
        graph_results: dict,
        name_prefix="fiedler"
    ):
        rows = []
        final_output = {}

        for label, mats in laplacian_results.items():
            sim_metric, rest = label.split("__", 1)
            if rest.startswith("knn"):
                method = "knn"
                k = int(rest.split("_k")[1])
                params = {"k": k}
                params_str = f'{{"k": {k}}}'
            else:
                method = "threshold"
                t = float(rest.split("_t")[1])
                params = {"t": t}
                params_str = f'{{"t": {t}}}'

            G = graph_results[label]["G"]
            nodes = G.number_of_nodes()
            edges = G.number_of_edges()

            L_std = mats["L"]
            L_sym = mats["L_sym"]
            L_rw = mats["L_rw"]

            lam2_std, f2_std = utils.fiedler_vector(L_std)
            lam2_sym, f2_sym = utils.fiedler_vector(L_sym)
            lam2_rw,  f2_rw  = utils.fiedler_vector(L_rw)

            cut_std = self._compute_cut_balance(f2_std)
            cut_sym = self._compute_cut_balance(f2_sym)
            cut_rw  = self._compute_cut_balance(f2_rw)

            out_dir = self.save_dir / f"{name_prefix}_{label}"
            out_dir.mkdir(parents=True, exist_ok=True)

            np.save(out_dir / "fiedler_standard.npy", f2_std)
            np.save(out_dir / "fiedler_symmetric.npy", f2_sym)
            np.save(out_dir / "fiedler_randomwalk.npy", f2_rw)

            if self.show_plots:
                self._subplot_fiedlers(label, f2_std, f2_sym, f2_rw)

            if self.show_info:
                print(f"\n[{label}]")
                print("λ₂ standard:    ", lam2_std)
                print("λ₂ symmetric:   ", lam2_sym)
                print("λ₂ random-walk: ", lam2_rw)

            rows.append({
                "sim_metric": sim_metric,
                "method": method,
                "params_str": params_str,
                "label": label,
                "params": params,
                "nodes": nodes,
                "edges": edges,
                "lam2_standard": lam2_std,
                "lam2_symmetric": lam2_sym,
                "lam2_randomwalk": lam2_rw,
                "cut_balance_standard": cut_std,
                "cut_balance_symmetric": cut_sym,
                "cut_balance_randomwalk": cut_rw
            })

            # Store in-memory Laplacians and Fiedler vectors for embeddings
            final_output[label] = {
                "Standard": {"L": L_std, "fiedler": f2_std, "labels_2way": (f2_std >= np.median(f2_std)).astype(int)},
                "Symmetric": {"L": L_sym, "fiedler": f2_sym, "labels_2way": (f2_sym >= np.median(f2_sym)).astype(int)},
                "RandomWalk": {"L": L_rw, "fiedler": f2_rw, "labels_2way": (f2_rw >= np.median(f2_rw)).astype(int)}
            }

        df = pd.DataFrame(rows)
        df.set_index(["sim_metric", "method", "params_str"], inplace=True)

        csv_path = self.save_dir / f"{name_prefix}_summary.csv"
        df.to_csv(csv_path)
        print(f"[INFO] Saved Fiedler summary → {csv_path.resolve()}")

        return final_output, df

    def _compute_cut_balance(self, fied):
        labels = (fied >= np.median(fied)).astype(int)
        return f"{np.sum(labels==0)}/{np.sum(labels==1)}"

    def _subplot_fiedlers(self, label, f2_std, f2_sym, f2_rw):
        fiedlers = [
            ("Standard", f2_std),
            ("Symmetric", f2_sym),
            ("Random-Walk", f2_rw),
        ]
        fig, axs = plt.subplots(1, 3, figsize=(14, 4))
        for ax, (name, vec) in zip(axs, fiedlers):
            order = np.argsort(vec)
            ax.plot(vec[order], "o-", markersize=3)
            ax.set_title(f"{label}\n{name} Fiedler")
            ax.grid(True)
        fig.tight_layout()
        plt.show()

class LaplacianEmbeddings:

    def __init__(self, save_dir: Path = None, show_plots: bool = True):
        self.show_plots = show_plots
        self.save_dir = Path(save_dir) if save_dir is not None else Path("outputs/embeddings")
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def compute_and_plot(self, fied_results: dict, label_name: str = "Graph"):
        """
        fied_results: dict
            {
                "Standard": {"L": ..., "fiedler": ..., "labels_2way": ...},
                "Symmetric": {...},
                "RandomWalk": {...}
            }
        """
        embeddings = {}
        n_laps = len(fied_results)
        fig, axs = plt.subplots(n_laps, 2, figsize=(12, 5*n_laps))

        if n_laps == 1:
            axs = np.array([axs])  # ensure axs is 2D array

        for i, (lap_type, info) in enumerate(fied_results.items()):
            L = info["L"]
            fied = info["fiedler"].reshape(-1, 1)
            labels = info.get("labels_2way", None)

            # 2D spectral embedding
            A_sim = np.eye(L.shape[0]) - L
            se = SpectralEmbedding(n_components=2, affinity='precomputed')
            emb2d = se.fit_transform(A_sim)

            embeddings[lap_type] = {1: fied, 2: emb2d}

            # Plotting
            ax1, ax2 = axs[i]
            ax1.scatter(range(len(fied)), fied[:, 0], c=labels, cmap='coolwarm', s=50)
            ax1.set_xlabel("Node index")
            ax1.set_ylabel("Fiedler value")
            ax1.set_title(f"{lap_type} Laplacian 1D")
            ax1.grid(True)

            ax2.scatter(emb2d[:, 0], emb2d[:, 1], c=labels, cmap='coolwarm', s=50)
            ax2.set_xlabel("Dimension 1")
            ax2.set_ylabel("Dimension 2")
            ax2.set_title(f"{lap_type} Laplacian 2D")
            ax2.grid(True)

        fig.suptitle(f"{label_name} - Laplacian Embeddings", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        if self.show_plots:
            plt.show()

        return embeddings