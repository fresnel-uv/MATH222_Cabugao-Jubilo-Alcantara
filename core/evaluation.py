import numpy as np
from pathlib import Path
import utils.utils_spectral as utils

class GraphClusteringEvaluator:
    """
    Evaluates graph clustering results from SpectralEmbeddingAnalysis.
    Computes:
        - spectral gap
        - silhouette
        - modularity
        - conductance
    """

    def __init__(self, utils_module=utils):
        self.utils = utils_module

    def evaluate_all_laplacians(self, results_dict, k=2):
        """
        Input format required:
            results_dict = {
                "A": adjacency matrix,
                "L":      {"U": ..., "vals": ..., "labels": ...},
                "L_sym":  {"U": ..., "vals": ..., "labels": ...},
                "L_rw":   {"U": ..., "vals": ..., "labels": ...}
            }
        """

        A = results_dict["A"]
        metrics_dict = {}

        for lap_name in ["L", "L_sym", "L_rw"]:
            if lap_name not in results_dict:
                continue

            res = results_dict[lap_name]

            U      = res["U"]      # embedding
            vals   = res["vals"]   # eigenvalues
            labels = res["labels"] # cluster labels

            metrics = {}

            # Safe spectral gap: only if enough eigenvalues
            if len(vals) > k:
                metrics["spectral_gap"] = self.utils.spectral_gap(vals, k=k)
            else:
                metrics["spectral_gap"] = np.nan

            # Silhouette score (embedding-level)
            metrics["silhouette"] = self.utils.silhouette(U, labels)

            # Modularity
            metrics["modularity"] = self.utils.modularity(A, labels)

            # Conductance (only for 2 clusters)
            if len(np.unique(labels)) == 2:
                metrics["conductance"] = self.utils.conductance(A, labels)
            else:
                metrics["conductance"] = np.nan

            metrics_dict[lap_name] = metrics

        return metrics_dict
