# File path: core/comparative.py

import numpy as np
from core.data_preparation import DataPreparer
from core.graph_generation import GraphGenerator
from core.laplacian import GraphLaplacianGenerator
from core.fielder import FiedlerAnalysis
from core.evaluation import GraphClusteringEvaluator

class ComparativeExperiment:
    """
    Run a full comparative study over similarity metrics, graph types, and Laplacians.
    """

    def __init__(self, data_mode="synthetic_voting", n_blocks=3, block_size=20):
        self.data_mode = data_mode
        self.n_blocks = n_blocks
        self.block_size = block_size
        self.sim_metrics = ["cosine", "correlation"]
        self.graph_methods = ["threshold", "knn"]
        self.laplacian_types = ["Standard", "Symmetric", "RandomWalk"]
        self.results = {}

        # Initialize evaluator
        self.evaluator = GraphClusteringEvaluator()

    def run(self):
        # Step 1: Prepare data
        dp = DataPreparer(mode=self.data_mode, n_blocks=self.n_blocks, block_size=self.block_size)
        data = dp.prepare()
        X = data["matrix"]

        for sim in self.sim_metrics:
            for method in self.graph_methods:
                # Generate graph
                gg = GraphGenerator(sim_metric=sim, graph_method=method, show_info=False, show_plot=False, show_heatmap=False)
                graph_data = gg.generate_graph(data)
                A = graph_data["A"]

                # Compute Laplacians
                gl = GraphLaplacianGenerator(sim_metric=sim, graph_method=method, show_info=False, show_plots=False)
                S, A_lap, (D, L, L_sym, L_rw) = gl.generate(X=X)

                # Fiedler analysis
                fiedler = FiedlerAnalysis(sim_metric=sim, graph_method=method, show_info=False, show_plots=False)
                fied_results = fiedler.run(X=X, A=A_lap, D=D)
                fied_results["A"] = A_lap  # add adjacency for evaluation

                # Evaluate
                metrics = self.evaluator.evaluate_all_laplacians(fied_results)
                self.results[(sim, method)] = metrics

        return self.results
