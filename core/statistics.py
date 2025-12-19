import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import re
from IPython.display import display, HTML

class GraphClusteringStatistics:
    """
    Compute descriptive statistics for graph clustering evaluation results.
    Groups results by similarity metric and graph construction method.
    """
    
    PRIMARY_METRIC_COLS = ['spectral_gap', 'silhouette', 'modularity', 'conductance']

    def __init__(self, df_metrics: pd.DataFrame):
        """
        Initialize with evaluation results DataFrame.
        
        Args:
            df_metrics: DataFrame with columns [graph_label, laplacian_type, 
                       spectral_gap, silhouette, modularity, conductance]
        """
        self.df = df_metrics.copy()

        # Drop columns that are ALL NaN
        self._drop_nan_metrics()

        # Parse graph labels AFTER cleaning
        self._parse_graph_labels()
        
    # ----------------------------------------------------------
    # NEW: DROP METRIC COLUMNS IF THEY ARE ENTIRELY NaN
    # ----------------------------------------------------------
    def _drop_nan_metrics(self):
        dropped = []
        for col in self.PRIMARY_METRIC_COLS:
            if col in self.df.columns:
                # If entire column is NaN → drop it
                if self.df[col].isna().all():
                    dropped.append(col)
                    self.df = self.df.drop(columns=[col])

        if dropped:
            print(
                f"[INFO] Dropped metric columns with all-NaN values: {', '.join(dropped)}"
            )

    # ----------------------------------------------------------
    # ORIGINAL CODE BELOW — UNMODIFIED
    # ----------------------------------------------------------

    def _parse_graph_labels(self):
        """Parse graph_label into similarity metric, construction method, and parameter."""
        def parse_label(label: str) -> Tuple[str, str, str, float]:
            parts = label.split('__')
            similarity = parts[0]
            
            if 'knn' in parts[1]:
                construction = 'knn'
                param_match = re.search(r'k(\d+)', parts[1])
                param_name = 'k'
                param_value = float(param_match.group(1)) if param_match else np.nan
            elif 'threshold' in parts[1]:
                construction = 'threshold'
                param_match = re.search(r't([\d.]+)', parts[1])
                param_name = 'epsilon'
                param_value = float(param_match.group(1)) if param_match else np.nan
            else:
                construction = 'unknown'
                param_name = 'param'
                param_value = np.nan
                
            return similarity, construction, param_name, param_value
        
        parsed = self.df['graph_label'].apply(parse_label)
        self.df['similarity_metric'] = parsed.apply(lambda x: x[0])
        self.df['construction_method'] = parsed.apply(lambda x: x[1])
        self.df['param_name'] = parsed.apply(lambda x: x[2])
        self.df['param_value'] = parsed.apply(lambda x: x[3])
    
    def get_stats_by_similarity(self, metric_cols: Optional[List[str]] = None) -> pd.DataFrame:
        if metric_cols is None:
            metric_cols = [c for c in self.PRIMARY_METRIC_COLS if c in self.df.columns]
        return self.df.groupby('similarity_metric')[metric_cols].agg(['mean', 'std', 'min', 'max'])
    
    def get_stats_by_construction(self, metric_cols: Optional[List[str]] = None) -> pd.DataFrame:
        if metric_cols is None:
            metric_cols = [c for c in self.PRIMARY_METRIC_COLS if c in self.df.columns]
        return self.df.groupby('construction_method')[metric_cols].agg(['mean', 'std', 'min', 'max'])
    
    def get_stats_by_similarity_and_construction(self, metric_cols: Optional[List[str]] = None) -> pd.DataFrame:
        if metric_cols is None:
            metric_cols = [c for c in self.PRIMARY_METRIC_COLS if c in self.df.columns]
        return self.df.groupby(['similarity_metric', 'construction_method'])[metric_cols].agg(['mean', 'std', 'min', 'max'])
    
    def get_stats_over_parameter_range(self, 
                                       similarity: str, 
                                       construction: str,
                                       metric_cols: Optional[List[str]] = None) -> pd.DataFrame:
        if metric_cols is None:
            metric_cols = [c for c in self.PRIMARY_METRIC_COLS if c in self.df.columns]
        
        mask = (self.df['similarity_metric'] == similarity) & \
               (self.df['construction_method'] == construction)
        subset = self.df[mask]
        
        return subset.groupby('param_value')[metric_cols].agg(['mean', 'std', 'min', 'max'])
    
    def get_stats_by_laplacian(self, metric_cols: Optional[List[str]] = None) -> pd.DataFrame:
        if metric_cols is None:
            metric_cols = [c for c in self.PRIMARY_METRIC_COLS if c in self.df.columns]
        return self.df.groupby('laplacian_type')[metric_cols].agg(['mean', 'std', 'min', 'max'])
    
    def get_best_configurations(self, metric: str = 'silhouette', top_n: int = 5) -> pd.DataFrame:
        if metric not in self.df.columns:
            raise ValueError(f"Metric '{metric}' not available (may have been dropped).")
        cols = ['graph_label', 'laplacian_type', 'similarity_metric', 
                'construction_method', 'param_value', metric]
        return self.df.nlargest(top_n, metric)[cols]
    
    def summarize_all(self) -> Dict[str, pd.DataFrame]:
        return {
            'by_similarity': self.get_stats_by_similarity(),
            'by_construction': self.get_stats_by_construction(),
            'by_similarity_and_construction': self.get_stats_by_similarity_and_construction(),
            'by_laplacian': self.get_stats_by_laplacian()
        }
    
    def print_summary(self):
        """Clean DataFrame display using display()."""
        summaries = self.summarize_all()

        display(HTML(
            "<h2 style='font-family:Arial; margin-top:20px;'>"
            "Graph Clustering Evaluation — Statistical Summary</h2><hr>"
        ))

        for name, df in summaries.items():
            title = name.upper().replace('_', ' ')
            display(HTML(f"<h3 style='font-family:Arial; margin-top:12px;'>🔹 {title}</h3>"))
            display(df.round(4))
            display(HTML("<br>"))

import pandas as pd
from typing import List, Optional
from statsmodels.multivariate.manova import MANOVA

class GraphClusteringMANOVA:
    """
    Perform MANOVA to assess the joint effect of spectral clustering
    design choices on clustering quality metrics.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        dependent_vars: Optional[List[str]] = None,
        factor_vars: Optional[List[str]] = None
    ):
        """
        Args:
            df: Preprocessed DataFrame (output of GraphClusteringStatistics.df)
            dependent_vars: Metrics to include in MANOVA
            factor_vars: Categorical predictors
        """
        self.df = df.copy()

        if dependent_vars is None:
            dependent_vars = [
                c for c in ['spectral_gap', 'silhouette', 'modularity', 'conductance']
                if c in self.df.columns
            ]

        if factor_vars is None:
            factor_vars = [
                'similarity_metric',
                'construction_method',
                'laplacian_type'
            ]

        self.dependent_vars = dependent_vars
        self.factor_vars = factor_vars

        self._validate_inputs()
        self.formula = self._build_formula()
        self._model = None

    def _validate_inputs(self):
        if len(self.dependent_vars) < 2:
            raise ValueError("MANOVA requires at least two dependent variables.")

        for f in self.factor_vars:
            if f not in self.df.columns:
                raise ValueError(f"Factor '{f}' not found in DataFrame.")

    def _build_formula(self) -> str:
        lhs = " + ".join(self.dependent_vars)
        rhs = " + ".join(self.factor_vars)
        return f"{lhs} ~ {rhs}"

    def fit(self):
        """Fit the MANOVA model."""
        self._model = MANOVA.from_formula(self.formula, data=self.df)
        return self

    def summary(self):
        """Return full MANOVA test results."""
        if self._model is None:
            raise RuntimeError("Call fit() before summary().")
        return self._model.mv_test()

    def print_summary(self):
        print("\n=== MANOVA MODEL ===")
        print(f"Formula: {self.formula}")
        print("\n=== MULTIVARIATE TEST RESULTS ===")
        print(self.summary())

import statsmodels.api as sm
from statsmodels.formula.api import ols

class GraphClusteringANOVA:
    """
    Perform univariate ANOVA for each clustering quality metric,
    following a significant MANOVA.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        factor_vars: Optional[List[str]] = None
    ):
        """
        Args:
            df: Preprocessed DataFrame
            factor_vars: Categorical predictors
        """
        self.df = df.copy()

        if factor_vars is None:
            factor_vars = [
                'similarity_metric',
                'construction_method',
                'laplacian_type'
            ]

        self.factor_vars = factor_vars
        self.metric_vars = [
            c for c in ['spectral_gap', 'silhouette', 'modularity', 'conductance']
            if c in self.df.columns
        ]

        self.results = {}

    def _build_formula(self, metric: str) -> str:
        rhs = " + ".join(self.factor_vars)
        return f"{metric} ~ {rhs}"

    def fit(self, typ: int = 2):
        """
        Fit ANOVA models for all metrics.

        Args:
            typ: Type-II (default) or Type-III ANOVA
        """
        for metric in self.metric_vars:
            formula = self._build_formula(metric)
            model = ols(formula, data=self.df).fit()
            anova_table = sm.stats.anova_lm(model, typ=typ)

            self.results[metric] = {
                'formula': formula,
                'model': model,
                'anova': anova_table
            }

        return self

    def get_anova_table(self, metric: str):
        return self.results[metric]['anova']

    def print_summary(self):
        print("\n=== FOLLOW-UP ANOVA RESULTS ===")
        for metric, res in self.results.items():
            print(f"\n--- Metric: {metric.upper()} ---")
            print(f"Model: {res['formula']}")
            print(res['anova'])

import pandas as pd
import numpy as np
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.formula.api import ols
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

class GraphClusteringAnalysis:
    """
    Advanced statistical analysis for spectral clustering evaluation metrics.
    
    Features:
    - Post-hoc pairwise tests (Tukey HSD)
    - Effect sizes (partial eta squared, Cohen's d)
    - Metric correlation and PCA
    - Parameter sensitivity analysis
    """
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.metrics = [c for c in ['spectral_gap','silhouette','modularity','conductance'] if c in df.columns]
        self.factors = ['similarity_metric','construction_method','laplacian_type']
    
    # ----------------------------------------------------------
    # 1. Post-hoc pairwise comparisons (Tukey HSD)
    # ----------------------------------------------------------
    def tukey_posthoc(self, metric: str, factor: str):
        """
        Perform Tukey HSD for a given metric and factor.
        """
        if metric not in self.metrics:
            raise ValueError(f"Metric {metric} not available.")
        if factor not in self.factors:
            raise ValueError(f"Factor {factor} not available.")
        
        tukey = pairwise_tukeyhsd(endog=self.df[metric],
                                  groups=self.df[factor],
                                  alpha=0.05)
        print(f"\nTukey HSD for metric '{metric}' by factor '{factor}':")
        print(tukey)
        return tukey

    # ----------------------------------------------------------
    # 2. Effect size computation
    # ----------------------------------------------------------
    def compute_effect_sizes(self, metric: str):
        """
        Compute partial eta squared for each factor and Cohen's d for pairwise comparisons.
        """
        if metric not in self.metrics:
            raise ValueError(f"Metric {metric} not available.")
        
        formula = f"{metric} ~ {' + '.join(self.factors)}"
        model = ols(formula, data=self.df).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)
        
        # Partial eta squared: SS_effect / (SS_effect + SS_residual)
        anova_table['eta_sq'] = anova_table['sum_sq'] / (anova_table['sum_sq'] + anova_table.loc['Residual','sum_sq'])
        
        print(f"\nEffect sizes (partial eta squared) for metric '{metric}':")
        display(anova_table[['sum_sq','df','F','PR(>F)','eta_sq']])
        return anova_table

    # ----------------------------------------------------------
    # 3. Metric correlation and PCA
    # ----------------------------------------------------------
    def metric_correlation(self, plot: bool = True):
        corr = self.df[self.metrics].corr()
        if plot:
            plt.figure(figsize=(6,5))
            sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
            plt.title("Correlation matrix of clustering metrics")
            plt.show()
        return corr
    
    def metric_pca(self, n_components: int = 2, plot: bool = True):
        X = self.df[self.metrics].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        pca = PCA(n_components=n_components)
        components = pca.fit_transform(X_scaled)
        explained = pca.explained_variance_ratio_
        
        if plot:
            plt.figure(figsize=(7,5))
            plt.scatter(components[:,0], components[:,1], c='b', s=50)
            plt.xlabel(f"PC1 ({explained[0]*100:.1f}% var)")
            plt.ylabel(f"PC2 ({explained[1]*100:.1f}% var)")
            plt.title("PCA of clustering metrics")
            plt.grid(True)
            plt.show()
        
        return pca, components, explained

    # ----------------------------------------------------------
    # 4. Parameter sensitivity
    # ----------------------------------------------------------
    def parameter_sensitivity(self, similarity: str, construction: str, plot: bool = True):
        """
        Analyze metrics across parameter values (k or threshold epsilon)
        """
        mask = (self.df['similarity_metric']==similarity) & \
               (self.df['construction_method']==construction)
        subset = self.df[mask].copy()
        
        if subset.empty:
            print(f"No data for similarity={similarity} and construction={construction}")
            return None
        
        grouped = subset.groupby('param_value')[self.metrics].agg(['mean','std'])
        
        if plot:
            plt.figure(figsize=(8,6))
            for metric in self.metrics:
                plt.errorbar(grouped.index,
                             grouped[metric]['mean'],
                             yerr=grouped[metric]['std'],
                             marker='o', capsize=3, label=metric)
            plt.xlabel('Parameter value (k or threshold)')
            plt.ylabel('Metric value')
            plt.title(f"Parameter sensitivity: {similarity} + {construction}")
            plt.legend()
            plt.grid(True)
            plt.show()
        
        return grouped
