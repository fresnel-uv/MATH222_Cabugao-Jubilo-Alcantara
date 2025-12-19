import json
import pandas as pd
import numpy as np
from pathlib import Path
from utils import utils_spectral as utils
from typing import Optional, Dict, Any, Tuple

# -------------------------
# Package data folder
# -------------------------
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PACKAGE_ROOT / "data"


class DataPreparer:
    """
    Handles data acquisition (synthetic or user-provided) for modular analysis.

    Supports:
    - Synthetic data: 'synthetic_voting', 'synthetic_performance'
    - User-provided edge lists (CSV/JSON/TXT)
    - User-provided matrices (CSV/JSON)
    """

    def __init__(
        self,
        mode: str = 'synthetic_voting',
        user_file_type: str = 'edge_list',
        user_path: str = '',
        src_col: str = 'src',
        dst_col: str = 'dst',
        weight_col: Optional[str] = None,
        n_blocks: int = 2,
        block_size: int = 30,
        seed: int = 42,
        undirected: bool = True,
        default_weight: float = 1.0,
    ):
        # Validate mode
        valid_modes = ['synthetic_voting', 'synthetic_performance', 'user_data']
        if mode not in valid_modes:
            raise ValueError(f"Unknown mode: {mode}, must be one of {valid_modes}")

        valid_types = ['edge_list', 'matrix']
        if user_file_type not in valid_types:
            raise ValueError(f"Unknown user_file_type: {user_file_type}")

        self.mode = mode
        self.user_file_type = user_file_type

        # -----------------------------
        # Path resolution
        # -----------------------------
        if user_path:
            p = Path(user_path)
            self.user_path = p if p.is_absolute() else DATA_DIR / p
        else:
            self.user_path = None

        self.src_col = src_col
        self.dst_col = dst_col
        self.weight_col = weight_col
        self.n_blocks = n_blocks
        self.block_size = block_size
        self.seed = seed
        self.undirected = undirected
        self.default_weight = default_weight
        self.data: Dict[str, Any] = {}

    # -----------------------------
    # PUBLIC INTERFACE
    # -----------------------------
    def prepare(self) -> Dict[str, Any]:
        """Main entry point: loads or generates data based on mode."""
        if self.data:
            return self.data  # cache

        if self.mode.startswith('synthetic'):
            return self._load_synthetic()
        elif self.mode == 'user_data':
            return self._load_user_data()
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    # -----------------------------
    # SYNTHETIC DATA
    # -----------------------------
    def _load_synthetic(self) -> Dict[str, Any]:
        if self.mode == 'synthetic_voting':
            V, meta = utils.make_synthetic_voting(
                n_blocks=self.n_blocks,
                block_size=self.block_size,
                seed=self.seed,
            )
            self.data = {'matrix': V, 'meta': meta, 'nodes': None}

        elif self.mode == 'synthetic_performance':
            X, meta = utils.make_synthetic_performance(
                n_blocks=self.n_blocks,
                block_size=self.block_size,
                seed=self.seed,
            )
            self.data = {'matrix': X, 'meta': meta, 'nodes': None}

        return self.data

    # -----------------------------
    # USER DATA LOADING
    # -----------------------------
    def _load_user_data(self) -> Dict[str, Any]:
        if not self.user_path or not self.user_path.exists():
            raise ValueError(f"File not found: {self.user_path}")

        if self.user_file_type == 'edge_list':
            return self._load_edge_list()
        elif self.user_file_type == 'matrix':
            return self._load_matrix()
        else:
            raise ValueError(f"Unsupported file type: {self.user_file_type}")

    # -----------------------------
    # EDGE LIST
    # -----------------------------
    def _load_edge_list(self) -> Dict[str, Any]:
        ext = self.user_path.suffix.lower()

        if ext == '.csv':
            df = pd.read_csv(self.user_path)
            missing = {self.src_col, self.dst_col} - set(df.columns)
            if missing:
                raise ValueError(f"CSV missing required columns: {missing}")

            if self.weight_col and self.weight_col not in df.columns:
                df[self.weight_col] = self.default_weight

            A, nodes = utils.load_edge_list_csv(
                str(self.user_path),
                self.src_col,
                self.dst_col,
                self.weight_col,
                n=None,
            )

        elif ext == '.json':
            A, nodes = self._load_edge_list_json()

        elif ext == '.txt':
            A, nodes = self._load_edge_list_txt()

        else:
            raise ValueError("Unsupported file format. Use .csv, .json, or .txt for edge lists.")

        if self.undirected:
            if not np.allclose(A, A.T):
                print("[WARN] Adjacency not symmetric. Symmetrizing.")
            A = np.maximum(A, A.T)

        self.data = {'matrix': A, 'nodes': nodes, 'meta': None}
        return self.data

    def _load_edge_list_json(self) -> Tuple[np.ndarray, np.ndarray]:
        with open(self.user_path, 'r') as f:
            edges = json.load(f)

        df = pd.DataFrame(edges)
        required = {self.src_col, self.dst_col}
        if not required.issubset(df.columns):
            raise ValueError(f"JSON must contain columns: {required}")

        srcs = df[self.src_col].astype(str).values
        dsts = df[self.dst_col].astype(str).values
        weights = (df[self.weight_col].astype(float).values
                   if self.weight_col and self.weight_col in df.columns
                   else np.full(len(df), self.default_weight, dtype=float))

        nodes = pd.Index(pd.unique(np.concatenate([srcs, dsts]))).astype(str)
        index = {n: i for i, n in enumerate(nodes)}
        n = len(nodes)

        A = np.zeros((n, n), dtype=float)
        for s, d, w in zip(srcs, dsts, weights):
            i, j = index[s], index[d]
            if i == j:
                continue
            A[i, j] += w
            if self.undirected:
                A[j, i] += w

        return A, nodes.values

    def _load_edge_list_txt(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load a space-separated TXT edge list (like SNAP datasets).
        Skips lines starting with '#'.
        """
        df = pd.read_csv(
            self.user_path,
            delim_whitespace=True,
            comment='#',
            header=None,
            names=[self.src_col, self.dst_col] if not self.weight_col else [self.src_col, self.dst_col, self.weight_col]
        )

        srcs = df[self.src_col].astype(str).values
        dsts = df[self.dst_col].astype(str).values
        weights = (df[self.weight_col].astype(float).values
                   if self.weight_col and self.weight_col in df.columns
                   else np.ones(len(df), dtype=float))

        nodes = pd.Index(pd.unique(np.concatenate([srcs, dsts]))).astype(str)
        index = {n: i for i, n in enumerate(nodes)}
        n = len(nodes)

        A = np.zeros((n, n), dtype=float)
        for s, d, w in zip(srcs, dsts, weights):
            i, j = index[s], index[d]
            if i == j:
                continue
            A[i, j] += w
            if self.undirected:
                A[j, i] += w

        return A, nodes.values

    # -----------------------------
    # MATRIX LOADING
    # -----------------------------
    def _load_matrix(self) -> Dict[str, Any]:
        ext = self.user_path.suffix.lower()

        if ext == '.csv':
            X = utils.load_matrix_csv(str(self.user_path))
        elif ext == '.json':
            with open(self.user_path, 'r') as f:
                raw = json.load(f)
            X = np.array(raw["matrix"] if isinstance(raw, dict) and "matrix" in raw else raw)
        else:
            raise ValueError("Unsupported file format. Use .csv or .json for matrices.")

        self.data = {'matrix': X, 'nodes': None, 'meta': None}
        return self.data
