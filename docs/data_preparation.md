# DataPreparer Class Documentation

## Overview
The `DataPreparer` class handles data acquisition for modular analysis in the Spectral Clustering pipeline. It supports both synthetic data generation and user-provided datasets (edge lists or matrices). The class ensures proper formatting, adjacency creation, and optional symmetrization for undirected graphs.

## Supported Data Sources

1. **Synthetic Data**
   - `synthetic_voting`: Generates a synthetic voting dataset.
   - `synthetic_performance`: Generates a synthetic performance dataset.

2. **User-Provided Data**
   - **Edge Lists**: CSV, JSON, or TXT (space-separated) formats.
   - **Matrices**: CSV or JSON formats.

## Main Features

- Automatic path resolution for relative and absolute paths.
- Supports optional weight columns for edge lists.
- Enforces undirected adjacency matrix symmetry.
- Handles self-loops (optionally skipped).
- Caches loaded/generated data for repeated access.

## Initialization

**Parameters:**

- `mode: str`  
  - `'synthetic_voting'`, `'synthetic_performance'`, `'user_data'`
- `user_file_type: str`  
  - `'edge_list'` or `'matrix'` (only used if `mode='user_data'`)
- `user_path: str`  
  - Path to the user-provided dataset (CSV, JSON, or TXT)
- `src_col: str`  
  - Column name or index for source nodes
- `dst_col: str`  
  - Column name or index for destination nodes
- `weight_col: Optional[str]`  
  - Column name or index for edge weights
- `n_blocks: int`  
  - Number of blocks for synthetic data
- `block_size: int`  
  - Size of each block for synthetic data
- `seed: int`  
  - Random seed for reproducibility
- `undirected: bool`  
  - If `True`, symmetrize adjacency matrix
- `default_weight: float`  
  - Default weight for edges when weight column is missing

## Public Methods

### `prepare() -> Dict[str, Any]`
- Main entry point.
- Returns a dictionary with keys:
  - `matrix`: `np.ndarray` adjacency or data matrix
  - `nodes`: `np.ndarray` of node labels (if edge list)
  - `meta`: metadata dictionary (synthetic data)

## Internal Methods

- `_load_synthetic() -> Dict[str, Any]`  
  Generates synthetic voting or performance datasets.

- `_load_user_data() -> Dict[str, Any]`  
  Loads user-provided datasets based on type.

- `_load_edge_list() -> Dict[str, Any]`  
  Dispatches edge list loading based on file extension (CSV, JSON, TXT).

- `_load_edge_list_json() -> Tuple[np.ndarray, np.ndarray]`  
  Loads JSON edge lists with optional weight column.

- `_load_edge_list_txt() -> Tuple[np.ndarray, np.ndarray]`  
  Loads space-separated TXT edge lists (e.g., SNAP datasets).  
  Skips comment lines starting with `#`. Automatically handles default weights.

- `_load_matrix() -> Dict[str, Any]`  
  Loads matrices from CSV or JSON.

## Usage Examples

### 1. Synthetic Voting Data
```python
dp = DataPreparer(mode='synthetic_voting', n_blocks=3, block_size=50)
data = dp.prepare()
print(data['matrix'].shape)
print(data['meta'])
``` 

### 2. User CSV edge list. 

```python 
dp = DataPreparer(
    mode='user_data',
    user_file_type='edge_list',
    user_path='my_edges.csv',
    src_col='source',
    dst_col='target',
    weight_col='weight',
    undirected=True
)
data = dp.prepare()
print(data['matrix'].shape)
print(len(data['nodes']))
``` 

### SNAP TXT Edge List

```python 
dp = DataPreparer(
    mode='user_data',
    user_file_type='edge_list',
    user_path='ca-GrQc.txt',
    src_col=0,
    dst_col=1,
    undirected=True
)
data = dp.prepare()
print(data['matrix'].shape)
print(len(data['nodes']))
```
## Notes 

- For undirected graphs, the adjacency matrix is symmetrized automatically.

- Self-loops are skipped by default.

- Weight column is optional; if missing, all edges default to weight 1.0.

- TXT edge lists must be space-separated; comment lines starting with # are ignored.

- Synthetic datasets return a meta dictionary describing block structure.