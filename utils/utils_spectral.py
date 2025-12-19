import numpy as np
import pandas as pd
from numpy.linalg import norm
from scipy.sparse.csgraph import laplacian as csgraph_laplacian
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans

# ----------------------
# Synthetic data
# ----------------------
def make_synthetic_voting(n_blocks=2, block_size=30, p_in=0.9, p_out=0.1, seed=42):
    """
    Returns a binary voting similarity-like matrix (co-vote probability model),
    and a metadata DataFrame with block labels.
    """
    rng = np.random.default_rng(seed)
    sizes = [block_size]*n_blocks
    n = sum(sizes)
    labels = np.concatenate([[b]*sizes[b] for b in range(n_blocks)])
    # simulate votes as Bernoulli per-bill with block-level prototypes
    # build co-voting similarity by cosine on vote vectors
    # Create V matrix: n x m votes, with m bills
    m = 60
    # prototypes per block
    proto = rng.binomial(1, 0.5, size=(n_blocks, m))
    V = np.zeros((n, m), dtype=float)
    idx = 0
    for b, sz in enumerate(sizes):
        for i in range(sz):
            flip = rng.binomial(1, p_out, size=m)  # sparse flips across blocks
            v = np.where(flip==1, 1-proto[b], proto[b])
            # add individual noise
            noise = rng.binomial(1, 0.05, size=m)
            v = np.where(noise==1, 1-v, v)
            V[idx] = v
            idx += 1
    meta = pd.DataFrame({"node": np.arange(n), "block": labels})
    return V, meta

def make_synthetic_performance(n_blocks=3, block_size=25, centers=None, noise=0.4, seed=123):
    """
    Returns a continuous rating matrix R (evaluators x performers) with block structure,
    and metadata for evaluators/performers.
    """
    rng = np.random.default_rng(seed)
    if centers is None:
        centers = np.array([[ 1.5,  1.0,  0.0],
                            [-1.0,  1.2, -0.5],
                            [ 0.5, -1.2,  1.5]], dtype=float)
    k = len(centers)
    sizes = [block_size]*k
    n = sum(sizes)
    d = centers.shape[1]
    X = np.zeros((n, d))
    labels = np.concatenate([[b]*sizes[b] for b in range(k)])
    idx = 0
    for b, sz in enumerate(sizes):
        X[idx:idx+sz] = centers[b] + noise * rng.normal(size=(sz, d))
        idx += sz
    # Treat X as feature vectors; later we'll construct similarity
    meta = pd.DataFrame({"node": np.arange(n), "block": labels})
    return X, meta

# ----------------------
# Similarities
# ----------------------
def cosine_similarity(X):
    Xn = X / (norm(X, axis=1, keepdims=True) + 1e-12)
    S = Xn @ Xn.T
    np.fill_diagonal(S, 0.0)
    return S

def correlation_similarity(X):
    Xc = X - X.mean(axis=1, keepdims=True)
    return cosine_similarity(Xc)

def jaccard_similarity_binary(Xbin):
    # Xbin: rows are nodes, columns are binary features
    inter = Xbin @ Xbin.T
    row_sums = Xbin.sum(axis=1, keepdims=True)
    union = row_sums + row_sums.T - inter + 1e-12
    S = inter / union
    np.fill_diagonal(S, 0.0)
    return S

# ----------------------
# Graph construction
# ----------------------
def adjacency_from_similarity(S, method="threshold", threshold=0.5, k=10, symmetrize=True):
    n = S.shape[0]
    A = np.zeros_like(S)
    if method == "threshold":
        A = (S >= threshold).astype(float) * S
    elif method == "knn":
        for i in range(n):
            idx = np.argsort(-S[i])[:k]
            A[i, idx] = S[i, idx]
    else:
        raise ValueError("method must be 'threshold' or 'knn'")
    if symmetrize:
        A = np.maximum(A, A.T)
    np.fill_diagonal(A, 0.0)
    return A

def degree_matrix(A):
    d = A.sum(axis=1)
    return np.diag(d)

# ----------------------
# Laplacians
# ----------------------
def laplacians(A):
    """
    Returns degree matrix D, unnormalized Laplacian L,
    symmetric normalized Laplacian L_sym, and random-walk normalized Laplacian L_rw.
    - This is a revised function from main.
    """
    D = degree_matrix(A)
    L = D - A

    d = np.diag(D).astype(float)
    # avoid division by zero for isolated nodes
    d_safe = np.where(d > 1e-12, d, np.inf)

    D_inv = np.diag(1.0 / d_safe)
    D_sqrt_inv = np.diag(1.0 / np.sqrt(d_safe))

    L_sym = np.eye(A.shape[0]) - D_sqrt_inv @ A @ D_sqrt_inv
    L_rw = np.eye(A.shape[0]) - D_inv @ A

    return D, L, L_sym, L_rw
# ----------------------
# Eigen and embedding
# ----------------------
def eigh_symmetric(M, k=None):
    # Full eigh; optionally return first k pairs
    vals, vecs = np.linalg.eigh(M)
    if k is None or k >= len(vals):
        return vals, vecs
    return vals[:k], vecs[:, :k]

def fiedler_vector(L):
    vals, vecs = np.linalg.eigh(L)
    return vals[1], vecs[:,1]

def spectral_embedding(L, k=2, skip_first=True):
    vals, vecs = np.linalg.eigh(L)
    start = 1 if skip_first else 0
    U = vecs[:, start:start+k]
    return U, vals[start:start+k]

# ----------------------
# Metrics
# ----------------------
def conductance(A, labels):
    # labels: array of cluster ids (0..k-1)
    A = np.asarray(A, float)
    labels = np.asarray(labels)
    unique = np.unique(labels)
    if len(unique) != 2:
        raise ValueError("conductance defined here for 2-way partition.")
    S = np.where(labels==unique[0])[0]
    T = np.where(labels==unique[1])[0]
    cut = A[np.ix_(S,T)].sum()
    d = A.sum(1)
    volS = d[S].sum()
    volT = d[T].sum()
    return cut / max(min(volS, volT), 1e-12)

def modularity(A, labels):
    A = np.asarray(A, float)
    m = A.sum() / 2.0
    d = A.sum(1, keepdims=True)
    P = (d @ d.T) / (2*m + 1e-12)
    B = A - P
    # Sum within communities
    Q = 0.0
    for c in np.unique(labels):
        idx = np.where(labels==c)[0]
        Q += B[np.ix_(idx, idx)].sum()
    return Q / (2*m + 1e-12)

def spectral_gap(vals, k=1):
    vals = np.sort(vals)
    return vals[k] - vals[k-1]

def silhouette(embedding, labels):
    if len(np.unique(labels)) < 2:
        return np.nan
    return silhouette_score(embedding, labels)

# ----------------------
# Clustering
# ----------------------
def kmeans_cluster(embedding, k, seed=123):
    km = KMeans(n_clusters=k, n_init=10, random_state=seed)
    labels = km.fit_predict(embedding)
    return labels

# ----------------------
# IO helpers
# ----------------------
def load_edge_list_csv(path, src='src', dst='dst', weight=None, n=None):
    df = pd.read_csv(path)
    if weight and weight in df.columns:
        W = df[weight].values
    else:
        W = np.ones(len(df), dtype=float)
    nodes = pd.Index(pd.unique(df[[src,dst]].values.ravel()))
    if n is None:
        n = len(nodes)
    idx = {node:i for i,node in enumerate(nodes[:n])}
    A = np.zeros((n,n), dtype=float)
    for s,t,w in zip(df[src], df[dst], W):
        if s in idx and t in idx:
            i, j = idx[s], idx[t]
            if i!=j:
                A[i,j] += w
                A[j,i] += w
    return A, nodes[:n]

def load_matrix_csv(path):
    X = pd.read_csv(path, header=None).values
    return X
