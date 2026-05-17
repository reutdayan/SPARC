from pathlib import Path
import argparse
import json
import os
import subprocess
import sys
import time
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Cap BLAS thread counts BEFORE importing numpy / scipy / sklearn.
#
# On big multi-socket nodes (e.g. hpc8h200-01 has >128 logical CPUs) sklearn's
# brute-force ``NearestNeighbors`` dispatches an OpenBLAS GEMM with one thread
# per core. The OpenBLAS shipped in this conda env was compiled with
# NUM_THREADS=128, so when the request exceeds that ceiling it allocates an
# "auxiliary array for thread metadata", hits a per-process memory-region
# cap, and SIGSEGVs:
#
#     OpenBLAS warning: precompiled NUM_THREADS exceeded ...
#     OpenBLAS : Program is Terminated. Because you tried to allocate
#     too many memory regions.
#     Segmentation fault (core dumped)
#
# Capping at 64 keeps every BLAS-using library (OpenBLAS, MKL, OMP) well
# inside that ceiling. We use setdefault so the user can still override
# from the shell. Must run before numpy is imported because numpy reads
# these once at load time.
# ---------------------------------------------------------------------------
for _v in (
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OMP_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_v, "64")

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA


DEFAULT_DATASETS = ["cora", "citeseer", "pubmed", "chameleon", "squirrel", "wikics"]

# All retrieval is done with L2 distance (per project spec).
DISTANCE_METRIC = "l2"

# Default grids requested by the user.
DEFAULT_TOPK_VALUES = [2, 5, 10, 20, 30]
DEFAULT_TRUE_HOPS_VALUES = [1, 2, 3]

# Spectral embeddings are computed at this fixed dimension (independent of SPARC's dim).
DEFAULT_SPECTRAL_DIM = 128

# Per-dataset spectral-dim caps. Used to keep ARPACK runtime/memory tractable
# on the very large graphs where computing 128 eigenpairs is heavy. Values
# here override the user-supplied ``--spectral_dim`` when smaller.
PER_DATASET_SPECTRAL_DIM_CAP = {
    "ogbn-products": 64,
}

# Optional per-dataset cap on the maximum hop value used to materialise the
# true cumulative h-hop neighborhood (``adjacency_neighbor_sets_by_hops``).
# Values here clip the user-supplied ``--true_hops_values`` when smaller.
# Leave this empty to disable capping for all datasets.
PER_DATASET_TRUE_HOPS_CAP = {
}

# Run/seed naming used to locate per-dataset artifacts.
DEFAULT_SPARC_RUN_NAME = "random_test0.1_seed42"
DEFAULT_COLD_BREW_SPLIT = "random"
DEFAULT_COLD_BREW_SEED = 42

# Defaults for the SPARC training step that runs before the SPARC kNN.
DEFAULT_SPARC_TRAIN_SPLIT = "random"
DEFAULT_SPARC_TRAIN_TEST_RATIO = 0.1
DEFAULT_SPARC_TRAIN_VAL_RATIO = 0.03
DEFAULT_SPARC_TRAIN_SEED = 42
# This suffix is appended to the SPARC run-dir name (`<split>_test<r>_seed<s>__<suffix>`)
# so retraining never clobbers a prior, pre-existing run.
DEFAULT_SPARC_TRAIN_SUFFIX = "nbpred_od64_ta0_pe1"
DEFAULT_SPARC_TRAIN_OVERRIDE = "sparc_train_override.json"
# Override delta stacked on top of DEFAULT_SPARC_TRAIN_OVERRIDE for these
# OGB-scale datasets: enables PE pretraining with smaller merged batches
# (higher num_clusters, lower bsize) so dense per-batch affinity fits in GPU
# memory; keeps eigenspace_agreement diagnostic off.
DEFAULT_SPARC_TRAIN_OVERRIDE_BIG = "sparc_train_override_big.json"
BIG_GRAPH_DATASETS = {"ogbn-arxiv", "reddit", "ogbn-products"}

# Override delta for *heterophilous* graphs. Stacked on top of
# DEFAULT_SPARC_TRAIN_OVERRIDE for datasets in HETEROPHILOUS_DATASETS.
# Cancels the base override's affinity_walk_order=1 (which is wrong for
# heterophilous graphs because 1-hop neighbors typically have *different*
# labels) and lowers spectral.lr from each dataset's stock 1e-3 to 3e-4
# (the stock LR caused the Rayleigh loss to diverge from epoch 1).
DEFAULT_SPARC_TRAIN_OVERRIDE_HETEROPHILOUS = (
    "sparc_train_override_heterophilous.json"
)
HETEROPHILOUS_DATASETS = {"chameleon", "squirrel", "minesweeper"}

# Override delta for datasets where PE-pretrain's dense per-batch eigh
# OOMs *because* the per-dataset config sets num_clusters=1 and the
# train graph is too big to fit a single dense Laplacian on the GPU.
# Currently just pubmed (~17K train nodes, 17K x 17K eigh ~3.3 GB).
# The override partitions the graph (num_clusters=8) so PE-pretrain
# operates on ~2K-node chunks. Distinct from BIG_GRAPH_DATASETS: the big
# override repartitions for memory-safe PE + Rayleigh batches, not a PE-off path.
DEFAULT_SPARC_TRAIN_OVERRIDE_PUBMED = "sparc_train_override_pubmed.json"
PE_REPARTITION_DATASETS = {"pubmed"}


def load_graphsage_adjacency(dataset_name: str, graphsage_root: Path) -> sp.csr_matrix:
    """Load the undirected adjacency matrix from GraphSAGE-formatted files."""
    dataset_dir = graphsage_root / dataset_name
    with open(dataset_dir / f"{dataset_name}-G.json", "r", encoding="utf-8") as f:
        graph_dict = json.load(f)

    with open(dataset_dir / f"{dataset_name}-id_map.json", "r", encoding="utf-8") as f:
        id_map = json.load(f)

    num_nodes = len(id_map)
    links = graph_dict["links"]
    row = np.array([int(link["source"]) for link in links], dtype=np.int64)
    col = np.array([int(link["target"]) for link in links], dtype=np.int64)
    data = np.ones(row.shape[0], dtype=np.float32)

    adj = sp.coo_matrix((data, (row, col)), shape=(
        num_nodes, num_nodes), dtype=np.float32).tocsr()
    # Make graph undirected for true neighborhood comparison and spectral embedding.
    adj = adj.maximum(adj.T)
    adj[adj > 1] = 1
    adj.eliminate_zeros()
    return adj


def load_cold_brew_embeddings(
    cold_brew_root: Path,
    dataset: str,
    split_name: str,
    seed: int,
) -> Optional[np.ndarray]:
    """
    Load Cold-Brew node embeddings produced by gnn-tail-generalization.

    Expected path:
        <cold_brew_root>/<dataset>/<split_name>/seed<seed>/part1.npz
    The npz must contain an `embeddings` array of shape (n_nodes, d).
    """
    cold_brew_path = (
        cold_brew_root / dataset / split_name / f"seed{seed}" / "part1.npz"
    )
    if not cold_brew_path.is_file():
        print(f"    [cold-brew] file not found: {cold_brew_path}")
        return None

    print(f"    [cold-brew] loading embeddings from: {cold_brew_path}")
    with np.load(cold_brew_path, allow_pickle=False) as data:
        if "embeddings" not in data.files:
            print(
                f"    [cold-brew] 'embeddings' key not found in {cold_brew_path}; "
                f"available keys: {data.files}"
            )
            return None
        return np.asarray(data["embeddings"], dtype=np.float32)


def project_embedding_to_dim(
    X: np.ndarray,
    target_dim: int,
    *,
    random_state: int = 0,
) -> np.ndarray:
    """
    Project any 2D embedding matrix to `target_dim`.

    - If X has more dims: PCA down-project.
    - If X has fewer dims: pad with zeros.
    """
    if X.ndim != 2:
        raise ValueError(f"Expected 2D feature matrix, got shape {X.shape}.")
    d = X.shape[1]
    if d == target_dim:
        return X.astype(np.float32, copy=False)

    X = X.astype(np.float32, copy=False)
    if d > target_dim:
        pca = PCA(n_components=target_dim, random_state=random_state)
        return pca.fit_transform(X).astype(np.float32, copy=False)

    pad = np.zeros((X.shape[0], target_dim - d), dtype=np.float32)
    return np.concatenate([X, pad], axis=1)


def compute_real_spectral_embedding(adj: sp.csr_matrix, dim: int) -> np.ndarray:
    """
    Bottom-`dim` eigenvectors of the symmetric normalized graph Laplacian
    ``L_sym = I - D^{-1/2} A D^{-1/2}`` (with the trivial constant
    eigenvector dropped).

    Implementation note: instead of asking ARPACK for the SMALLEST
    eigenvalues of ``L_sym`` (``eigsh(L, which='SM')``, slow), we compute
    the LARGEST eigenvalues of the normalized adjacency
    ``M = D^{-1/2} A D^{-1/2}``. Since ``L_sym = I - M``, the eigenvectors
    are identical and ``λ_L = 1 - λ_M``, so largest eigenpairs of M
    correspond exactly to smallest eigenpairs of L. ARPACK is dramatically
    faster at "LA" than "SM", typically by 10-100x on big graphs.
    """
    n = adj.shape[0]
    if n < 2:
        return np.zeros((n, dim), dtype=np.float32)

    deg = np.asarray(adj.sum(axis=1)).reshape(-1)
    with np.errstate(divide="ignore"):
        inv_sqrt_deg = 1.0 / np.sqrt(np.clip(deg, a_min=1e-12, a_max=None))
    d_inv_sqrt = sp.diags(inv_sqrt_deg)
    # Symmetric normalized adjacency. Numerical symmetrisation is cheap and
    # avoids ARPACK occasionally complaining about asymmetric input on
    # graphs with floating-point round-off.
    norm_adj = (d_inv_sqrt @ adj @ d_inv_sqrt).astype(np.float32)
    norm_adj = ((norm_adj + norm_adj.T) * 0.5).tocsr()

    k_desired = max(1, min(dim + 1, n - 1))
    ks_to_try = list(range(k_desired, max(0, k_desired - 6), -2))
    if ks_to_try[-1] != 1:
        ks_to_try.append(1)

    tol = 1e-3
    maxiter = 5000

    last_err = None
    vals_M = None
    vecs = None

    for k in ks_to_try:
        try:
            vals_M, vecs = eigsh(
                norm_adj, k=k, which="LA", tol=tol, maxiter=maxiter
            )
            break
        except Exception as e:
            last_err = e
            if hasattr(e, "eigenvalues") and hasattr(e, "eigenvectors"):
                vals_M = getattr(e, "eigenvalues")
                vecs = getattr(e, "eigenvectors")
                if vecs is not None and vecs.shape[1] > 0:
                    break

    if vecs is None or vecs.shape[1] == 0:
        raise RuntimeError(
            f"Failed to compute real spectral embedding (n={n}, dim={dim}, "
            f"k_desired={k_desired}). Last error: {last_err}"
        )

    # Largest eigenvalues of M (descending) -> smallest eigenvalues of
    # L_sym (ascending). Sort eigenvectors so column 0 is the trivial one.
    order = np.argsort(-vals_M)  # descending in M == ascending in L
    vecs = vecs[:, order]

    # Drop the trivial eigenvector (λ_M ≈ 1, i.e. λ_L ≈ 0) when possible.
    if vecs.shape[1] > 1:
        vecs = vecs[:, 1:]
    if vecs.shape[1] < dim:
        print(
            f"Padding {dim - vecs.shape[1]} zeros to real spectral embedding.")
        pad = np.zeros((n, dim - vecs.shape[1]), dtype=vecs.dtype)
        vecs = np.concatenate([vecs, pad], axis=1)
    elif vecs.shape[1] > dim:
        vecs = vecs[:, :dim]

    return vecs.astype(np.float32)


def ranked_neighbors(
    emb: np.ndarray,
    max_k: int,
    *,
    debug: bool = False,
    label: Optional[str] = None,
) -> np.ndarray:
    """
    Return ranked neighbor indices per node up to `max_k` using L2 distance.
    """
    n_nodes = emb.shape[0]
    if n_nodes <= 1 or max_k <= 0:
        return np.zeros((n_nodes, 0), dtype=np.int64)

    k_eff = min(max_k, n_nodes - 1)

    # Request extra neighbors so that, after removing the query node itself,
    # we still have at least k_eff candidates for every row.
    extra = 50
    n_neighbors = min(n_nodes, k_eff + 1 + extra)

    _t_nn = time.perf_counter()
    nn = NearestNeighbors(
        n_neighbors=n_neighbors,
        metric="euclidean",
        algorithm="brute",
    )
    nn.fit(emb)
    _t_fit = time.perf_counter()
    _, indices = nn.kneighbors(emb, return_distance=True)
    _t_kn = time.perf_counter()

    if debug:
        ar = np.arange(n_nodes)[:, None]
        self_in_candidates = (indices == ar).any(axis=1)
        self_in_count = int(self_in_candidates.sum())
        after_counts = int(((indices != ar).sum(axis=1) < k_eff).sum())
        print(
            f"    [debug topk] {label or 'embedding'} | metric=l2 | "
            f"n={n_nodes} | max_k={max_k} | k_eff={k_eff} | candidates={indices.shape[1]} | "
            f"self_in_candidates={self_in_count}/{n_nodes} | "
            f"rows_after_removal_lt_k={after_counts}"
        )

    neighbors = np.empty((n_nodes, k_eff), dtype=np.int64)
    for i in range(n_nodes):
        row = indices[i]
        row = row[row != i]
        if row.shape[0] >= k_eff:
            neighbors[i] = row[:k_eff]
        else:
            if row.shape[0] == 0:
                neighbors[i] = i
            else:
                pad = np.full(k_eff - row.shape[0], row[0], dtype=np.int64)
                padded = np.concatenate([row, pad], axis=0)[:k_eff]
                neighbors[i] = padded

    _t_post = time.perf_counter()
    print(
        f"      [timing] ranked_neighbors {label or '?'} "
        f"(n={n_nodes}, max_k={max_k}, k_eff={k_eff}, dim={emb.shape[1]}): "
        f"fit={_t_fit - _t_nn:.2f}s | kneighbors={_t_kn - _t_fit:.2f}s | "
        f"post_filter_loop={_t_post - _t_kn:.2f}s | total={_t_post - _t_nn:.2f}s"
    )

    return neighbors


def _snapshot_to_binary_csr(
    indptr_np: np.ndarray, indices_np: np.ndarray, n: int
) -> sp.csr_matrix:
    """
    Build a binary scipy CSR (int8 data, all 1s) from raw ``indptr`` /
    ``indices`` arrays and drop the diagonal (self isn't a "neighbor"
    in the eval semantics). Used as the common snapshot path for both
    the scipy and torch CUDA backends below.

    Two correctness-critical pieces here:

    * ``torch.sparse.mm`` for CSR x CSR currently returns a CSR with
      col_indices that are NOT sorted within each row. scipy's
      ``setdiag``/``eliminate_zeros`` silently malfunctions on
      unsorted-indices CSRs (it can fail to locate the diagonal
      entry and end up dropping legitimate non-diagonal nonzeros).
      We therefore force ``sort_indices`` once up-front; for
      already-sorted scipy outputs this is a near-noop.

    * We allocate fresh ``indptr`` / ``indices`` arrays before
      handing them to scipy. When the caller's arrays already point
      at memory shared with a live torch tensor (e.g.
      ``cum.col_indices().numpy()`` on a CPU tensor returns a view,
      not a copy), scipy's in-place ``sort_indices`` would otherwise
      reorder the torch tensor's column indices without reordering
      its values, silently corrupting the next iteration's matmul.
    """
    nnz_t = int(indices_np.size)
    target_idx_dtype = np.int32 if n < (1 << 31) else np.int64
    indptr_np = np.array(indptr_np, dtype=target_idx_dtype, copy=True)
    indices_np = np.array(indices_np, dtype=target_idx_dtype, copy=True)
    data_np = np.ones(nnz_t, dtype=np.int8)
    out = sp.csr_matrix((data_np, indices_np, indptr_np), shape=(n, n))
    out.sort_indices()
    out.setdiag(0)
    out.eliminate_zeros()
    return out


def _adjacency_neighbor_sets_by_hops_scipy(
    adj: sp.csr_matrix,
    hops_values: List[int],
) -> Dict[int, sp.csr_matrix]:
    """
    CPU/scipy backend for ``adjacency_neighbor_sets_by_hops``.

    Uses the ``(I + A1)``-power trick. Let ``B = I + binary(A1)`` (with
    A1's self-loops zeroed first, then I added back). Because I commutes
    with everything, ``(I + A1)^h = sum_{k=0..h} C(h,k) * A1^k``, so the
    *support* of ``B^h`` row ``i`` is exactly "nodes reachable from i in
    0..h hops". We iterate ``cum = cum @ B`` once per hop, snapshot
    (binarise + drop diagonal) at every requested hop, and never
    explicitly materialise A^h or do a separate sparse OR -- both are
    folded into the single matmul, which is the bulk of the work.

    Values stay in float32 throughout (path counts grow with h but
    stay bounded by ``(deg+1)^(h-1)``, well within float32 for every
    dataset we run) so we don't pay the O(nnz) ones-write between
    iterations that the previous int8 implementation needed to keep
    the int8 accumulator from overflowing.
    """
    max_hops = max(hops_values)
    requested_hops = set(int(h) for h in hops_values)
    n = adj.shape[0]

    A1 = adj.tocsr().copy()
    A1.data = np.ones_like(A1.data, dtype=np.float32)
    A1.setdiag(0)
    A1.eliminate_zeros()
    B = (A1 + sp.eye(n, dtype=np.float32, format="csr")).tocsr()

    def _snapshot(M: sp.csr_matrix) -> sp.csr_matrix:
        return _snapshot_to_binary_csr(M.indptr, M.indices, n)

    cum = B
    neighbors_by_h: Dict[int, sp.csr_matrix] = {}
    if 1 in requested_hops:
        neighbors_by_h[1] = _snapshot(cum)

    for hop_idx in range(2, max_hops + 1):
        cum = (cum @ B).tocsr()
        if hop_idx in requested_hops:
            neighbors_by_h[hop_idx] = _snapshot(cum)

    return neighbors_by_h


def _adjacency_neighbor_sets_by_hops_torch_cuda(
    adj: sp.csr_matrix,
    hops_values: List[int],
) -> Dict[int, sp.csr_matrix]:
    """
    GPU backend via ``torch.sparse_csr_tensor`` + ``torch.sparse.mm``
    (cuSPARSE spgemm). Same ``(I + A1)``-power algorithm as the scipy
    backend; the matmul is the only thing that runs on CUDA, every
    snapshot is converted back to a scipy CSR with int8 data so the
    downstream eval (``evaluate_predicted_neighborhoods``) is unchanged.

    On ogbn-arxiv (n=169k, hops=[1,2,3]) this drops the wall from
    ~330s on scipy single-threaded to single-digit seconds.
    """
    import torch  # local import: keeps the scipy fallback usable without torch.

    max_hops = max(hops_values)
    requested_hops = set(int(h) for h in hops_values)
    n = adj.shape[0]
    device = torch.device("cuda")
    # cuSPARSE accepts int32 indices when n fits; saves ~half the device
    # memory budget for ``col_indices`` on a ~460M-nnz matrix.
    idx_dtype = torch.int32 if n < (1 << 31) else torch.int64

    A1 = adj.tocsr().copy()
    A1.data = np.ones_like(A1.data, dtype=np.float32)
    A1.setdiag(0)
    A1.eliminate_zeros()
    B_sp = (A1 + sp.eye(n, dtype=np.float32, format="csr")).tocsr()

    crow = torch.as_tensor(B_sp.indptr, dtype=idx_dtype, device=device)
    col = torch.as_tensor(B_sp.indices, dtype=idx_dtype, device=device)
    vals = torch.as_tensor(B_sp.data, dtype=torch.float32, device=device)
    B = torch.sparse_csr_tensor(crow, col, vals, size=(n, n))
    cum = B  # alias is safe: we never mutate B in place

    def _to_scipy(M: "torch.Tensor") -> sp.csr_matrix:
        # ``.cpu().numpy()`` returns a numpy array backed by a fresh
        # CPU tensor -- safe to hand to scipy (which may overwrite it
        # in eliminate_zeros).
        return _snapshot_to_binary_csr(
            M.crow_indices().cpu().numpy(),
            M.col_indices().cpu().numpy(),
            n,
        )

    neighbors_by_h: Dict[int, sp.csr_matrix] = {}
    if 1 in requested_hops:
        neighbors_by_h[1] = _to_scipy(cum)

    for hop_idx in range(2, max_hops + 1):
        new_cum = torch.sparse.mm(cum, B)
        if cum is not B:
            del cum
        cum = new_cum
        if hop_idx in requested_hops:
            neighbors_by_h[hop_idx] = _to_scipy(cum)

    if cum is not B:
        del cum
    del B
    torch.cuda.empty_cache()

    return neighbors_by_h


def adjacency_neighbor_sets_by_hops(
    adj: sp.csr_matrix,
    hops_values: List[int],
) -> Dict[int, sp.csr_matrix]:
    """
    Build cumulative true-neighbor reachability matrices for every
    requested hop value.

    Returns a dict mapping ``h -> A_h_cum`` where ``A_h_cum`` is a
    ``(n, n)`` sparse boolean CSR matrix whose nonzero entries in row
    ``i`` are exactly the set of nodes reachable from ``i`` in 1..h
    walk-hops (self excluded, edge multiplicity discarded).

    Why a sparse matrix and not a ``List[Set[int]]``?
    The previous implementation materialised one Python ``set[int]``
    per node and per requested hop. On ogbn-arxiv (n=169_343) the
    cumulative hop-3 neighborhood averages ~2_700 nodes/node; with
    ~80 bytes per ``set`` slot that is ~35 GB *just* for the int
    objects, and the full pipeline OOM-killed the SLURM job at 128 GB
    (Neighborhood_prediction_13239158.err:31). The CSR representation
    here costs ``8 * nnz`` bytes (int32 column indices) -- e.g. ~250 MB
    for ogbn-arxiv at hop=2 vs. tens of GB of Python-int sets. The
    downstream eval (``evaluate_predicted_neighborhoods``) does
    membership tests on a per-node basis by slicing
    ``A_h_cum.indptr`` / ``A_h_cum.indices`` and building a small
    local set inside the loop only, so peak memory stays bounded.

    Backends
    --------
    Two implementations are available; both run the same algorithm
    (the ``(I + A1)``-power trick) so they are bit-equivalent up to
    floating-point reordering of the spgemm:

    * ``_adjacency_neighbor_sets_by_hops_torch_cuda`` (preferred when
      a CUDA device is visible): cuSPARSE spgemm via
      ``torch.sparse.mm``. Multi-threaded on the GPU and ~50x faster
      than scipy on dense layers (ogbn-arxiv hop=3: ~330s -> seconds).
    * ``_adjacency_neighbor_sets_by_hops_scipy``: pure scipy CPU
      fallback, used when torch / CUDA aren't available or if the
      GPU run errors out.
    """
    if not hops_values:
        raise ValueError("hops_values must be non-empty.")
    if min(hops_values) < 1:
        raise ValueError("All hop values must be >= 1.")

    use_torch_cuda = False
    try:
        import torch  # noqa: F401
        if torch.cuda.is_available():
            use_torch_cuda = True
    except ImportError:
        pass

    if use_torch_cuda:
        try:
            return _adjacency_neighbor_sets_by_hops_torch_cuda(adj, hops_values)
        except Exception as e:
            print(
                f"  [adj-by-hops] torch.sparse CUDA path failed "
                f"({type(e).__name__}: {e}); falling back to scipy."
            )

    return _adjacency_neighbor_sets_by_hops_scipy(adj, hops_values)


def evaluate_predicted_neighborhoods(
    pred_neighbors: np.ndarray,
    true_neighbors: sp.csr_matrix,
    node_indices: Optional[np.ndarray],
) -> dict:
    """
    Evaluate predicted top-k neighborhoods against the cumulative true
    h-hop neighborhoods stored as a binary sparse CSR matrix.

    ``true_neighbors`` is a (n, n) sparse boolean matrix whose row
    ``i`` lists the cumulative 1..h-hop reachability set of node ``i``
    (self excluded). For each evaluated node we slice the row's
    ``indices`` (a small numpy int64 array) and build a tiny local
    Python ``set`` -- this is O(deg_h) per node and, critically, the
    set goes out of scope each iteration so peak memory is bounded by
    the largest single-node neighborhood instead of n * avg_deg_h
    Python ints (which OOM-killed the SLURM job at 128 GB on
    ogbn-arxiv).
    """
    k = pred_neighbors.shape[1]
    if node_indices is None:
        node_indices = np.arange(pred_neighbors.shape[0], dtype=np.int64)
    else:
        node_indices = np.asarray(node_indices, dtype=np.int64)
        if node_indices.ndim != 1:
            raise ValueError("node_indices must be a 1D array of node ids.")

    if not sp.issparse(true_neighbors):
        raise TypeError(
            "true_neighbors must be a scipy.sparse matrix; got "
            f"{type(true_neighbors).__name__}."
        )
    true_csr = true_neighbors.tocsr()
    indptr = true_csr.indptr
    indices = true_csr.indices

    overlaps = []
    precisions = []
    recalls = []
    f1_scores = []
    accuracies = []
    ndcgs = []
    for node in node_indices:
        node_int = int(node)
        true_arr = indices[indptr[node_int]:indptr[node_int + 1]]
        true_size = int(true_arr.size)
        true_set = set(true_arr.tolist())

        pred_list = [int(x) for x in pred_neighbors[node_int]]
        pred_set = set(pred_list)
        overlap = sum(1 for x in pred_set if x in true_set)

        len_neighbors = min(len(pred_set), true_size)
        if len_neighbors == 0:
            accuracy = 1
        else:
            accuracy = overlap / len_neighbors
        precision = overlap / max(len(pred_set), 1)
        recall = overlap / max(true_size, 1)
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2.0 * precision * recall / (precision + recall)

        if k == 0:
            ndcg = 0.0
        else:
            dcg = 0.0
            for rank, pred in enumerate(pred_list):
                if pred in true_set:
                    dcg += 1.0 / np.log2(rank + 2.0)

            ideal_hits = min(true_size, k)
            if ideal_hits == 0:
                ndcg = 1.0
            else:
                idcg = sum(1.0 / np.log2(i + 2.0) for i in range(ideal_hits))
                ndcg = dcg / idcg if idcg > 0 else 0.0

        overlaps.append(overlap)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
        accuracies.append(accuracy)
        ndcgs.append(ndcg)
    return {
        "k": k,
        "avg_overlap": float(np.mean(overlaps)),
        "avg_precision": float(np.mean(precisions)),
        "avg_recall": float(np.mean(recalls)),
        "avg_f1": float(np.mean(f1_scores)),
        "avg_accuracy": float(np.mean(accuracies)),
        "avg_ndcg": float(np.mean(ndcgs)),
    }


def pick_result_run(dataset_results_dir: Path, run_name: Optional[str]) -> Path:
    if run_name:
        selected = dataset_results_dir / run_name
        if not selected.is_dir():
            raise FileNotFoundError(
                f"Requested run '{run_name}' not found in {dataset_results_dir}")
        return selected

    runs = sorted([p for p in dataset_results_dir.iterdir() if p.is_dir()])
    if not runs:
        raise FileNotFoundError(
            f"No run folders found inside {dataset_results_dir}")
    return runs[0]


def _format_test_ratio(test_ratio: float) -> str:
    """Replicate SPARC's run-dir formatting (`str(float)`) for the test_ratio."""
    return str(float(test_ratio))


def derive_sparc_run_name(
    split_name: str, test_ratio: float, seed: int, suffix: Optional[str]
) -> str:
    """
    Build the SPARC run-dir name that ``SPARC/src/main.py`` would produce
    for the given (split, test_ratio, seed[, suffix]) combination.
    """
    base = f"{split_name}_test{_format_test_ratio(test_ratio)}_seed{int(seed)}"
    if suffix:
        return f"{base}__{suffix}"
    return base


def train_sparc_for_dataset(
    *,
    dataset: str,
    sparc_src_dir: Path,
    sparc_results_root: Path,
    overrides_path: Path,
    overrides_path_big: Optional[Path],
    overrides_path_heterophilous: Optional[Path],
    overrides_path_pubmed: Optional[Path],
    overrides_path_tail: Optional[Path] = None,
    overrides_path_trial: Optional[Path] = None,
    split_name: str,
    test_ratio: float,
    val_ratio: float,
    seed: int,
    suffix: str,
    python_exe: str,
    retrain: bool,
) -> Path:
    """
    Train SPARC for a single dataset by invoking ``SPARC/src/main.py`` as a
    subprocess. The output is written to ``<sparc_results_root>/<dataset>/
    <split>_test<test_ratio>_seed<seed>__<suffix>`` (the ``__<suffix>`` is
    produced by SPARC via the ``SPARC_RESULT_SUFFIX`` env var).

    SPARC's ``--config_overrides`` accepts a colon-separated list of JSON
    paths and deep-merges them left-to-right. We stack dataset-class-
    specific deltas on top of the base override:

    * ``BIG_GRAPH_DATASETS`` (ogbn-arxiv, reddit, ogbn-products):
      ``overrides_path_big`` keeps PE pretraining on with tighter
      ``num_clusters`` / ``bsize`` so merged batches stay small enough for
      dense GPU affinities; leaves eigenspace diagnostics off.
    * ``HETEROPHILOUS_DATASETS`` (chameleon, squirrel, minesweeper):
      ``overrides_path_heterophilous`` cancels the base override's
      ``affinity_walk_order=1`` (heterophilous 1-hop is anti-homophilous)
      and lowers the spectral LR (the stock 1e-3 diverges).
    * ``PE_REPARTITION_DATASETS`` (pubmed): ``overrides_path_pubmed``
      bumps ``num_clusters`` so the PE-pretrain dense ``eigh`` runs on
      partition-sized matrices and fits in GPU memory while keeping PE
      pretraining ON.
    * ``overrides_path_tail`` (optional): merged after big/hetero/pubmed
      blocks (e.g. ``sparc_train_override_nb_best.json`` for stronger
      embeddings on the default random split).
    * ``overrides_path_trial`` (optional): appended last so its keys win
      over all stacked JSON files (e.g. Optuna trial deltas).

    Returns the resulting run directory.
    """
    run_name = derive_sparc_run_name(split_name, test_ratio, seed, suffix)
    run_dir = sparc_results_root / dataset / run_name
    embed_path = run_dir / "embeddings.npy"

    if embed_path.is_file() and not retrain:
        print(f"  [sparc-train] reusing existing run: {run_dir}")
        return run_dir

    if not sparc_src_dir.is_dir():
        raise FileNotFoundError(
            f"SPARC source directory not found: {sparc_src_dir}")
    if not (sparc_src_dir / "main.py").is_file():
        raise FileNotFoundError(
            f"Expected SPARC entry point at {sparc_src_dir / 'main.py'}")
    if not overrides_path.is_file():
        raise FileNotFoundError(
            f"SPARC config-overrides JSON not found: {overrides_path}")

    overrides_chain = [str(overrides_path.resolve())]
    if dataset in BIG_GRAPH_DATASETS and overrides_path_big is not None:
        if not overrides_path_big.is_file():
            raise FileNotFoundError(
                "Big-graph SPARC config-overrides JSON not found: "
                f"{overrides_path_big}"
            )
        overrides_chain.append(str(overrides_path_big.resolve()))
        print(
            f"  [sparc-train] '{dataset}' is a big-graph dataset; stacking "
            f"override {overrides_path_big.name} (PE on, smaller batches)."
        )
    if (dataset in HETEROPHILOUS_DATASETS
            and overrides_path_heterophilous is not None):
        if not overrides_path_heterophilous.is_file():
            raise FileNotFoundError(
                "Heterophilous SPARC config-overrides JSON not found: "
                f"{overrides_path_heterophilous}"
            )
        overrides_chain.append(str(overrides_path_heterophilous.resolve()))
        print(
            f"  [sparc-train] '{dataset}' is a heterophilous dataset; "
            f"stacking override {overrides_path_heterophilous.name} "
            f"(walk_order=2, lr=3e-4)."
        )
    if (dataset in PE_REPARTITION_DATASETS
            and overrides_path_pubmed is not None):
        if not overrides_path_pubmed.is_file():
            raise FileNotFoundError(
                "PE-repartition SPARC config-overrides JSON not found: "
                f"{overrides_path_pubmed}"
            )
        overrides_chain.append(str(overrides_path_pubmed.resolve()))
        print(
            f"  [sparc-train] '{dataset}' needs PE-pretrain repartitioning; "
            f"stacking override {overrides_path_pubmed.name} "
            f"(num_clusters bumped so eigh fits in GPU)."
        )
    if overrides_path_tail is not None:
        if not overrides_path_tail.is_file():
            raise FileNotFoundError(
                "Tail SPARC config-overrides JSON not found: "
                f"{overrides_path_tail}"
            )
        overrides_chain.append(str(overrides_path_tail.resolve()))
        print(
            f"  [sparc-train] stacking tail quality override "
            f"{overrides_path_tail.name}."
        )
    if overrides_path_trial is not None:
        if not overrides_path_trial.is_file():
            raise FileNotFoundError(
                "Trial / extra SPARC config-overrides JSON not found: "
                f"{overrides_path_trial}"
            )
        overrides_chain.append(str(overrides_path_trial.resolve()))
        print(
            f"  [sparc-train] stacking tail override {overrides_path_trial.name} "
            "(highest precedence in --config_overrides merge)."
        )
    overrides_arg = ":".join(overrides_chain)

    cmd = [
        python_exe,
        "-u",
        "main.py",
        "--dataset", dataset,
        "--seed", str(int(seed)),
        "--test_ratio", _format_test_ratio(test_ratio),
        "--val_ratio", _format_test_ratio(val_ratio),
        "--split_name", split_name,
        "--config_overrides", overrides_arg,
    ]
    env = dict(os.environ)
    env["SPARC_RESULT_SUFFIX"] = suffix

    print(
        f"  [sparc-train] training SPARC for '{dataset}' "
        f"(split={split_name}, test_ratio={test_ratio}, val_ratio={val_ratio}, "
        f"seed={seed}, suffix='{suffix}')"
    )
    print(f"  [sparc-train] cmd: {' '.join(cmd)}  (cwd={sparc_src_dir})")

    _t_sparc_sub = time.perf_counter()
    proc = subprocess.run(cmd, cwd=str(sparc_src_dir), env=env)
    print(
        f"  [timing] sparc-train subprocess '{dataset}' wall: "
        f"{time.perf_counter() - _t_sparc_sub:.1f}s"
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"SPARC training failed for dataset '{dataset}' "
            f"(returncode={proc.returncode})."
        )
    if not embed_path.is_file():
        raise FileNotFoundError(
            f"SPARC training did not produce {embed_path}. "
            f"Check the SPARC log above."
        )
    print(f"  [sparc-train] done: {run_dir}")
    return run_dir


def run_dataset(
    dataset: str,
    sparc_results_root: Path,
    cold_brew_root: Path,
    graphsage_root: Path,
    topk_values: List[int],
    sparc_run_name: Optional[str],
    cold_brew_split: str,
    cold_brew_seed: int,
    average_on: str,
    debug_topk: bool,
    true_hops_values: List[int],
    spectral_dim: int,
    train_sparc: bool,
    sparc_src_dir: Path,
    sparc_train_overrides_path: Path,
    sparc_train_overrides_big_path: Optional[Path],
    sparc_train_overrides_heterophilous_path: Optional[Path],
    sparc_train_overrides_pubmed_path: Optional[Path],
    sparc_train_overrides_tail_path: Optional[Path],
    sparc_train_split: str,
    sparc_train_test_ratio: float,
    sparc_train_val_ratio: float,
    sparc_train_seed: int,
    sparc_train_suffix: str,
    sparc_python: str,
    retrain_sparc: bool,
    project_to_sparc_dim: bool,
):
    _t_dataset = time.perf_counter()
    print(f"[timing] --- dataset={dataset} --- start")

    if train_sparc:
        # Retrain SPARC up-front so the kNN below uses the new embeddings.
        # The resulting run-dir name is fully determined by training params,
        # which overrides any manually passed `--sparc_run_name`.
        _t = time.perf_counter()
        sparc_run_dir = train_sparc_for_dataset(
            dataset=dataset,
            sparc_src_dir=sparc_src_dir,
            sparc_results_root=sparc_results_root,
            overrides_path=sparc_train_overrides_path,
            overrides_path_big=sparc_train_overrides_big_path,
            overrides_path_heterophilous=(
                sparc_train_overrides_heterophilous_path),
            overrides_path_pubmed=sparc_train_overrides_pubmed_path,
            overrides_path_tail=sparc_train_overrides_tail_path,
            split_name=sparc_train_split,
            test_ratio=sparc_train_test_ratio,
            val_ratio=sparc_train_val_ratio,
            seed=sparc_train_seed,
            suffix=sparc_train_suffix,
            python_exe=sparc_python,
            retrain=retrain_sparc,
        )
        print(
            f"[timing] train_sparc_for_dataset (incl. reuse check): "
            f"{time.perf_counter() - _t:.1f}s"
        )
    else:
        sparc_dataset_dir = sparc_results_root / dataset
        if not sparc_dataset_dir.is_dir():
            print(
                f"[SKIP] {dataset}: missing SPARC results dir {sparc_dataset_dir}")
            return
        sparc_run_dir = pick_result_run(sparc_dataset_dir, sparc_run_name)

    sparc_embed_path = sparc_run_dir / "embeddings.npy"
    features_path = sparc_run_dir / "features.npy"
    if not sparc_embed_path.exists() or not features_path.exists():
        print(
            f"[SKIP] {dataset}: expected files not found in SPARC run "
            f"(features={features_path}, sparc={sparc_embed_path})"
        )
        return

    _t = time.perf_counter()
    sparc_embeddings = np.load(sparc_embed_path).astype(np.float32, copy=False)
    node_features = np.load(features_path)
    print(
        f"[timing] load embeddings.npy + features.npy: "
        f"{time.perf_counter() - _t:.2f}s"
    )
    if node_features.ndim != 2 or sparc_embeddings.ndim != 2:
        print(
            f"[SKIP] {dataset}: invalid embedding array shape(s) in {sparc_run_dir}")
        return

    _t = time.perf_counter()
    adj = load_graphsage_adjacency(dataset, graphsage_root)
    print(
        f"[timing] load_graphsage_adjacency: {time.perf_counter() - _t:.2f}s"
    )
    n_nodes = adj.shape[0]
    if node_features.shape[0] != n_nodes or sparc_embeddings.shape[0] != n_nodes:
        print(
            f"[SKIP] {dataset}: node count mismatch "
            f"(adj={n_nodes}, features={node_features.shape[0]}, sparc={sparc_embeddings.shape[0]})"
        )
        return

    # SPARC stays at its native dim. Cold-Brew and node-features stay at
    # their native dims by default; pass --project_to_sparc_dim to PCA-
    # project them down to SPARC's dim instead. The spectral embedding
    # uses its own fixed dim (default 128), which can be further capped
    # per dataset via PER_DATASET_SPECTRAL_DIM_CAP (e.g. 64 for
    # ogbn-products).
    sparc_dim = sparc_embeddings.shape[1]
    requested_spectral_dim = int(spectral_dim)
    eff_spectral_dim = requested_spectral_dim
    dataset_cap = PER_DATASET_SPECTRAL_DIM_CAP.get(dataset)
    if dataset_cap is not None and eff_spectral_dim > int(dataset_cap):
        print(
            f"  [spectral] requested dim={requested_spectral_dim} capped to "
            f"{int(dataset_cap)} for dataset '{dataset}'."
        )
        eff_spectral_dim = int(dataset_cap)
    eff_spectral_dim = max(1, min(eff_spectral_dim, n_nodes - 1))
    if eff_spectral_dim != requested_spectral_dim and dataset_cap is None:
        print(
            f"  [spectral] requested dim={requested_spectral_dim} clipped to "
            f"{eff_spectral_dim} (n_nodes={n_nodes})."
        )

    _t = time.perf_counter()
    real_spectral = compute_real_spectral_embedding(adj, eff_spectral_dim)
    print(
        f"[timing] compute_real_spectral_embedding "
        f"(n={n_nodes}, dim={eff_spectral_dim}): "
        f"{time.perf_counter() - _t:.2f}s"
    )

    _t = time.perf_counter()
    if project_to_sparc_dim:
        node_features_proj = project_embedding_to_dim(node_features, sparc_dim)
    else:
        node_features_proj = node_features.astype(np.float32, copy=False)
    print(
        f"[timing] node feature prep (project={project_to_sparc_dim}): "
        f"{time.perf_counter() - _t:.2f}s"
    )

    # Cold-Brew is mandatory: always evaluate over all four embeddings.
    cold_brew_embeddings_raw: Optional[np.ndarray] = None
    _t = time.perf_counter()
    try:
        cold_brew_embeddings_raw = load_cold_brew_embeddings(
            cold_brew_root, dataset, cold_brew_split, cold_brew_seed
        )
    except Exception as exc:
        print(f"  [cold-brew] load failed for {dataset}: {exc}")
        cold_brew_embeddings_raw = None
    print(
        f"[timing] cold_brew load attempt: {time.perf_counter() - _t:.2f}s"
    )

    if cold_brew_embeddings_raw is None:
        print(
            f"[SKIP] {dataset}: Cold-Brew embeddings missing; cannot run all 4 spaces."
        )
        return
    if cold_brew_embeddings_raw.shape[0] != n_nodes:
        print(
            f"[SKIP] {dataset}: Cold-Brew node count mismatch "
            f"(adj={n_nodes}, cold_brew={cold_brew_embeddings_raw.shape[0]})"
        )
        return
    _t = time.perf_counter()
    if project_to_sparc_dim:
        cold_brew_embeddings = project_embedding_to_dim(
            cold_brew_embeddings_raw, sparc_dim
        )
    else:
        cold_brew_embeddings = cold_brew_embeddings_raw.astype(
            np.float32, copy=False
        )
    print(
        f"[timing] cold_brew projection/prepare: "
        f"{time.perf_counter() - _t:.2f}s"
    )

    # Apply per-dataset cap on the maximum requested true-hop value.
    # Without this the cumulative reachability matrix at hop>=2 OOMs on
    # ogbn-products (2.4M nodes; A^2 has billions of nonzeros).
    hops_cap = PER_DATASET_TRUE_HOPS_CAP.get(dataset)
    if hops_cap is not None:
        capped_hops = sorted({h for h in true_hops_values if h <= int(hops_cap)})
        if capped_hops != list(true_hops_values):
            print(
                f"  [eval] capping --true_hops_values {list(true_hops_values)} "
                f"to {capped_hops} for '{dataset}' "
                f"(per PER_DATASET_TRUE_HOPS_CAP, max hop = {int(hops_cap)})."
            )
            true_hops_values = capped_hops
        if not true_hops_values:
            print(
                f"[SKIP] {dataset}: PER_DATASET_TRUE_HOPS_CAP capped all "
                f"requested hops away; nothing to evaluate."
            )
            return

    _t = time.perf_counter()
    true_neighbors_by_hops = adjacency_neighbor_sets_by_hops(
        adj, true_hops_values)
    print(
        f"[timing] adjacency_neighbor_sets_by_hops "
        f"(hops={true_hops_values}, n={n_nodes}, nnz_adj={adj.nnz}): "
        f"{time.perf_counter() - _t:.2f}s"
    )

    # Use train/test masks saved by SPARC.
    train_mask_path = sparc_run_dir / "train_mask.npy"
    test_mask_path = sparc_run_dir / "test_mask.npy"
    if average_on in ("train", "test") and (
        not train_mask_path.exists() or not test_mask_path.exists()
    ):
        raise FileNotFoundError(
            f"Missing train/test masks in {sparc_run_dir}. "
            "Expected 'train_mask.npy' and 'test_mask.npy'."
        )

    if average_on == "all":
        node_indices = np.arange(n_nodes, dtype=np.int64)
    elif average_on == "train":
        train_mask = np.load(train_mask_path)
        node_indices = np.where(train_mask.astype(bool))[0].astype(np.int64)
    elif average_on == "test":
        test_mask = np.load(test_mask_path)
        node_indices = np.where(test_mask.astype(bool))[0].astype(np.int64)
    else:
        raise ValueError(
            f"Unknown average_on='{average_on}' (expected all/train/test).")

    eval_items = [
        ("real_spectral", real_spectral),
        ("sparc_embeddings", sparc_embeddings),
        ("cold_brew", cold_brew_embeddings),
        ("node_features", node_features_proj),
    ]

    max_topk = max(topk_values)
    ranked_neighbors_by_emb: Dict[str, np.ndarray] = {}
    _t_knn_all = time.perf_counter()
    for name, emb in eval_items:
        ranked_neighbors_by_emb[name] = ranked_neighbors(
            emb,
            max_k=max_topk,
            debug=debug_topk,
            label=name,
        )
    print(
        f"[timing] ranked_neighbors (all 4 spaces): "
        f"{time.perf_counter() - _t_knn_all:.2f}s"
    )

    print(
        f"\nDataset: {dataset} | sparc_run: {sparc_run_dir.name} "
        f"| cold_brew: {cold_brew_split}/seed{cold_brew_seed} "
        f"| nodes: {n_nodes} | sparc_dim={sparc_dim} "
        f"| cold_brew_dim={cold_brew_embeddings.shape[1]} "
        f"| feat_dim={node_features_proj.shape[1]} "
        f"| spectral_dim={eff_spectral_dim} "
        f"| project_to_sparc_dim={project_to_sparc_dim} "
        f"| metric={DISTANCE_METRIC}"
    )
    print(f"  Average on: {average_on} (|nodes|={len(node_indices)})")
    print(f"  Similarity metric: {DISTANCE_METRIC}")
    _t_eval_grid = time.perf_counter()
    for true_hops in true_hops_values:
        true_neighbors = true_neighbors_by_hops[true_hops]
        for topk in topk_values:
            metrics_by_embedding = {}
            for name, _emb in eval_items:
                ranked = ranked_neighbors_by_emb[name]
                predicted = ranked[:, :min(topk, ranked.shape[1])]
                metrics = evaluate_predicted_neighborhoods(
                    predicted, true_neighbors, node_indices=node_indices
                )
                metrics_by_embedding[name] = metrics
                print(
                    f"    - {name:16s} topk={topk:4d} true_hops={true_hops} "
                    f"overlap={metrics['avg_overlap']:.4f} "
                    f"precision={metrics['avg_precision']:.4f} "
                    f"recall={metrics['avg_recall']:.4f} "
                    f"f1={metrics['avg_f1']:.4f} "
                    f"ndcg={metrics['avg_ndcg']:.4f} "
                    f"accuracy={metrics['avg_accuracy']:.4f}"
                )

            wanted = ["real_spectral", "sparc_embeddings",
                      "cold_brew", "node_features"]
            if all(k in metrics_by_embedding for k in wanted):
                # Pass if EITHER:
                #   spectral > sparc > cold_brew > features
                #   spectral > sparc > features  > cold_brew
                # i.e. spectral > sparc AND sparc beats both Cold-Brew and
                # raw features (their relative order doesn't matter).
                acc_spec = metrics_by_embedding["real_spectral"]["avg_accuracy"]
                acc_sparc = metrics_by_embedding["sparc_embeddings"]["avg_accuracy"]
                acc_cb = metrics_by_embedding["cold_brew"]["avg_accuracy"]
                acc_feat = metrics_by_embedding["node_features"]["avg_accuracy"]
                order_pass = (
                    acc_spec > acc_sparc
                    and acc_sparc > acc_cb
                    and acc_sparc > acc_feat
                )
                print(
                    "    - ranking_target "
                    f"topk={topk:4d} true_hops={true_hops} "
                    f"(spectral>sparc>{{cold_brew,features}} by accuracy): "
                    f"{'PASS' if order_pass else 'FAIL'}"
                )

    print(
        f"[timing] evaluation grid (topk×hops×metrics): "
        f"{time.perf_counter() - _t_eval_grid:.2f}s"
    )
    print(
        f"[timing] --- dataset={dataset} total wall: "
        f"{time.perf_counter() - _t_dataset:.1f}s ---"
    )


def parse_args():
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    default_sparc_results = project_root / "SPARC" / "sparc_results"
    default_cold_brew_root = (
        project_root / "gnn-tail-generalization" / "embeddings"
    )
    default_graphsage = project_root / "data" / "graphsage"
    default_sparc_src_dir = project_root / "SPARC" / "src"
    default_sparc_train_override = script_dir / DEFAULT_SPARC_TRAIN_OVERRIDE
    default_sparc_train_override_big = (
        script_dir / DEFAULT_SPARC_TRAIN_OVERRIDE_BIG
    )
    default_sparc_train_override_heterophilous = (
        script_dir / DEFAULT_SPARC_TRAIN_OVERRIDE_HETEROPHILOUS
    )
    default_sparc_train_override_pubmed = (
        script_dir / DEFAULT_SPARC_TRAIN_OVERRIDE_PUBMED
    )

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate predicted neighborhoods over four embedding spaces: "
            "real spectral, SPARC, Cold-Brew, and raw features. "
            "All retrieval uses L2 distance and a unified embedding dim equal to SPARC's."
        )
    )
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument(
        "--topk_values",
        nargs="+",
        type=int,
        default=DEFAULT_TOPK_VALUES,
        help="Grid of top-k values for kNN retrieval (default: 2 5 10 20 30).",
    )
    parser.add_argument(
        "--true_hops_values",
        nargs="+",
        type=int,
        default=DEFAULT_TRUE_HOPS_VALUES,
        help="Grid of cumulative hop neighborhoods for ground truth (default: 1 2 3).",
    )
    parser.add_argument(
        "--sparc_results_root",
        type=Path,
        default=default_sparc_results,
        help="Root containing per-dataset SPARC run folders (default: SPARC_project/SPARC/sparc_results).",
    )
    parser.add_argument(
        "--cold_brew_root",
        type=Path,
        default=default_cold_brew_root,
        help=(
            "Root containing Cold-Brew embeddings "
            "(default: SPARC_project/gnn-tail-generalization/embeddings)."
        ),
    )
    parser.add_argument(
        "--graphsage_root",
        type=Path,
        default=default_graphsage,
        help="Root with GraphSAGE-format graph data (default: SPARC_project/data/graphsage).",
    )
    parser.add_argument(
        "--sparc_run_name",
        type=str,
        default=DEFAULT_SPARC_RUN_NAME,
        help="SPARC run folder name inside sparc_results/<dataset> (default: random_test0.1_seed42).",
    )
    parser.add_argument(
        "--cold_brew_split",
        type=str,
        default=DEFAULT_COLD_BREW_SPLIT,
        help="Cold-Brew split-name folder (default: random).",
    )
    parser.add_argument(
        "--cold_brew_seed",
        type=int,
        default=DEFAULT_COLD_BREW_SEED,
        help="Cold-Brew seed (default: 42).",
    )
    parser.add_argument(
        "--average_on",
        choices=["all", "train", "test"],
        default="train",
        help="Which nodes to average metrics over.",
    )
    parser.add_argument(
        "--spectral_dim",
        type=int,
        default=DEFAULT_SPECTRAL_DIM,
        help=(
            "Dimension used to compute the real spectral embedding "
            "(independent of SPARC's dim). Default: 250."
        ),
    )
    parser.add_argument(
        "--debug_topk",
        action="store_true",
        help="Print debug info about top-k neighbor selection.",
    )

    # ------------------------------------------------------------------
    # Whether to PCA-project Cold-Brew and node features down to SPARC's
    # native dim. By default each space keeps its native dimension and
    # the L2 kNN is computed independently per space.
    # ------------------------------------------------------------------
    proj_group = parser.add_mutually_exclusive_group()
    proj_group.add_argument(
        "--project_to_sparc_dim",
        dest="project_to_sparc_dim",
        action="store_true",
        default=False,
        help=(
            "PCA-project node features and Cold-Brew embeddings down to "
            "SPARC's native dim before kNN. Default: off (each space keeps "
            "its native dim)."
        ),
    )
    proj_group.add_argument(
        "--no_project_to_sparc_dim",
        dest="project_to_sparc_dim",
        action="store_false",
        help="Disable projection (default).",
    )

    # ------------------------------------------------------------------
    # SPARC retraining knobs. By default, SPARC is trained for each
    # dataset *before* its kNN is computed, with task_aware off (lambda=0),
    # output_dim=64 and spectral-PE pretraining ON. The output is written
    # to ``sparc_results/<dataset>/<split>_test<r>_seed<s>__<suffix>``,
    # which is then used as the SPARC run dir for kNN computation.
    # ------------------------------------------------------------------
    train_group = parser.add_mutually_exclusive_group()
    train_group.add_argument(
        "--train_sparc",
        dest="train_sparc",
        action="store_true",
        default=True,
        help="Retrain SPARC for each dataset before computing the SPARC kNN (default).",
    )
    train_group.add_argument(
        "--no_train_sparc",
        dest="train_sparc",
        action="store_false",
        help="Skip SPARC retraining and use --sparc_run_name instead.",
    )
    parser.add_argument(
        "--retrain_sparc",
        action="store_true",
        help=(
            "Force re-running SPARC training even if the resolved run dir "
            "already contains embeddings.npy."
        ),
    )
    parser.add_argument(
        "--sparc_src_dir",
        type=Path,
        default=default_sparc_src_dir,
        help="Path to SPARC/src (default: SPARC_project/SPARC/src).",
    )
    parser.add_argument(
        "--sparc_train_override",
        type=Path,
        default=default_sparc_train_override,
        help=(
            "JSON file deep-merged into the dataset's SPARC config for retraining. "
            f"Default: {DEFAULT_SPARC_TRAIN_OVERRIDE} (task_aware off, "
            "output_dim=64, pe_pretrain on)."
        ),
    )
    parser.add_argument(
        "--sparc_train_override_big",
        type=Path,
        default=default_sparc_train_override_big,
        help=(
            "Extra JSON stacked on top of --sparc_train_override for "
            f"big-graph datasets ({sorted(BIG_GRAPH_DATASETS)}). "
            f"Default: {DEFAULT_SPARC_TRAIN_OVERRIDE_BIG} (PE on, "
            "num_clusters/bsize tuned to reduce dense-batch OOM risk)."
        ),
    )
    parser.add_argument(
        "--sparc_train_override_heterophilous",
        type=Path,
        default=default_sparc_train_override_heterophilous,
        help=(
            "Extra JSON stacked on top of --sparc_train_override for "
            f"heterophilous datasets ({sorted(HETEROPHILOUS_DATASETS)}). "
            f"Default: {DEFAULT_SPARC_TRAIN_OVERRIDE_HETEROPHILOUS} "
            "(cancels affinity_walk_order=1 and lowers spectral.lr; "
            "fixes the chameleon/squirrel/minesweeper Rayleigh-loss "
            "divergence at lr=1e-3)."
        ),
    )
    parser.add_argument(
        "--sparc_train_override_pubmed",
        type=Path,
        default=default_sparc_train_override_pubmed,
        help=(
            "Extra JSON stacked on top of --sparc_train_override for "
            f"datasets in PE_REPARTITION_DATASETS ({sorted(PE_REPARTITION_DATASETS)}). "
            f"Default: {DEFAULT_SPARC_TRAIN_OVERRIDE_PUBMED} (bumps "
            "num_clusters so the PE-pretrain dense eigh fits in GPU "
            "memory while keeping PE pretraining ON)."
        ),
    )
    parser.add_argument(
        "--sparc_train_override_tail",
        type=Path,
        default=None,
        help=(
            "Optional JSON merged after big/hetero/pubmed overrides and "
            "before any Optuna trial file. Use e.g. "
            "sparc_train_override_nb_best.json together with "
            "--sparc_train_suffix nb_best_v1 for stronger SPARC embeddings "
            "on random_test0.1_seed42 (see run_neighborhood_prediction_best_quality.sh)."
        ),
    )
    parser.add_argument(
        "--sparc_train_split",
        type=str,
        default=DEFAULT_SPARC_TRAIN_SPLIT,
        help="--split_name forwarded to SPARC training (default: random).",
    )
    parser.add_argument(
        "--sparc_train_test_ratio",
        type=float,
        default=DEFAULT_SPARC_TRAIN_TEST_RATIO,
        help="--test_ratio forwarded to SPARC training (default: 0.1).",
    )
    parser.add_argument(
        "--sparc_train_val_ratio",
        type=float,
        default=DEFAULT_SPARC_TRAIN_VAL_RATIO,
        help="--val_ratio forwarded to SPARC training (default: 0.03).",
    )
    parser.add_argument(
        "--sparc_train_seed",
        type=int,
        default=DEFAULT_SPARC_TRAIN_SEED,
        help="--seed forwarded to SPARC training (default: 42).",
    )
    parser.add_argument(
        "--sparc_train_suffix",
        type=str,
        default=DEFAULT_SPARC_TRAIN_SUFFIX,
        help=(
            "Suffix appended to the SPARC run-dir name "
            f"(default: '{DEFAULT_SPARC_TRAIN_SUFFIX}')."
        ),
    )
    parser.add_argument(
        "--sparc_python",
        type=str,
        default=sys.executable,
        help="Python executable used to invoke SPARC training (default: current interpreter).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    topk_values = sorted(set(int(k) for k in args.topk_values if int(k) > 0))
    true_hops_values = sorted(
        set(int(h) for h in args.true_hops_values if int(h) > 0))
    if not topk_values:
        raise ValueError("Provide at least one positive --topk_values entry.")
    if not true_hops_values:
        raise ValueError(
            "Provide at least one positive --true_hops_values entry.")
    if int(args.spectral_dim) <= 0:
        raise ValueError("--spectral_dim must be a positive integer.")

    if args.train_sparc:
        if float(args.sparc_train_test_ratio) <= 0.0:
            raise ValueError("--sparc_train_test_ratio must be positive.")
        if float(args.sparc_train_val_ratio) < 0.0:
            raise ValueError("--sparc_train_val_ratio must be non-negative.")

    for dataset in args.datasets:
        _t_one = time.perf_counter()
        try:
            run_dataset(
                dataset=dataset,
                sparc_results_root=args.sparc_results_root,
                cold_brew_root=args.cold_brew_root,
                graphsage_root=args.graphsage_root,
                topk_values=topk_values,
                sparc_run_name=args.sparc_run_name,
                cold_brew_split=args.cold_brew_split,
                cold_brew_seed=args.cold_brew_seed,
                average_on=args.average_on,
                debug_topk=args.debug_topk,
                true_hops_values=true_hops_values,
                spectral_dim=int(args.spectral_dim),
                train_sparc=bool(args.train_sparc),
                sparc_src_dir=args.sparc_src_dir,
                sparc_train_overrides_path=args.sparc_train_override,
                sparc_train_overrides_big_path=args.sparc_train_override_big,
                sparc_train_overrides_heterophilous_path=(
                    args.sparc_train_override_heterophilous),
                sparc_train_overrides_pubmed_path=(
                    args.sparc_train_override_pubmed),
                sparc_train_overrides_tail_path=args.sparc_train_override_tail,
                sparc_train_split=args.sparc_train_split,
                sparc_train_test_ratio=float(args.sparc_train_test_ratio),
                sparc_train_val_ratio=float(args.sparc_train_val_ratio),
                sparc_train_seed=int(args.sparc_train_seed),
                sparc_train_suffix=args.sparc_train_suffix,
                sparc_python=args.sparc_python,
                retrain_sparc=bool(args.retrain_sparc),
                project_to_sparc_dim=bool(args.project_to_sparc_dim),
            )
        except Exception as exc:
            print(f"[ERROR] {dataset}: {exc}")
        print(
            f"[timing] main loop iteration '{dataset}' wall "
            f"(incl. exceptions): {time.perf_counter() - _t_one:.1f}s"
        )


if __name__ == "__main__":
    main()
