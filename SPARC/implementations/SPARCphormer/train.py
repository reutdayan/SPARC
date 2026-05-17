import sys

import time
import utils
import random
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import glob
from early_stop import EarlyStopping, Stop_args
from model import TransformerModel
from lr import PolynomialDecayLR
import os.path
import torch.utils.data as Data
import argparse
import scipy.sparse as sp
from scipy.linalg import svd
from sklearn.metrics import roc_auc_score


def numpy_int_labels_to_torch(arr, device):
    """Integer node labels -> ``torch.long`` on ``device``.

    Avoids ``torch.tensor(numpy_ndarray)`` dtype inference, which fails when
    PyTorch was built against NumPy 1.x but the env has NumPy 2.x (broken
    ``_ARRAY_API``). Prefer ``pip install 'numpy<2'`` or a newer PyTorch build
    for a full fix; this path stays compatible via the buffer protocol.
    """
    arr = np.asarray(arr, dtype=np.int64).ravel()
    if arr.size == 0:
        return torch.empty(0, dtype=torch.long, device=device)
    arr = np.ascontiguousarray(arr)
    try:
        t = torch.frombuffer(arr, dtype=torch.int64, count=arr.size)
    except (TypeError, RuntimeError, ValueError):
        t = torch.frombuffer(
            memoryview(arr.tobytes(order="C")),
            dtype=torch.int64,
            count=arr.size,
        )
    return t.to(device)


# Training settings
def parse_args():
    """
    Generate a parameters parser.
    """
    # parse parameters
    parser = argparse.ArgumentParser()

    # main parameters
    parser.add_argument('--name', type=str, default=None)
    parser.add_argument('--dataset', type=str, default='pubmed',
                        help='Choose from {pubmed}')
    parser.add_argument('--device', type=int, default=0,
                        help='Device cuda id (clamped to available GPUs; '
                             'use -1 to force CPU)')
    parser.add_argument('--seed', type=int, default=3407,
                        help='Random seed.')

    # model parameters
    parser.add_argument('--hops', type=int, default=7,
                        help='Hop of neighbors to be calculated')
    parser.add_argument('--hidden_dim', type=int, default=512,
                        help='Hidden layer size')
    parser.add_argument('--ffn_dim', type=int, default=64,
                        help='FFN layer size')
    parser.add_argument('--n_layers', type=int, default=1,
                        help='Number of Transformer layers')
    parser.add_argument('--pe_dim', type=int, default=250,
                        help='Positional encoding dimension')
    parser.add_argument('--n_heads', type=int, default=8,
                        help='Number of Transformer heads')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout')
    parser.add_argument('--attention_dropout', type=float, default=0.1,
                        help='Dropout in the attention layer')

    # training parameters
    parser.add_argument('--batch_size', type=int, default=1000,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=500,
                        help='Number of epochs to train.')
    parser.add_argument('--tot_updates',  type=int, default=1000,
                        help='used for optimizer learning rate scheduling')
    parser.add_argument('--warmup_updates', type=int, default=400,
                        help='warmup steps')
    parser.add_argument('--peak_lr', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--end_lr', type=float, default=0.0001,
                        help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.00001,
                        help='weight decay')
    parser.add_argument('--patience', type=int, default=50,
                        help='Patience for early stopping')

    parser.add_argument('--space', type=str, default='spectral',
                        help="Retrieval/token space: spectral, features, computed, "
                             "computed_symmetric_multihop_laplace, real_graph, "
                             "hops, cold-brew, fusion")
    parser.add_argument('--multihop_walk_order', type=int, default=3,
                        help='Walk order p used by --space=computed_symmetric_multihop_laplace. '
                             'The Laplacian is built from W + W^2 + ... + W^p (where W is the '
                             'self-loop-free symmetric adjacency), exactly mirroring '
                             'SpectralTrainer._get_valid_affinity_matrix. Set this to match the '
                             'spectral.affinity_walk_order in the dataset config that produced '
                             'the SPARC embeddings (e.g. 3 for cora, 2 for citeseer/pubmed/wikics, '
                             '1 for reddit). Default 3 matches SpectralTrainer\'s default.')
    parser.add_argument('--feature_concat', type=str, default='spectral',
                        choices=['spectral', 'X_embedded', 'none'],
                        help='What to concatenate with raw features: spectral encoding, X_embedded, or nothing')
    parser.add_argument('--hop_avg_style', type=str, default='exponential',
                        choices=['exponential', 'linear'],
                        help='How to determine the number of neighbors per hop (exponential=powers of 2, linear=evenly spaced)')
    parser.add_argument('--knn_metric', type=str, default='minkowski',
                        help='Distance metric for KNN (e.g. minkowski, cosine, manhattan)')

    # Cold-start split parameters (must match the SPARC main.py run that produced the embeddings)
    parser.add_argument('--split_name', type=str, default='random',
                        help='Split strategy used by SPARC main.py (random | low_degree | high_degree)')
    parser.add_argument('--test_ratio', type=float, default=0.10,
                        help='Test ratio used by SPARC main.py / make_cold_start_split.')
    parser.add_argument('--val_ratio', type=float, default=0.03,
                        help='Validation ratio; must match SPARC main.py / '
                             'make_cold_start_split (SPARC/src/main.py DEFAULT_VAL_RATIO).')
    parser.add_argument('--sparc_seed', type=int, default=42,
                        help='Seed used by SPARC main.py for data splitting. '
                             'Defaults to --seed when not provided.')
    parser.add_argument('--sparc_result_suffix', type=str, default='',
                        help=('Optional SPARC main.py result-directory tag. Must match '
                              'how that run was saved: legacy '
                              '<split>_test<r>_seed<s>__<suffix> (SPARC_RESULT_SUFFIX or '
                              '--result_suffix without a leading underscore), or '
                              '<split>_test<r>_seed<s><suffix> when suffix starts with '
                              '"_" (e.g. --result_suffix _test_binary in SPARC main.py '
                              '→ ...seed42_test_binary). Empty = base dir only.'))
    parser.add_argument('--cold_brew_root', type=str, default=None,
                        help='Root directory for Cold-Brew embeddings (.npz); defaults to <repo>/gnn-tail-generalization')
    parser.add_argument('--cold_brew_split', type=str, default='random',
                        help='Cold-Brew split name used in path: embeddings/<dataset>/<split>/<run_tag>/part1.npz')
    parser.add_argument('--cold_brew_seed', type=int, default=42,
                        help='Cold-Brew seed used in <run_tag>: ..._seed<seed>/part1.npz')
    parser.add_argument('--cold_brew_subdir', type=str, default='',
                        help=('Explicit Cold-Brew subdirectory tag, e.g. '
                              '"test0.1_val0.1_arch2layer_topk3_dpMLP0.5_nl2_h64_se000_seed42". '
                              'When set, takes precedence over the field-by-field '
                              '--cold_brew_* args and is used verbatim as '
                              'embeddings/<dataset>/<split>/<this>/part1.npz.'))
    parser.add_argument('--cold_brew_test_ratio', type=float, default=None,
                        help='Cold-Brew test_ratio in the saved subdir. Defaults to --test_ratio.')
    parser.add_argument('--cold_brew_val_ratio', type=float, default=None,
                        help='Cold-Brew val_ratio in the saved subdir. Defaults to --val_ratio.')
    parser.add_argument('--cold_brew_part1_arch', type=str, default='2layer',
                        help='Cold-Brew SEMLP part-1 architecture string used in the subdir.')
    parser.add_argument('--cold_brew_topk', type=int, default=3,
                        help='Cold-Brew SEMLP top-K replacement used in the subdir.')
    parser.add_argument('--cold_brew_dropout_mlp', type=float, default=0.5,
                        help='Cold-Brew SEMLP MLP dropout used in the subdir.')
    parser.add_argument('--cold_brew_num_layers', type=int, default=None,
                        help='Cold-Brew TeacherGNN num_layers in the subdir. Required unless --cold_brew_subdir is given.')
    parser.add_argument('--cold_brew_dim_hidden', type=int, default=None,
                        help='Cold-Brew TeacherGNN dim_hidden in the subdir. Required unless --cold_brew_subdir is given.')
    parser.add_argument('--cold_brew_whether_has_se', type=str, default='000',
                        help='Cold-Brew TeacherGNN whetherHasSE flag (e.g. "000") in the subdir.')

    return parser.parse_args()


def convert_adj_matrix(adjacency_matrix):
    # Convert the sparse matrix to COO format and make sure it's of type float32
    adjacency_matrix_coo = adjacency_matrix.tocoo().astype(np.float32)
    # Convert the SciPy sparse COO matrix to a PyTorch sparse tensor
    row = torch.tensor(adjacency_matrix_coo.row, dtype=torch.long)
    col = torch.tensor(adjacency_matrix_coo.col, dtype=torch.long)
    value = torch.tensor(adjacency_matrix_coo.data,
                         dtype=torch.float32)  # Convert to float32
    # Create the PyTorch sparse tensor
    adjacency_matrix_sparse = torch.sparse_coo_tensor(
        torch.stack([row, col]), value, adjacency_matrix_coo.shape)
    adjacency_matrix = adjacency_matrix + sp.eye(adjacency_matrix.shape[0])
    D1 = np.array(adjacency_matrix.sum(axis=1))**(-0.5)
    D2 = np.array(adjacency_matrix.sum(axis=0))**(-0.5)
    D1 = sp.diags(D1[:, 0], format='csr')
    D2 = sp.diags(D2[0, :], format='csr')
    A = adjacency_matrix.dot(D1)
    A = D2.dot(A)

    adj = A
    # adj = utils.sparse_mx_to_torch_sparse_tensor(adj)
    return adj


def load_data(dataset, split_name="random", test_ratio=0.03, sparc_seed=123,
              feature_concat="spectral", result_suffix="", load_adj=True):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_results_root = os.path.join(
        script_dir, '..', '..', 'sparc_results', dataset)
    dataset_results_root = os.path.normpath(dataset_results_root)
    base = "{}_test{}_seed{}".format(split_name, test_ratio, sparc_seed)
    suffix = (result_suffix or "").strip()
    preferred_candidates = [os.path.join(dataset_results_root, base)]
    if suffix:
        # Order matches SPARC/src/main.py: ``--result_suffix`` starting with ``_``
        # appends a single segment (...seed42_test_binary); legacy uses ``__suffix``.
        if suffix.startswith("_"):
            preferred_candidates.insert(
                0, os.path.join(dataset_results_root, base + suffix))
            preferred_candidates.insert(
                1, os.path.join(dataset_results_root,
                                "{}__{}".format(base, suffix)))
        else:
            preferred_candidates.insert(
                0, os.path.join(dataset_results_root,
                                "{}__{}".format(base, suffix)))

    preferred_directory = None
    for cand in preferred_candidates:
        if os.path.isdir(cand):
            preferred_directory = cand
            break

    if preferred_directory is not None:
        directory = preferred_directory
    else:
        preferred_directory = preferred_candidates[0]
        existing_options = sorted(
            glob.glob(os.path.join(dataset_results_root, "*")))
        existing_options = [
            path for path in existing_options if os.path.isdir(path)]
        if not existing_options:
            raise FileNotFoundError(
                "No SPARC results found under {} and preferred directory {} does not exist.".format(
                    dataset_results_root, preferred_directory
                )
            )
        directory = existing_options[0]
        print(
            "Preferred SPARC results directory not found; using first existing option:",
            directory
        )

    directory = directory + os.sep
    print("Loading SPARC results from:", directory)
    spectral_encoding = np.load(os.path.join(directory, 'embeddings.npy'))
    features = np.load(os.path.join(directory, 'features.npy'))
    labels = np.load(os.path.join(directory, 'labels.npy'))
    train_mask = np.load(os.path.join(directory, 'train_mask.npy'))
    val_mask = np.load(os.path.join(directory, 'val_mask.npy'))
    test_mask = np.load(os.path.join(directory, 'test_mask.npy'))
    # X_embedded.npy is a SPARC-only artifact (X @ E from SpectralTrainer);
    # SpectralNet/main.py does not produce it. Treat it as optional so the
    # SPARCphormer pipeline can also consume SpectralNet result directories.
    x_embedded_path = os.path.join(directory, 'X_embedded.npy')
    if os.path.isfile(x_embedded_path):
        X_embedded = np.load(x_embedded_path)
    else:
        X_embedded = None
        print("X_embedded.npy not found in", directory,
              "-- assuming this directory was produced by "
              "SpectralNet/main.py rather than SPARC/main.py "
              "(X_embedded is only used by --feature_concat=X_embedded, "
              "which is currently a dead branch).")

    # SPARC main.py persists the full (symmetric, self-looped) adjacency as
    # full_adj.npz. It can be ~hundreds of MB / 1e8+ nnz on ogbn-products;
    # skip loading when this training run never uses it.
    full_adj_path = os.path.join(directory, 'full_adj.npz')
    if load_adj:
        if os.path.isfile(full_adj_path):
            adj = sp.load_npz(full_adj_path)
            print("Loaded full adjacency from:", full_adj_path,
                  "shape=", adj.shape, "nnz=", adj.nnz)
        else:
            adj = None
            print("No full_adj.npz in", directory,
                  "-- adjacency-based spaces (computed, hops, real_graph) "
                  "will not work.")
    else:
        adj = None
        print("Skipping full_adj.npz (not needed for this --space).")
    features = features.astype(np.float32)

    # if feature_concat == "spectral":
    #     features = np.concatenate((features, spectral_encoding), axis=1)
    # elif feature_concat == "X_embedded":
    #     features = np.concatenate((features, X_embedded), axis=1)
    # "none" → use raw features only

    labels = numpy_int_labels_to_torch(labels, device)
    return spectral_encoding, features, labels, adj, train_mask, val_mask, test_mask


def load_real_graph_data(dataset, split_name="random", test_ratio=0.10,
                         val_ratio=0.03, seed=42, return_adj=True):
    """Load the real graph (features, adjacency, labels) and a cold-start
    train/val/test split built by ``make_cold_start_split`` from
    ``SPARC_project/data/graphsage`` / ``SPARC_project/data/split_data.py``.

    Used when ``--space=real_graph`` (full adjacency for ``re_features_hops``),
    ``--space=features``, or ``--space=cold-brew`` / ``cold-btew`` (raw feats +
    same split as SPARC; no ``sparc_results`` I/O — avoids picking the wrong run
    directory when the preferred folder is missing).

    The adjacency is symmetrised, self-looped, binarised (matching what
    ``SPARC/src/main.py`` writes to ``full_adj.npz``), and converted to a
    sparse torch tensor so it is ready for ``torch.matmul`` inside
    ``utils.re_features_hops``. The split is built on the symmetrised
    adjacency *before* self-loops are added, mirroring SPARC main.py.

    Parameters
    ----------
    dataset : str
        Dataset name (matches the directory under
        ``SPARC_project/data/graphsage/``).
    split_name : str
        ``test_strategy`` forwarded to ``make_cold_start_split``
        (e.g. ``random``, ``low_degree``, ``high_degree``, ``cold_brew``,
        ``original``, ``SA-MLP``).
    test_ratio, val_ratio : float
        Fractions of nodes assigned to the test/val partitions.
    seed : int
        Random seed for ``make_cold_start_split``.
    return_adj : bool
        If False, skip building the self-looped adjacency tensor (saves memory on
        large graphs when only features + masks are needed, e.g. ``--space=features``).

    Returns
    -------
    spectral_encoding : None
        Real-graph mode does not produce a spectral encoding; ``None`` keeps
        the return signature compatible with the SPARC ``load_data`` path.
    features : np.ndarray (float32, N x F)
    labels : torch.Tensor (long, on ``device``)
    adj : torch.sparse_coo_tensor or None
        Self-looped symmetrised adjacency, or None if ``return_adj`` is False.
    train_mask, val_mask, test_mask : np.ndarray (bool, N)
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_root = os.path.normpath(os.path.join(
        script_dir, '..', '..', '..', 'data'))
    graph_dir = os.path.join(data_root, 'graphsage')
    split_dir = os.path.join(data_root, 'graphsage_splits')

    if data_root not in sys.path:
        sys.path.insert(0, data_root)
    from load_data import load_graphsage_graph
    from split_data import make_cold_start_split

    log_prefix = "[real_graph]" if return_adj else "[features]"
    print("{} Loading GraphSAGE graph for '{}' from {}".format(
        log_prefix, dataset, graph_dir))
    adj_raw, features, labels_np = load_graphsage_graph(
        dataset, graph_dir=graph_dir)
    num_nodes = features.shape[0]

    # Mirror SPARC main.py: symmetrise (GraphSAGE stores each edge once)
    # and binarise BEFORE building the split, so degree-based strategies
    # (low_degree / high_degree / cold_brew) see the true undirected degree.
    adj_sym = adj_raw + adj_raw.T
    adj_sym[adj_sym > 1] = 1

    print("{} Building cold-start split via make_cold_start_split "
          "(test_strategy='{}', val_strategy='random', "
          "test_ratio={}, val_ratio={}, seed={})".format(
              log_prefix, split_name, test_ratio, val_ratio, seed))
    idx_train, idx_val, idx_test = make_cold_start_split(
        adj=adj_sym,
        test_frac=test_ratio,
        val_frac=val_ratio,
        test_strategy=split_name,
        val_strategy="random",
        seed=seed,
        dataset_name=dataset,
        split_dir=split_dir,
        labels=labels_np,
    )

    train_mask = np.zeros(num_nodes, dtype=bool)
    val_mask = np.zeros(num_nodes, dtype=bool)
    test_mask = np.zeros(num_nodes, dtype=bool)
    train_mask[idx_train] = True
    val_mask[idx_val] = True
    test_mask[idx_test] = True

    features = features.astype(np.float32)
    labels = numpy_int_labels_to_torch(labels_np, device)

    if return_adj:
        from load_data import sparse_mx_to_torch_sparse_tensor
        # Now add self-loops + binarise to match SPARC's saved full_adj.npz.
        adj_full = adj_sym + sp.eye(num_nodes, dtype=np.float32, format="csr")
        adj_full[adj_full > 1] = 1

        # re_features_hops uses torch.matmul, which does not accept scipy
        # sparse matrices, so hand it a torch sparse tensor.
        adj_torch = sparse_mx_to_torch_sparse_tensor(adj_full)
        edge_report = int(adj_full.nnz)
    else:
        adj_torch = None
        edge_report = int(adj_sym.nnz)

    print("{} N={}, |E (sym{})|={}, "
          "|train|={}, |val|={}, |test|={}".format(
              log_prefix, num_nodes,
              ", self-looped" if return_adj else ", no self-loops",
              edge_report,
              int(train_mask.sum()), int(val_mask.sum()),
              int(test_mask.sum())))
    return None, features, labels, adj_torch, train_mask, val_mask, test_mask


def _cold_brew_subdir_from_fields(test_ratio, val_ratio, seed,
                                  num_layers, dim_hidden,
                                  part1_arch, topk, dropout_mlp,
                                  whether_has_se):
    """Mirror of trainer_node_classification.cold_brew_part1_subdir.

    Replicated here to avoid importing from gnn-tail-generalization at
    SPARCphormer training time (different conda envs may share the disk
    layout but not the Python path).
    """
    return (
        f"test{float(test_ratio)}"
        f"_val{float(val_ratio)}"
        f"_arch{str(part1_arch)}"
        f"_topk{int(topk)}"
        f"_dpMLP{float(dropout_mlp)}"
        f"_nl{int(num_layers)}"
        f"_h{int(dim_hidden)}"
        f"_se{str(whether_has_se)}"
        f"_seed{int(seed)}"
    )


def load_cold_brew_npz(dataset, cold_brew_root=None, split_name="random",
                       seed=42, subdir=None,
                       test_ratio=None, val_ratio=None,
                       num_layers=None, dim_hidden=None,
                       part1_arch="2layer", topk=3, dropout_mlp=0.5,
                       whether_has_se="000"):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if cold_brew_root is None:
        cold_brew_root = os.path.join(
            script_dir, '..', '..', '..', 'gnn-tail-generalization')
    cold_brew_root = os.path.normpath(cold_brew_root)

    dataset_key = str(dataset).lower()

    # Build the candidate subdir tag in priority order:
    #   1. caller-provided ``subdir`` (verbatim, fully encoded).
    #   2. Field-by-field tag if (test_ratio, val_ratio, num_layers,
    #      dim_hidden) are all supplied.
    #   3. Legacy ``seed<s>`` layout (kept for backward compatibility with
    #      .npz files saved by the previous Cold-Brew code).
    candidate_subdirs = []
    if subdir:
        candidate_subdirs.append(str(subdir))
    if (test_ratio is not None and val_ratio is not None
            and num_layers is not None and dim_hidden is not None):
        candidate_subdirs.append(_cold_brew_subdir_from_fields(
            test_ratio=test_ratio, val_ratio=val_ratio, seed=seed,
            num_layers=num_layers, dim_hidden=dim_hidden,
            part1_arch=part1_arch, topk=topk, dropout_mlp=dropout_mlp,
            whether_has_se=whether_has_se,
        ))
    candidate_subdirs.append(f"seed{int(seed)}")  # legacy layout

    npz_path = None
    for cand in candidate_subdirs:
        path = os.path.normpath(os.path.join(
            cold_brew_root, 'embeddings', dataset_key, split_name,
            cand, 'part1.npz'))
        if os.path.isfile(path):
            npz_path = path
            print("Cold-Brew embeddings: matched subdir tag '{}'".format(cand))
            break

    if npz_path is None:
        pattern = os.path.join(
            cold_brew_root, 'embeddings', dataset_key, '**', '*.npz')
        candidates = sorted(glob.glob(pattern, recursive=True))
        if not candidates:
            tried = "\n  ".join(
                os.path.join(cold_brew_root, 'embeddings', dataset_key,
                             split_name, c, 'part1.npz')
                for c in candidate_subdirs
            )
            raise FileNotFoundError(
                "No Cold-Brew .npz files found for dataset '{}'. "
                "Tried (in priority order):\n  {}\nAnd no '*.npz' under {} either.".format(
                    dataset_key, tried,
                    os.path.join(cold_brew_root, 'embeddings', dataset_key))
            )
        npz_path = candidates[0]
        print("Preferred Cold-Brew subdirs not found; using first existing "
              "option (may be a stale run!):", npz_path)

    print("Loading Cold-Brew embeddings from:", npz_path)
    data = np.load(npz_path, allow_pickle=False)
    required_keys = ['embeddings', 'y', 'train_mask', 'val_mask', 'test_mask']
    missing_keys = [k for k in required_keys if k not in data]
    if missing_keys:
        raise KeyError(
            "Cold-Brew file {} is missing keys: {}".format(npz_path, ', '.join(missing_keys)))

    embeddings = data['embeddings'].astype(np.float32)
    labels = data['y']
    train_mask = data['train_mask'].astype(bool)
    val_mask = data['val_mask'].astype(bool)
    test_mask = data['test_mask'].astype(bool)
    return embeddings, labels, train_mask, val_mask, test_mask


def laplacian_eigenvectors(A, dim):
    """
    Compute Graph Laplacian eigenvectors.

    Parameters:
        A (torch.Tensor or scipy.sparse matrix): Adjacency matrix of the graph. If a torch.Tensor, it will be converted to a sparse matrix.
        dim (int): Number of eigenvectors to compute.

    Returns:
        torch.Tensor: Matrix of Laplacian eigenvectors.
    """
    # Convert A to a sparse matrix if it's a torch.Tensor
    if isinstance(A, torch.Tensor):
        # If it's a sparse tensor, convert it to dense first
        if A.is_sparse:
            A = A.to_dense()
        A = A.numpy()  # Convert to numpy array
        A = sp.csr_matrix(A)  # Convert to scipy sparse matrix

    # Ensure the adjacency matrix is in CSR format
    A = A.tocsr().astype(float)

    # Degree matrix with normalized values
    degrees = np.asarray(A.sum(axis=1)).flatten()
    degrees = np.clip(degrees, 1, None)  # Avoid division by zero
    N = sp.diags(degrees ** -0.5, dtype=float)

    # Compute the normalized Laplacian matrix
    L = sp.eye(A.shape[0]) - N @ A @ N

    # Compute eigenvalues and eigenvectors of the Laplacian
    EigVal, EigVec = sp.linalg.eigs(L, k=dim + 1, which='SR', tol=1e-2)

    # Sort eigenvectors by ascending eigenvalue order
    EigVec = EigVec[:, EigVal.argsort()]

    # Select the top positional encoding dimensions
    laplace_eigenvectors = torch.from_numpy(EigVec[:, :dim]).float()

    return laplace_eigenvectors


def laplacian_eigenvectors_multihop(A, dim, walk_order=3):
    """Eigenvectors of L_sym(W_acc) where W_acc = W + W^2 + ... + W^walk_order.

    This mirrors `SpectralTrainer._get_valid_affinity_matrix` exactly so the
    "computed_symmetric_multihop_laplace" space tests the *same* operator
    SpectralNet is trained against (the only thing left to differ is then
    SpectralNet's approximation quality, not the operator/graph).

    Steps:
        1. Symmetrize and drop self-loops on A -> W (raw adjacency, unweighted ok).
        2. Sum walks: W_acc = W + W^2 + ... + W^p.
        3. Drop self-loops introduced by even-length walks; resymmetrize.
        4. Build L_sym = I - D^{-1/2} W_acc D^{-1/2} from W_acc's row sums.
        5. Compute the bottom (dim+1) eigenpairs and drop the trivial one (~0)
           so the returned `dim` eigenvectors do not include the constant
           component (aligned with SpectralNet once batches subtract the mean).

    Parameters:
        A (torch.Tensor or scipy.sparse matrix): graph adjacency.
        dim (int): number of non-trivial eigenvectors to return.
        walk_order (int): p in W + W^2 + ... + W^p. Must be >= 1.
    """
    if walk_order < 1:
        raise ValueError("walk_order must be >= 1")

    # Convert to scipy sparse CSR
    if isinstance(A, torch.Tensor):
        if A.is_sparse:
            A = A.to_dense()
        A = A.numpy()
        A = sp.csr_matrix(A)
    A = A.tocsr().astype(np.float64).copy()

    # 1) drop self-loops, symmetrize -> W
    A.setdiag(0.0)
    A.eliminate_zeros()
    A = ((A + A.T) * 0.5).tocsr()

    # 2) accumulate W + W^2 + ... + W^p
    W = A
    W_acc = W
    Wi = W
    for _ in range(1, walk_order):
        Wi = (Wi @ W).tocsr()
        W_acc = (W_acc + Wi).tocsr()

    # 3) drop self-loops re-introduced by even-length walks; resymmetrize
    W_acc.setdiag(0.0)
    W_acc.eliminate_zeros()
    W_acc = ((W_acc + W_acc.T) * 0.5).tocsr()

    # 4) symmetric normalized Laplacian of W_acc
    deg = np.asarray(W_acc.sum(axis=1)).flatten()
    d_inv_sqrt = np.zeros_like(deg)
    nz = deg > 0
    d_inv_sqrt[nz] = 1.0 / np.sqrt(deg[nz])
    D_inv_sqrt = sp.diags(d_inv_sqrt)
    n = W_acc.shape[0]
    L = sp.eye(n, dtype=np.float64) - D_inv_sqrt @ W_acc @ D_inv_sqrt
    L = ((L + L.T) * 0.5).tocsr()

    # 5) bottom (dim+1) eigenpairs via shift-invert near 0
    k_req = max(1, min(int(dim) + 1, n - 1))
    rng = np.random.default_rng(0)
    v0 = rng.standard_normal(n).astype(np.float64)
    v0 /= np.linalg.norm(v0) + 1e-30
    try:
        EigVal, EigVec = sp.linalg.eigsh(
            L, k=k_req, sigma=-1e-5, which='LM', tol=1e-6,
            maxiter=2000, v0=v0)
    except Exception as exc:
        print("[multihop] eigsh shift-invert failed ({}); "
              "retrying with which='SA'.".format(exc))
        EigVal, EigVec = sp.linalg.eigsh(
            L, k=k_req, which='SA', tol=1e-6, maxiter=2000, v0=v0)

    order = np.argsort(EigVal)
    EigVal = EigVal[order]
    EigVec = EigVec[:, order]
    # Drop the trivial smallest eigenvector (constant on each connected
    # component) so we keep `dim` informative directions.
    EigVec = EigVec[:, 1:1 + dim]
    return torch.from_numpy(EigVec).float()


def create_data_loaders(processed_features, labels, train_mask, val_mask, test_mask, batch_size):
    batch_data_train = Data.TensorDataset(
        processed_features[train_mask], labels[train_mask])
    batch_data_val = Data.TensorDataset(
        processed_features[val_mask], labels[val_mask])
    batch_data_test = Data.TensorDataset(
        processed_features[test_mask], labels[test_mask])

    train_data_loader = Data.DataLoader(
        batch_data_train, batch_size=args.batch_size, shuffle=True)
    val_data_loader = Data.DataLoader(
        batch_data_val, batch_size=args.batch_size, shuffle=True)
    test_data_loader = Data.DataLoader(
        batch_data_test, batch_size=args.batch_size, shuffle=True)

    return train_data_loader, val_data_loader, test_data_loader


def grassmann_distance(A, B):
    """
    Compute the Grassmann distance between two matrices A and B.

    Parameters:
        A (ndarray): Matrix representing the first subspace (m x n).
        B (ndarray): Matrix representing the second subspace (m x n).

    Returns:
        float: The Grassmann distance.
    """
    # Compute the SVD of A.T @ B
    U, singular_values, Vh = svd(A.T @ B)

    # Compute the principal angles (theta)
    theta = np.arccos(np.clip(singular_values, -1.0, 1.0))

    # Grassmann distance is the sum of the squares of the sines of the principal angles
    distance = np.sqrt(np.sum(np.sin(theta) ** 2))
    return distance


def train_valid_epoch(epoch):

    model.train()
    loss_train_b = 0
    acc_train_b = 0
    train_seen = 0
    train_probs_list, train_labels_list = [], []
    for _, item in enumerate(train_data_loader):

        nodes_features = item[0].to(device)
        labels = item[1].to(device)

        optimizer.zero_grad()
        output = model(nodes_features)
        loss_train = F.nll_loss(output, labels)
        loss_train.backward()
        optimizer.step()
        lr_scheduler.step()

        loss_train_b += loss_train.item()
        acc_train = utils.accuracy_batch(output, labels)
        acc_train_b += acc_train.item()
        train_seen += labels.size(0)
        train_probs_list.append(torch.exp(output).detach().cpu())
        train_labels_list.append(labels.cpu())

    model.eval()
    loss_val_b = 0
    acc_val_b = 0
    val_seen = 0
    val_probs_list, val_labels_list = [], []
    for _, item in enumerate(val_data_loader):
        nodes_features = item[0].to(device)
        labels = item[1].to(device)

        output = model(nodes_features)
        loss_val = F.nll_loss(output, labels)

        loss_val_b += loss_val.item()
        acc_val = utils.accuracy_batch(output, labels)
        acc_val_b += acc_val.item()
        val_seen += labels.size(0)
        val_probs_list.append(torch.exp(output).detach().cpu())
        val_labels_list.append(labels.cpu())

    train_denom = max(train_seen, 1)
    val_denom = max(val_seen, 1)
    train_roc = _compute_roc_auc(
        torch.cat(train_probs_list).numpy(), torch.cat(train_labels_list).numpy())
    val_roc = _compute_roc_auc(
        torch.cat(val_probs_list).numpy(), torch.cat(val_labels_list).numpy())
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train_b),
          'acc_train: {:.4f}'.format(acc_train_b / train_denom),
          'roc_auc_train: {:.4f}'.format(train_roc),
          'loss_val: {:.4f}'.format(loss_val_b),
          'acc_val: {:.4f}'.format(acc_val_b / val_denom),
          'roc_auc_val: {:.4f}'.format(val_roc))

    return loss_val_b, acc_val_b


def _compute_roc_auc(all_probs, all_labels):
    """Compute ROC-AUC from collected probability and label arrays."""
    try:
        n_classes = all_probs.shape[1]
        if n_classes == 2:
            return roc_auc_score(all_labels, all_probs[:, 1])
        else:
            return roc_auc_score(all_labels, all_probs, multi_class="ovr")
    except ValueError:
        return float("nan")


def test():

    model.eval()

    loss_train = 0
    acc_train = 0
    train_seen = 0
    train_probs_list, train_labels_list = [], []
    for _, item in enumerate(train_data_loader):
        nodes_features = item[0].to(device)
        labels = item[1].to(device)
        output = model(nodes_features)
        loss_train += F.nll_loss(output, labels).item()
        acc_train += utils.accuracy_batch(output, labels).item()
        train_seen += labels.size(0)
        train_probs_list.append(torch.exp(output).detach().cpu())
        train_labels_list.append(labels.cpu())

    loss_val = 0
    acc_val = 0
    val_seen = 0
    val_probs_list, val_labels_list = [], []
    for _, item in enumerate(val_data_loader):
        nodes_features = item[0].to(device)
        labels = item[1].to(device)
        output = model(nodes_features)
        loss_val += F.nll_loss(output, labels).item()
        acc_val += utils.accuracy_batch(output, labels).item()
        val_seen += labels.size(0)
        val_probs_list.append(torch.exp(output).detach().cpu())
        val_labels_list.append(labels.cpu())

    loss_test = 0
    acc_test = 0
    test_seen = 0
    test_probs_list, test_labels_list = [], []
    for _, item in enumerate(test_data_loader):
        nodes_features = item[0].to(device)
        labels = item[1].to(device)
        output = model(nodes_features)
        loss_test += F.nll_loss(output, labels).item()
        acc_test += utils.accuracy_batch(output, labels).item()
        test_seen += labels.size(0)
        test_probs_list.append(torch.exp(output).detach().cpu())
        test_labels_list.append(labels.cpu())

    train_roc = _compute_roc_auc(
        torch.cat(train_probs_list).numpy(), torch.cat(train_labels_list).numpy())
    val_roc = _compute_roc_auc(
        torch.cat(val_probs_list).numpy(), torch.cat(val_labels_list).numpy())
    test_roc = _compute_roc_auc(
        torch.cat(test_probs_list).numpy(), torch.cat(test_labels_list).numpy())

    print("Train accuracy = {:.4f}".format(acc_train / max(train_seen, 1)))
    print("Val accuracy = {:.4f}".format(acc_val / max(val_seen, 1)))
    print("Test accuracy = {:.4f}".format(acc_test / max(test_seen, 1)))
    print("Train ROC-AUC = {:.4f}".format(train_roc))
    print("Val ROC-AUC = {:.4f}".format(val_roc))
    print("Test ROC-AUC = {:.4f}".format(test_roc))


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


args = parse_args()

_argv = sys.argv
if args.sparc_seed is None:
    args.sparc_seed = args.seed
    print("[sparc] --sparc_seed not provided; using --seed={} as the SPARC split seed.".format(args.seed))

if args.space == 'fusion':
    if '--split_name' not in _argv:
        args.split_name = 'random'
    if '--cold_brew_split' not in _argv:
        args.cold_brew_split = 'random'
    if '--cold_brew_seed' not in _argv:
        args.cold_brew_seed = 42
    if '--feature_concat' not in _argv:
        args.feature_concat = 'none'
    print("[fusion] Using split={}, sparc_seed={}, cold_brew_split={}, cold_brew_seed={}, feature_concat={}".format(
        args.split_name, args.sparc_seed, args.cold_brew_split, args.cold_brew_seed, args.feature_concat))

VALID_SPACES = {'spectral', 'features', 'computed',
                'computed_symmetric_multihop_laplace',
                'real_graph', 'hops', 'cold-brew', 'cold-btew', 'fusion'}
if args.space not in VALID_SPACES:
    raise ValueError(
        "Unknown --space='{}'. Valid options are: {}".format(
            args.space, sorted(VALID_SPACES)))

seed = args.seed
set_seed(seed)

if args.device is not None and args.device < 0:
    device = torch.device('cpu')
elif torch.cuda.is_available():
    n_gpus = torch.cuda.device_count()
    requested = args.device if args.device is not None else 0
    if requested >= n_gpus:
        print("[device] requested cuda:{} but only {} GPU(s) visible; "
              "falling back to cuda:0.".format(requested, n_gpus))
        requested = 0
    device = torch.device('cuda:{}'.format(requested))
else:
    device = torch.device('cpu')
print("[device] using:", device)


# Load data. Most spaces use SPARC main.py outputs under sparc_results/.
# --space=features uses raw GraphSAGE feats + make_cold_start_split (same as SPARC
# main.py) so masks match --split_name / --test_ratio / --val_ratio / --sparc_seed
# without falling back to an arbitrary sparc_results subfolder.
# --space=cold-brew / cold-btew only need those raw feats (Cold-Brew .npz supplies
# embeddings + split); requiring sparc_results here would force SPARC main.py to
# run before the Cold-Brew step, contradicting run_full_pipeline.py step order.
# Avoid loading full_adj.npz (large on ogbn-products) unless the run uses it.
need_sparc_adj = args.space in (
    'computed', 'computed_symmetric_multihop_laplace', 'real_graph', 'hops')
if args.space == 'features':
    spectral_encoding, features, labels, adj, train_mask, val_mask, test_mask = (
        load_real_graph_data(
            args.dataset,
            split_name=args.split_name,
            test_ratio=args.test_ratio,
            val_ratio=args.val_ratio,
            seed=args.sparc_seed,
            return_adj=False))
elif args.space in ('cold-brew', 'cold-btew'):
    spectral_encoding, features, labels, adj, train_mask, val_mask, test_mask = (
        load_real_graph_data(
            args.dataset,
            split_name=args.split_name,
            test_ratio=args.test_ratio,
            val_ratio=args.val_ratio,
            seed=args.sparc_seed,
            return_adj=False))
else:
    spectral_encoding, features, labels, adj, train_mask, val_mask, test_mask = load_data(
        args.dataset, split_name=args.split_name, test_ratio=args.test_ratio, sparc_seed=args.sparc_seed,
        feature_concat=args.feature_concat, result_suffix=args.sparc_result_suffix,
        load_adj=need_sparc_adj)
# Adjacency for training comes from SPARC outputs (full_adj.npz) via load_data above.
# load_real_graph_data (GraphSAGE JSON) is intentionally not used for spectral/etc.:
# it duplicated I/O on large datasets (e.g. ogbn-products). Spaces that need a torch
# sparse adjacency for torch.matmul (hops, real_graph) convert the scipy CSR below.

if args.space == 'spectral':
    print('spectral')
    processed_features = utils.re_features_spectral_diffusion_distance_avarage_seq_spectral_encoding_coldstart(
        spectral_encoding, features, args.hops, train_mask, val_mask, test_mask,
        hop_avg_style=args.hop_avg_style, knn_metric=args.knn_metric)
elif args.space == 'features':
    print('features')
    processed_features = utils.re_features_spectral_diffusion_distance_avarage_seq_spectral_encoding_coldstart(
        features, features, args.hops, train_mask, val_mask, test_mask,
        hop_avg_style=args.hop_avg_style, knn_metric=args.knn_metric)
elif args.space == 'computed':
    print('computed')
    if adj is None:
        raise RuntimeError(
            "--space=computed requires an adjacency matrix, but none was loaded. "
            "Make sure train_adj.npz exists in the SPARC results directory "
            "(SPARC main.py saves it automatically).")
    spectral_encoding_computed = laplacian_eigenvectors(adj, args.pe_dim)
    processed_features = utils.re_features_spectral_diffusion_distance_avarage_seq_spectral_encoding_coldstart(
        spectral_encoding_computed, features, args.hops, train_mask, val_mask, test_mask,
        hop_avg_style=args.hop_avg_style, knn_metric=args.knn_metric)

    # processed_features = utils.re_features_knn_sequence_coldstart(
    #     spectral_encoding_computed, features, args.hops, train_mask, val_mask, test_mask)
elif args.space == 'computed_symmetric_multihop_laplace':
    print('computed_symmetric_multihop_laplace (walk_order={}, pe_dim={})'.format(
        args.multihop_walk_order, args.pe_dim))
    if adj is None:
        raise RuntimeError(
            "--space=computed_symmetric_multihop_laplace requires an adjacency matrix, "
            "but none was loaded. Make sure full_adj.npz exists in the SPARC results "
            "directory (SPARC main.py saves it automatically).")
    spectral_encoding_multihop = laplacian_eigenvectors_multihop(
        adj, args.pe_dim, walk_order=args.multihop_walk_order)
    processed_features = utils.re_features_spectral_diffusion_distance_avarage_seq_spectral_encoding_coldstart(
        spectral_encoding_multihop, features, args.hops, train_mask, val_mask, test_mask,
        hop_avg_style=args.hop_avg_style, knn_metric=args.knn_metric)
elif args.space == 'real_graph':
    print('real_graph')
    if adj is None:
        raise RuntimeError(
            "--space=real_graph requires an adjacency matrix, but none was loaded. "
            "Make sure full_adj.npz exists in the SPARC results directory "
            "(SPARC main.py saves it automatically).")
    if sp.issparse(adj):
        adj = utils.sparse_mx_to_torch_sparse_tensor(adj.tocsr())
    processed_features = utils.re_features_hops(adj, features, args.hops)
elif args.space == 'hops':
    print('hops')
    if adj is None:
        raise RuntimeError(
            "--space=hops requires an adjacency matrix, but none was loaded. "
            "Make sure train_adj.npz exists in the SPARC results directory "
            "(SPARC main.py saves it automatically).")
    if sp.issparse(adj):
        adj = utils.sparse_mx_to_torch_sparse_tensor(adj.tocsr())
    processed_features = utils.re_features_hops(adj, features, args.hops)
elif args.space in ['cold-brew', 'cold-btew']:
    print('cold-brew')
    _cb_test_ratio = (args.cold_brew_test_ratio
                      if args.cold_brew_test_ratio is not None
                      else args.test_ratio)
    _cb_val_ratio = (args.cold_brew_val_ratio
                     if args.cold_brew_val_ratio is not None
                     else args.val_ratio)
    cold_brew_embeddings, cold_brew_labels, cold_brew_train_mask, cold_brew_val_mask, cold_brew_test_mask = load_cold_brew_npz(
        args.dataset,
        cold_brew_root=args.cold_brew_root,
        split_name=args.cold_brew_split,
        seed=args.cold_brew_seed,
        subdir=args.cold_brew_subdir or None,
        test_ratio=_cb_test_ratio,
        val_ratio=_cb_val_ratio,
        num_layers=args.cold_brew_num_layers,
        dim_hidden=args.cold_brew_dim_hidden,
        part1_arch=args.cold_brew_part1_arch,
        topk=args.cold_brew_topk,
        dropout_mlp=args.cold_brew_dropout_mlp,
        whether_has_se=args.cold_brew_whether_has_se,
    )
    if cold_brew_embeddings.shape[0] != features.shape[0]:
        raise ValueError(
            "Cold-Brew embeddings node count ({}) does not match feature node count ({}).".format(
                cold_brew_embeddings.shape[0], features.shape[0]
            )
        )
    labels = numpy_int_labels_to_torch(cold_brew_labels, device)
    train_mask = cold_brew_train_mask
    val_mask = cold_brew_val_mask
    test_mask = cold_brew_test_mask
    processed_features = utils.re_features_spectral_diffusion_distance_avarage_seq_spectral_encoding_coldstart(
        cold_brew_embeddings, features, args.hops, train_mask, val_mask, test_mask,
        hop_avg_style=args.hop_avg_style, knn_metric=args.knn_metric)
elif args.space == 'fusion':
    print('fusion (SPARC + Cold-Brew embeddings concatenated)')
    _cb_test_ratio = (args.cold_brew_test_ratio
                      if args.cold_brew_test_ratio is not None
                      else args.test_ratio)
    _cb_val_ratio = (args.cold_brew_val_ratio
                     if args.cold_brew_val_ratio is not None
                     else args.val_ratio)
    cold_brew_embeddings, cb_labels, cb_train_mask, cb_val_mask, cb_test_mask = load_cold_brew_npz(
        args.dataset,
        cold_brew_root=args.cold_brew_root,
        split_name=args.cold_brew_split,
        seed=args.cold_brew_seed,
        subdir=args.cold_brew_subdir or None,
        test_ratio=_cb_test_ratio,
        val_ratio=_cb_val_ratio,
        num_layers=args.cold_brew_num_layers,
        dim_hidden=args.cold_brew_dim_hidden,
        part1_arch=args.cold_brew_part1_arch,
        topk=args.cold_brew_topk,
        dropout_mlp=args.cold_brew_dropout_mlp,
        whether_has_se=args.cold_brew_whether_has_se,
    )
    if cold_brew_embeddings.shape[0] != spectral_encoding.shape[0]:
        raise ValueError(
            "Fusion requires matching node counts. "
            "SPARC spectral_encoding has N={} but Cold-Brew embeddings have N={}.".format(
                spectral_encoding.shape[0], cold_brew_embeddings.shape[0]
            )
        )

    sparc_train_mask_bool = np.asarray(train_mask).astype(bool)
    sparc_val_mask_bool = np.asarray(val_mask).astype(bool)
    sparc_test_mask_bool = np.asarray(test_mask).astype(bool)
    cb_train_mask_bool = np.asarray(cb_train_mask).astype(bool)
    cb_val_mask_bool = np.asarray(cb_val_mask).astype(bool)
    cb_test_mask_bool = np.asarray(cb_test_mask).astype(bool)

    mask_mismatches = {}
    if not np.array_equal(sparc_train_mask_bool, cb_train_mask_bool):
        mask_mismatches['train_mask'] = (
            int(sparc_train_mask_bool.sum()), int(cb_train_mask_bool.sum()))
    if not np.array_equal(sparc_val_mask_bool, cb_val_mask_bool):
        mask_mismatches['val_mask'] = (
            int(sparc_val_mask_bool.sum()), int(cb_val_mask_bool.sum()))
    if not np.array_equal(sparc_test_mask_bool, cb_test_mask_bool):
        mask_mismatches['test_mask'] = (
            int(sparc_test_mask_bool.sum()), int(cb_test_mask_bool.sum()))

    sparc_labels_np = labels.detach().cpu().numpy()
    cb_labels_np = np.asarray(cb_labels)
    if sparc_labels_np.shape != cb_labels_np.shape or not np.array_equal(sparc_labels_np, cb_labels_np):
        mask_mismatches['labels'] = ("shape={}".format(sparc_labels_np.shape),
                                     "shape={}".format(cb_labels_np.shape))

    if mask_mismatches:
        details = "\n".join(
            "  {}: SPARC={}  ColdBrew={}".format(k, v[0], v[1])
            for k, v in mask_mismatches.items()
        )
        raise ValueError(
            "[fusion] SPARC and Cold-Brew splits/labels do NOT match. Aborting.\n"
            "Re-run the Cold-Brew embedding generation with --use_sparc_masks 1 "
            "(and matching --split_name / --random_seed / --test_ratio / --val_ratio) "
            "so both pipelines see identical test nodes.\n"
            "Mismatch summary (count of True nodes, or shape):\n" + details
        )
    print("[fusion] verified SPARC and Cold-Brew splits/labels match "
          "(|train|={}, |val|={}, |test|={}).".format(
              int(sparc_train_mask_bool.sum()),
              int(sparc_val_mask_bool.sum()),
              int(sparc_test_mask_bool.sum())))

    fused_embeddings = np.concatenate(
        (spectral_encoding.astype(np.float32),
         cold_brew_embeddings.astype(np.float32)),
        axis=1)
    print("Fused embedding shape:", fused_embeddings.shape,
          "(SPARC dim={}, Cold-Brew dim={})".format(
              spectral_encoding.shape[1], cold_brew_embeddings.shape[1]))
    processed_features = utils.re_features_spectral_diffusion_distance_avarage_seq_spectral_encoding_coldstart(
        fused_embeddings, features, args.hops, train_mask, val_mask, test_mask,
        hop_avg_style=args.hop_avg_style, knn_metric=args.knn_metric)

# creat data loaders
train_data_loader, val_data_loader, test_data_loader = create_data_loaders(
    processed_features, labels, train_mask, val_mask, test_mask, args.batch_size)

# model configuration
model = TransformerModel(hops=args.hops,
                         n_class=labels.max().item() + 1,
                         input_dim=(features.shape[1]),
                         n_layers=args.n_layers,
                         num_heads=args.n_heads,
                         hidden_dim=args.hidden_dim,
                         ffn_dim=args.ffn_dim,
                         dropout_rate=args.dropout,
                         attention_dropout_rate=args.attention_dropout).to(device)

# print(model)
# print('total params:', sum(p.numel() for p in model.parameters()))

optimizer = torch.optim.AdamW(
    model.parameters(), lr=args.peak_lr, weight_decay=args.weight_decay)
lr_scheduler = PolynomialDecayLR(
    optimizer,
    warmup_updates=args.warmup_updates,
    tot_updates=args.tot_updates,
    lr=args.peak_lr,
    end_lr=args.end_lr,
    power=1.0,
)


print("training...")

t_total = time.time()
stopping_args = Stop_args(patience=args.patience, max_epochs=args.epochs)
early_stopping = EarlyStopping(model, **stopping_args)
for epoch in range(args.epochs):
    loss_val, acc_val = train_valid_epoch(epoch)
    if early_stopping.check([acc_val, loss_val], epoch):
        break

print("Optimization Finished!")
print("Train cost: {:.4f}s".format(time.time() - t_total))
# Restore best model
# print('Loading {}th epoch'.format(early_stopping.best_epoch+1))
model.load_state_dict(early_stopping.best_state)

print("testing...")
test()
