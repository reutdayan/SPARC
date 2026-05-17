import argparse
import sys
import json
import matplotlib.pyplot as plt
import networkx as nx
import torch
import random
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh, lobpcg
from scipy.sparse import csgraph
from scipy.sparse.csgraph import connected_components
from sklearn.neighbors import KNeighborsClassifier
import utils
import partition_utils
import partition_cache
from utils import *
from metrics import Metrics
from sklearn.cluster import KMeans
from SpectralNet import SpectralNet
from scipy.spatial.distance import cdist
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import os
from scipy.sparse import save_npz
from sklearn.cluster import SpectralClustering
from eigenspace_diagnostics import run_diagnostic

# ---------------------------------------------------------------------------
# Data utilities from SPARC_project/data/
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
_PROJECT_ROOT = os.path.normpath(
    os.path.join(_SCRIPT_DIR, os.pardir, os.pardir))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from data.load_data import load_graphsage_graph, build_inductive_adj  # noqa: E402
from data.split_data import make_cold_start_split                     # noqa: E402

DEFAULT_SEED = 42
DEFAULT_TEST_RATIO = 0.03
DEFAULT_VAL_RATIO = 0.03

DATASET_BASES = ("minesweeper")


class InvalidMatrixException(Exception):
    pass


def _deep_merge(base, overrides):
    """Recursively merge *overrides* into *base* dict (in-place)."""
    for k, v in overrides.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v


class Unbuffered(object):
    def __init__(self, stream):
        self.stream = stream

    def write(self, data):
        self.stream.write(data)
        self.stream.flush()

    def writelines(self, datas):
        self.stream.writelines(datas)
        self.stream.flush()

    def __getattr__(self, attr):
        return getattr(self.stream, attr)


def set_seed(seed: int = 42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_data(dataset, seed, test_ratio, val_ratio, split_name, graph_dir):
    """Load graph via shared data utilities and create a cold-start split.

    Returns the same tuple layout the rest of the SPARC pipeline expects:
        (train_adj, full_adj, train_feats, test_feats,
         y_train, y_val, y_test,
         train_mask, val_mask, test_mask,
         idx_train, idx_val, idx_test, num_data, visible_data)
    """
    adj_raw, features, labels_int = load_graphsage_graph(
        dataset, graph_dir=graph_dir)
    num_data = adj_raw.shape[0]
    num_classes = len(set(labels_int.tolist()))

    # Symmetrise adjacency (GraphSAGE JSON stores each edge once)
    adj_sym = adj_raw + adj_raw.T
    adj_sym[adj_sym > 1] = 1

    # Cold-start split
    idx_train, idx_val, idx_test = make_cold_start_split(
        adj=adj_sym,
        test_frac=test_ratio,
        val_frac=val_ratio,
        test_strategy=split_name,
        val_strategy="random",
        dataset_name=dataset,
        seed=seed,
    )
    print("|train|={}, |val|={}, |test|={}".format(
        len(idx_train), len(idx_val), len(idx_test)))

    # Inductive train adjacency (train-train edges only) + self-loops
    train_adj = build_inductive_adj(adj_sym, idx_train)
    train_adj = train_adj + sp.eye(num_data, dtype=np.float32, format="csr")
    train_adj[train_adj > 1] = 1

    # Full adjacency + self-loops
    full_adj = adj_sym + sp.eye(num_data, dtype=np.float32, format="csr")
    full_adj[full_adj > 1] = 1

    # Normalise features using train-set statistics
    scaler = StandardScaler()
    scaler.fit(features[idx_train])
    features = scaler.transform(features)

    # One-hot labels
    labels = np.zeros((num_data, num_classes), dtype=np.float32)
    for i in range(num_data):
        labels[i, labels_int[i]] = 1

    y_train = np.zeros(labels.shape, dtype=np.float32)
    y_val = np.zeros(labels.shape, dtype=np.float32)
    y_train[idx_train, :] = labels[idx_train, :]
    y_val[idx_val, :] = labels[idx_val, :]
    y_test = labels

    train_mask = utils.sample_mask(idx_train, num_data)
    val_mask = utils.sample_mask(idx_val, num_data)
    test_mask = utils.sample_mask(idx_test, num_data)

    train_feats = features
    test_feats = features
    visible_data = idx_train

    return (train_adj, full_adj, train_feats, test_feats,
            y_train, y_val, y_test,
            train_mask, val_mask, test_mask,
            idx_train, idx_val, idx_test, num_data, visible_data)


def build_symmetric_feature_knn_adjacency(features, k=10, metric="cosine"):
    """Binary symmetric adjacency from feature kNN (no graph edges).

    For each node i, set ``A_ij = 1`` for each j among the ``k`` nearest
    neighbors of i (excluding i). Symmetry is enforced by taking the union
    of the two directions: ``A = max(A_directed, A_directed.T)``, so an
    edge exists iff ``j ∈ kNN(i)`` or ``i ∈ kNN(j)``.

    Self-loops are not included; add ``sp.eye(n)`` to match ``load_data``.

    Implemented via ``utils.build_feature_knn_adj`` with ``mutual=False``.
    """
    return utils.build_feature_knn_adj(
        features, k=int(k), metric=str(metric), mutual=False)


def test_symmetric_feature_knn_replaces_graph_adjacency():
    """Use feature kNN (k=10) for A instead of graph structure: shape, symmetry, pipeline."""
    n, dim = 55, 12
    rng = np.random.default_rng(42)
    feats = rng.standard_normal((n, dim)).astype(np.float32)
    feats = StandardScaler().fit_transform(feats).astype(np.float32)

    k = 10
    A_features = build_symmetric_feature_knn_adjacency(
        feats, k=k, metric="cosine").tocsr()

    assert A_features.shape == (n, n)
    Ad = A_features.toarray()
    assert np.allclose(Ad, Ad.T), "A_features must be symmetric"
    assert np.allclose(np.diag(Ad), 0.0), "no self-loops before adding I"
    uniq = np.unique(Ad)
    assert uniq.size <= 2 and set(uniq.tolist()).issubset({0.0, 1.0})

    # Same pipeline as graph A: self-loops + inductive train subgraph.
    full_adj = A_features + sp.eye(n, dtype=np.float32, format="csr")
    full_adj[full_adj > 1] = 1
    idx_train = np.arange(40, dtype=np.int64)
    train_adj = build_inductive_adj(full_adj, idx_train)
    train_adj = train_adj + sp.eye(n, dtype=np.float32, format="csr")
    train_adj[train_adj > 1] = 1

    assert train_adj.shape == (n, n)
    Ta = train_adj.toarray()
    assert np.allclose(Ta, Ta.T), "train_adj must be symmetric"


def build_label_q_adj(y_train, train_mask, weight=1.0,
                      pair_encoding="binary"):
    """Build label-structure Q from train labels only (for A + Q tests).

    pair_encoding:
        * ``"binary"`` (default): Q_ij = ``weight`` for same-label train
          pairs (i != j), else 0. Entries are in ``{0, weight}``.
        * ``"signed"``: for labeled train pairs (i != j), Q_ij = ``weight``
          if same class and ``-weight`` if different class; 0 elsewhere.
          Entries are in ``{-weight, 0, weight}`` (typically ±1, 0).
    """
    train_idx = np.where(np.asarray(train_mask).astype(bool))[0]
    if train_idx.size == 0:
        return sp.csr_matrix(y_train.shape[:1] * 2, dtype=np.float32)

    labels = np.asarray(y_train[train_idx]).argmax(axis=1)
    encoding = str(pair_encoding or "binary").lower().strip()
    if encoding not in ("binary", "signed"):
        raise ValueError(
            "pair_encoding must be 'binary' or 'signed', got {!r}".format(
                pair_encoding))
    rows = []
    cols = []
    datas = []

    for cls in np.unique(labels):
        cls_idx = train_idx[labels == cls]
        if cls_idx.size < 2:
            continue
        rr = np.repeat(cls_idx, cls_idx.size)
        cc = np.tile(cls_idx, cls_idx.size)
        keep = rr != cc
        rows.append(rr[keep])
        cols.append(cc[keep])
        datas.append(
            np.full(np.count_nonzero(keep), float(weight), dtype=np.float32))

    if encoding == "signed":
        classes = np.unique(labels)
        for i, c1 in enumerate(classes):
            idx1 = train_idx[labels == c1]
            if idx1.size == 0:
                continue
            for c2 in classes[i + 1:]:
                idx2 = train_idx[labels == c2]
                if idx2.size == 0:
                    continue
                r_fwd = np.repeat(idx1, idx2.size)
                c_fwd = np.tile(idx2, idx1.size)
                rows.append(r_fwd)
                cols.append(c_fwd)
                datas.append(
                    np.full(r_fwd.shape[0], -float(weight), dtype=np.float32))
                rows.append(c_fwd)
                cols.append(r_fwd)
                datas.append(
                    np.full(r_fwd.shape[0], -float(weight), dtype=np.float32))

    if not rows:
        return sp.csr_matrix((y_train.shape[0], y_train.shape[0]),
                             dtype=np.float32)

    row = np.concatenate(rows)
    col = np.concatenate(cols)
    data = np.concatenate(datas)
    return sp.coo_matrix(
        (data, (row, col)), shape=(y_train.shape[0], y_train.shape[0])
    ).tocsr()


def main(dataset, seed=DEFAULT_SEED, test_ratio=DEFAULT_TEST_RATIO,
         val_ratio=DEFAULT_VAL_RATIO, split_name="random",
         config_overrides_path=None,
         constrained_loss=None, wd_mu=None, wd_cl_weight=None,
         wd_min_train_nodes=None,
         wd_mu_mode=None, wd_target_ratio=None,
         wd_mu_min=None, wd_mu_max=None, wd_mu_ema_decay=None,
         lr=None, lr_decay=None, lr_patience=None, lr_min=None,
         lr_threshold=None, lr_threshold_mode=None, lr_cooldown=None,
         eigenspace_diagnostic=None,
         label_q_pair_encoding=None,
         result_suffix=None,
         adjacency_source="graph",
         feature_adj_k=10,
         feature_adj_metric="cosine",
         no_task_aware=False):
    """
    Load data with a cold-start split, then run the SPARC pipeline.

    Args:
        dataset:     base dataset name, e.g. "cora", "citeseer", "pubmed".
        seed:        random seed for reproducibility.
        test_ratio:  fraction of nodes used for test.
        val_ratio:   fraction of nodes used for validation.
        split_name:  how test nodes are chosen: "random" | "original".
        config_overrides_path:  optional path to a JSON file whose contents
                     are deep-merged on top of the loaded dataset config.
                     Used by the hyperparameter tuning script.
        constrained_loss:  optional bool. When True, force-enable the
                     Wang-Davidson constrained-spectral-clustering
                     regularizer; when False, force-disable it. When
                     None (default), fall back to the config file; the
                     trainer's built-in default is True. Maps to
                     ``config.spectral.constrained_loss.enabled``.
        wd_mu, wd_cl_weight, wd_min_train_nodes:
                     optional CLI overrides for
                     ``config.spectral.constrained_loss.{mu,cl_weight,
                     min_train_nodes}``. Pass None to fall back to the
                     config file / trainer defaults.
        wd_mu_mode, wd_target_ratio, wd_mu_min, wd_mu_max,
        wd_mu_ema_decay:
                     optional CLI overrides for the auto-scaled mu
                     knobs on ``config.spectral.constrained_loss``.
                     When ``wd_mu_mode='auto'`` the trainer lets
                     ``ConstrainedSpectralLoss`` EMA-rescale mu
                     per-batch so ``|mu * wd_reward|`` targets
                     ``target_ratio * |spec_loss|``.
        lr, lr_decay, lr_patience, lr_min, lr_threshold,
        lr_threshold_mode, lr_cooldown:
                     optional CLI overrides for the
                     ``ReduceLROnPlateau`` knobs on
                     ``config.spectral``. ``lr_threshold`` and
                     ``lr_cooldown`` in particular are crucial on
                     large inductive graphs where noisy val-loss
                     epochs would otherwise spuriously decay the LR.
        eigenspace_diagnostic:
                     optional bool. When False, skip the post-training
                     eigenspace-agreement (Grassmann) diagnostic, which
                     is expensive on large graphs (it runs a full
                     eigendecomposition on the training Laplacian).
                     When True, force-enable it. When None (default),
                     fall back to ``config.diagnostics.eigenspace_agreement``
                     (which itself defaults to True).
        adjacency_source:
                     ``"graph"`` (default) uses dataset edges from
                     ``load_data``. ``"feature_knn"`` replaces ``train_adj``
                     and ``full_adj`` with symmetric feature kNN
                     (``build_symmetric_feature_knn_adjacency``) plus
                     self-loops and the same inductive train mask as the
                     graph path.
        feature_adj_k, feature_adj_metric:
                     Neighbor count and metric for ``adjacency_source ==
                     "feature_knn"`` (passed to ``NearestNeighbors``).
        no_task_aware:
                     When True, force ``spectral.task_aware.enabled`` false
                     and ``lambda`` 0 so the task-aware CE head is off.
        label_q_pair_encoding:
                     optional ``"binary"`` | ``"signed"``. Overrides
                     ``config.spectral.label_q_augmentation.pair_encoding``
                     when label-Q augmentation is enabled: ``binary`` uses
                     Q in ``{0, weight}`` (same-label links only); ``signed``
                     uses ``{-weight, 0, +weight}`` (same-label +weight,
                     different-label train pairs ``-weight``). Pass None to
                     use the config value (defaults to ``binary``).
        result_suffix:
                     optional tag for this run's output folder and a copy of
                     the embedding matrix. When non-empty: the directory name
                     is ``<split>_test<ratio>_seed<seed>`` plus this suffix
                     (if it starts with ``_``, it is appended as-is; otherwise
                     ``__<suffix>`` is used for compatibility with
                     ``SPARC_RESULT_SUFFIX``). Also writes
                     ``embeddings_<suffix>.npy`` with leading underscores
                     stripped from the filename stem (e.g. ``_test_binary``
                     → ``embeddings_test_binary.npy``). When None, only the
                     environment variable ``SPARC_RESULT_SUFFIX`` is
                     considered (legacy ``__`` join).
    """
    set_seed(seed)

    # Resolve config from dataset base name
    base = dataset
    for b in DATASET_BASES:
        if dataset == b or dataset.startswith(b + "_"):
            base = b
            break

    config_dir = os.path.join(_SCRIPT_DIR, "config")
    config_path = os.path.join(config_dir, base + ".json")
    with open(config_path, "r") as f:
        config = json.load(f)

    if config_overrides_path:
        # Accept a single path or a colon-separated list of paths. Stacks
        # are applied left-to-right so later files override earlier ones.
        paths = [p for p in config_overrides_path.split(":") if p]
        for p in paths:
            with open(p, "r") as f:
                _deep_merge(config, json.load(f))

    # Apply CLI overrides for the Wang-Davidson constrained loss. Only
    # the keys the user explicitly set on the CLI are merged in, so the
    # dataset config / JSON overrides retain authority when no CLI flag
    # is passed. The trainer itself defaults ``enabled`` to True when the
    # key is absent (so the new behavior is the default overall).
    cl_overrides = {}
    if constrained_loss is not None:
        cl_overrides["enabled"] = bool(constrained_loss)
    if wd_mu is not None:
        cl_overrides["mu"] = float(wd_mu)
    if wd_cl_weight is not None:
        cl_overrides["cl_weight"] = float(wd_cl_weight)
    if wd_min_train_nodes is not None:
        cl_overrides["min_train_nodes"] = int(wd_min_train_nodes)
    if wd_mu_mode is not None:
        cl_overrides["mu_mode"] = str(wd_mu_mode)
    if wd_target_ratio is not None:
        cl_overrides["target_ratio"] = float(wd_target_ratio)
    if wd_mu_min is not None:
        cl_overrides["mu_min"] = float(wd_mu_min)
    if wd_mu_max is not None:
        cl_overrides["mu_max"] = float(wd_mu_max)
    if wd_mu_ema_decay is not None:
        cl_overrides["mu_ema_decay"] = float(wd_mu_ema_decay)
    if cl_overrides:
        config.setdefault("spectral", {})
        config["spectral"].setdefault("constrained_loss", {})
        _deep_merge(config["spectral"]["constrained_loss"], cl_overrides)

    # LR-scheduler CLI overrides. When passed, these replace the
    # matching config.spectral.* values (patience maps to the
    # SpectralTrainer's ReduceLROnPlateau patience, which is the
    # number of epochs with no improvement before the LR is decayed).
    lr_overrides = {}
    if lr is not None:
        lr_overrides["lr"] = float(lr)
    if lr_decay is not None:
        lr_overrides["lr_decay"] = float(lr_decay)
    if lr_patience is not None:
        lr_overrides["patience"] = int(lr_patience)
    if lr_min is not None:
        lr_overrides["min_lr"] = float(lr_min)
    if lr_threshold is not None:
        lr_overrides["lr_threshold"] = float(lr_threshold)
    if lr_threshold_mode is not None:
        lr_overrides["lr_threshold_mode"] = str(lr_threshold_mode)
    if lr_cooldown is not None:
        lr_overrides["lr_cooldown"] = int(lr_cooldown)
    if lr_overrides:
        config.setdefault("spectral", {})
        _deep_merge(config["spectral"], lr_overrides)

    # CLI override for the post-training eigenspace-agreement diagnostic.
    # Useful on large graphs where the Laplacian eigendecomposition would
    # otherwise dominate wall-clock time.
    if eigenspace_diagnostic is not None:
        config.setdefault("diagnostics", {})
        config["diagnostics"]["eigenspace_agreement"] = bool(
            eigenspace_diagnostic)

    if label_q_pair_encoding is not None:
        enc = str(label_q_pair_encoding).lower().strip()
        if enc not in ("binary", "signed"):
            raise ValueError(
                "label_q_pair_encoding must be 'binary' or 'signed', "
                "got {!r}".format(label_q_pair_encoding))
        config.setdefault("spectral", {})
        config["spectral"].setdefault("label_q_augmentation", {})
        config["spectral"]["label_q_augmentation"]["pair_encoding"] = enc

    if no_task_aware:
        config.setdefault("spectral", {})
        ta = config["spectral"].setdefault("task_aware", {})
        ta["enabled"] = False
        ta["lambda"] = 0.0

    spectral = config["spectral"]
    label_q_cfg = spectral.get("label_q_augmentation", {}) or {}
    label_q_enabled = bool(label_q_cfg.get("enabled", False))
    if label_q_enabled and bool(label_q_cfg.get("disable_label_losses", True)):
        spectral.setdefault("task_aware", {})
        spectral["task_aware"]["enabled"] = False
        spectral["task_aware"]["lambda"] = 0.0
        spectral.setdefault("constrained_loss", {})
        spectral["constrained_loss"]["enabled"] = False
        spectral["spectral_loss_weight"] = 1.0

    n_clusters = config["n_clusters"]
    bsize = spectral["bsize"]
    num_clusters = spectral["num_clusters"]
    num_clusters_val = spectral["num_clusters_val"]
    num_clusters_test = spectral["num_clusters_test"]
    diag_lambda = spectral["diag_lambda"]
    output_dim = spectral["architecture"]["output_dim"]
    cache_partitions = spectral.get("cache_partitions", True)
    env_off = os.environ.get("SPARC_NO_PARTITION_CACHE", "").strip().lower()
    if env_off in ("1", "true", "yes"):
        cache_partitions = False
    cache_root = spectral.get("partition_cache_dir")
    if not cache_root:
        cache_root = os.path.join(_PROJECT_ROOT, "data", "partition_cache")

    graph_dir = os.path.join(_PROJECT_ROOT, "data", "data")

    print("=" * 60)
    print("Dataset: {}, split: {}, test_ratio: {}, val_ratio: {}, seed: {}".format(
        dataset, split_name, test_ratio, val_ratio, seed))
    print("=" * 60)

    print("Loading data...")
    (train_adj, full_adj, train_feats, test_feats, y_train, y_val, y_test,
     train_mask, val_mask, test_mask, idx_train, idx_val, idx_test,
     num_data, visible_data) = load_data(
        dataset, seed, test_ratio, val_ratio, split_name, graph_dir)

    adjacency_source = str(adjacency_source or "graph").lower().strip()
    if adjacency_source not in ("graph", "feature_knn"):
        raise ValueError(
            "adjacency_source must be 'graph' or 'feature_knn', got {!r}".format(
                adjacency_source))
    adjacency_is_feature_knn_only = adjacency_source == "feature_knn"
    if adjacency_is_feature_knn_only:
        k_feat = int(feature_adj_k)
        if k_feat < 1:
            raise ValueError("feature_adj_k must be >= 1, got {}".format(k_feat))
        met_feat = str(feature_adj_metric or "cosine")
        A_feat = build_symmetric_feature_knn_adjacency(
            train_feats, k=k_feat, metric=met_feat).tocsr()
        full_adj = A_feat + sp.eye(num_data, dtype=np.float32, format="csr")
        full_adj[full_adj > 1] = 1
        train_adj = build_inductive_adj(full_adj, idx_train)
        train_adj = train_adj + sp.eye(num_data, dtype=np.float32, format="csr")
        train_adj[train_adj > 1] = 1
        print(
            "[adjacency] feature_knn only: k={}, metric={}, nnz(train)={}, "
            "nnz(full)={}".format(
                k_feat, met_feat, int(train_adj.nnz), int(full_adj.nnz)))

    # -----------------------------------------------------------------
    # Optional label-Q adjacency augmentation.
    #
    # This is a graph-operator version of the task-aware signal: instead
    # of adding a CE/WD label loss, add same-label train-node connections
    # to A and then train with spectral loss only. The Q matrix is built
    # strictly from y_train/train_mask, so val/test labels are not used.
    # -----------------------------------------------------------------
    if label_q_enabled:
        q_weight = float(label_q_cfg.get("weight", 1.0))
        apply_to_full_adj = bool(label_q_cfg.get("apply_to_full_adj", True))
        pair_enc = str(label_q_cfg.get("pair_encoding", "binary")).lower()
        if pair_enc not in ("binary", "signed"):
            raise ValueError(
                "spectral.label_q_augmentation.pair_encoding must be "
                "'binary' or 'signed', got {!r}".format(pair_enc))
        q_adj = build_label_q_adj(
            y_train, train_mask, weight=q_weight, pair_encoding=pair_enc)
        train_nnz_before = int(train_adj.nnz)
        full_nnz_before = int(full_adj.nnz)
        train_adj = (train_adj.tocsr() + q_adj).tocsr()
        if apply_to_full_adj:
            full_adj = (full_adj.tocsr() + q_adj).tocsr()
        print(
            "[label-Q] enabled: pair_encoding={}, added label-structure Q to A "
            "(weight={}, q_nnz={}, train_adj nnz {} -> {}, full_adj nnz {} -> {})"
            .format(
                pair_enc, q_weight, int(q_adj.nnz), train_nnz_before,
                int(train_adj.nnz), full_nnz_before, int(full_adj.nnz)))
        if bool(label_q_cfg.get("disable_label_losses", True)):
            print("[label-Q] task-aware CE and WD constrained loss disabled; "
                  "training objective is spectral loss only.")

    # -----------------------------------------------------------------
    # Optional feature-similarity augmentation of train_adj / full_adj.
    # Adding edges between feature-similar nodes mitigates the high
    # connected-component count of inductive train splits (cora/citeseer
    # in particular have hundreds of tiny components, which produce
    # many degenerate 0-eigenvalues that an MLP cannot reproduce).
    #
    # Two modes are supported via ``augmentation.feature_knn.mode``:
    #   * ``"knn"`` (default, legacy): connect every node to its top-k
    #     feature neighbors. Adds up to ``n*k`` edges; can leave the
    #     train graph disconnected because the kNN edges that linked
    #     train and val/test nodes are dropped to keep the inductive
    #     split honest.
    #   * ``"bridge"``: add only the *minimum* number of edges (= one
    #     less than the connected-component count) to make the graph
    #     fully connected, picking the most-similar node pair across
    #     each pair of components (a Minimum Spanning Tree over
    #     components). This yields exactly one connected component on
    #     both train_adj (within ``idx_train``) and full_adj.
    # -----------------------------------------------------------------
    aug_cfg = (config.get("augmentation") or {}).get("feature_knn") or {}
    if bool(aug_cfg.get("enabled", False)):
        mode = str(aug_cfg.get("mode", "knn")).lower()
        skip_knn_aug = (
            adjacency_is_feature_knn_only
            and mode in ("knn", "feature_knn"))
        if skip_knn_aug:
            print(
                "[augmentation] Skipping augmentation.feature_knn in "
                "'knn' mode: adjacency is already feature-kNN-only "
                "(use mode 'bridge' in config if you need extra connectivity).")
        else:
            metric = str(aug_cfg.get("metric", "cosine"))

            def _nc(A, sub_idx=None):
                A2 = A.tocsr().copy()
                A2.setdiag(0)
                A2.eliminate_zeros()
                if sub_idx is not None:
                    A2 = A2[sub_idx][:, sub_idx]
                nc, _ = connected_components(A2, directed=False)
                return int(nc)

            train_mask_bool = np.asarray(train_mask).astype(bool)
            idx_tr = np.where(train_mask_bool)[0]

            # Count components both on the full n*n train_adj (which includes
            # isolated val/test self-loop singletons that the inductive split
            # leaves behind) and on the train-induced subgraph (the only
            # piece the bridge mode actually merges).
            nc_tr_before = _nc(train_adj)
            nc_tr_sub_before = _nc(train_adj, sub_idx=idx_tr)
            nc_full_before = _nc(full_adj)

            if mode in ("knn", "feature_knn"):
                k_aug = int(aug_cfg.get("k", 10))
                mutual = bool(aug_cfg.get("mutual", True))

                knn_full = utils.build_feature_knn_adj(
                    train_feats, k=k_aug, metric=metric, mutual=mutual)

                # Restrict the feature-kNN edges used for training to
                # train-train only (no val/test leakage on the inductive split).
                knn_tr_sub = knn_full.tocsr()[idx_tr][:, idx_tr].tocoo()
                knn_train_only = sp.csr_matrix(
                    (knn_tr_sub.data.astype(np.float32),
                     (idx_tr[knn_tr_sub.row], idx_tr[knn_tr_sub.col])),
                    shape=(num_data, num_data),
                )

                train_adj = train_adj.tocsr().maximum(knn_train_only).tocsr()
                full_adj = full_adj.tocsr().maximum(knn_full).tocsr()

                print("[augmentation] feature-kNN: k={}, metric={}, mutual={}".format(
                    k_aug, metric, mutual))

            elif mode in ("bridge", "mst", "mst_bridge"):
                # Minimum-spanning-tree over connected components, on
                # feature-distance. Each call adds exactly
                # (n_components - 1) undirected edges; combined with the
                # existing structural adjacency via ``maximum`` it produces
                # a graph with a single connected component.
                train_bridge, _comp_tr, n_br_tr = utils.build_component_bridge_adj(
                    train_feats, train_adj, metric=metric,
                    restrict_idx=idx_tr, verbose=False)
                full_bridge, _comp_full, n_br_full = utils.build_component_bridge_adj(
                    test_feats, full_adj, metric=metric,
                    restrict_idx=None, verbose=False)

                train_adj = train_adj.tocsr().maximum(train_bridge).tocsr()
                full_adj = full_adj.tocsr().maximum(full_bridge).tocsr()

                print(
                    "[augmentation] feature-bridge (MST over components): "
                    "metric={}".format(metric))
                print(
                    "[augmentation] bridge edges added: train={} (over idx_train), "
                    "full={}".format(n_br_tr, n_br_full))

            else:
                raise ValueError(
                    f"Unknown augmentation.feature_knn.mode: {mode!r}. "
                    f"Expected one of: 'knn', 'bridge'.")

            nc_tr_after = _nc(train_adj)
            nc_tr_sub_after = _nc(train_adj, sub_idx=idx_tr)
            nc_full_after = _nc(full_adj)

            print(
                "[augmentation] train_adj components (full n*n): {} -> {}".format(
                    nc_tr_before, nc_tr_after))
            print(
                "[augmentation] train_adj components (idx_train only): "
                "{} -> {}".format(nc_tr_sub_before, nc_tr_sub_after))
            print(
                "[augmentation] full_adj  components: {} -> {}".format(
                    nc_full_before, nc_full_after))
            print(
                "[augmentation] train_adj nnz: {}, full_adj nnz: {}".format(
                    int(train_adj.nnz), int(full_adj.nnz)))

    print("Partitioning graph...")

    train_cache_path = None
    val_cache_path = None
    test_cache_path = None
    if cache_partitions:
        fp_train = partition_cache.fingerprint_train(
            dataset, split_name, seed, test_ratio, val_ratio, num_clusters,
            train_adj, visible_data)
        train_cache_path = partition_cache.cache_file_path(
            cache_root, fp_train, "train")
        fp_val = partition_cache.fingerprint_full_graph(
            dataset, num_clusters_val, "val", full_adj)
        val_cache_path = partition_cache.cache_file_path(
            cache_root, fp_val, "val")
        fp_test = partition_cache.fingerprint_full_graph(
            dataset, num_clusters_test, "test", full_adj)
        test_cache_path = partition_cache.cache_file_path(
            cache_root, fp_test, "test")

    _, parts = partition_utils.partition_graph(
        train_adj, visible_data, num_clusters, cache_path=train_cache_path)
    parts = [np.array(pt) for pt in parts]

    print("Preprocessing data...")

    # Whether to pre-normalize support matrices to row-stochastic form
    # upstream of the spectral trainer. The default (False) is the
    # post-fixes behavior: pass the raw symmetric adjacency submatrix
    # through; the trainer builds a symmetric multihop affinity and
    # the SpectralNet loss handles D^{-1/2} rescaling. Set True to
    # reproduce the legacy "original" configuration for the ablation.
    preprocess_normalize = bool(
        config.get("spectral", {}).get("preprocess_normalize", False))

    (features_batches, support_batches, y_train_batches,
     train_mask_batches) = utils.preprocess_multicluster(
        train_adj, parts, train_feats, y_train, train_mask,
        num_clusters, bsize, diag_lambda, normalize=preprocess_normalize)

    (_, val_features_batches, val_support_batches, y_val_batches,
     val_mask_batches) = utils.preprocess(full_adj, test_feats, y_val, val_mask,
                                          np.arange(num_data),
                                          num_clusters_val,
                                          diag_lambda,
                                          partition_cache_path=val_cache_path,
                                          normalize=preprocess_normalize)

    (_, test_features_batches, test_support_batches, y_test_batches,
     test_mask_batches) = utils.preprocess(full_adj, test_feats, y_test,
                                           test_mask, np.arange(num_data),
                                           num_clusters_test,
                                           diag_lambda,
                                           partition_cache_path=test_cache_path,
                                           normalize=preprocess_normalize)
    idx_parts = list(range(len(parts)))

    train = (parts, features_batches, support_batches,
             y_train_batches, train_mask_batches)
    val = (idx_parts, val_features_batches,
           val_support_batches, y_val_batches, val_mask_batches)
    test = (idx_parts, test_features_batches, test_support_batches,
            y_test_batches, test_mask_batches)

    print("Training SpectralNet...")

    X = torch.tensor(test_feats, dtype=torch.float32)
    spectralnet = SpectralNet(n_clusters=n_clusters, config=config)
    spectralnet.fit(train, val)

    print("Predicting clusters...")
    Y = spectralnet.predict(X)
    Y = Y / np.sqrt(Y.shape[0])

    X_embedded = spectralnet.embed(X)
    labels = np.argmax(y_test, axis=1)

    # -----------------------------------------------------------------
    # Downstream clustering metrics (NMI / ARI / ACC).
    #
    # Y is the learned spectral embedding (rescaled by 1/sqrt(N)).
    # We KMeans it into ``n_clusters`` groups and score against the
    # ground-truth labels on the test set (headline), plus train / val
    # / full for sanity. Done here so every training run produces a
    # comparable metrics.json that downstream ablations can pick up.
    # -----------------------------------------------------------------
    print('Computing classification metrics...')
    try:
        km = KMeans(n_clusters=n_clusters, n_init=10,
                    random_state=seed).fit(Y)
        cluster_assignments_full = km.predict(Y)

        train_mask_bool = np.asarray(train_mask).astype(bool)
        val_mask_bool = np.asarray(val_mask).astype(bool)
        test_mask_bool = np.asarray(test_mask).astype(bool)

        def _score(mask_bool: np.ndarray) -> dict:
            if not mask_bool.any():
                return {"n": 0, "acc": float("nan"),
                        "nmi": float("nan"), "ari": float("nan")}
            y_sub = labels[mask_bool]
            c_sub = cluster_assignments_full[mask_bool]
            return {
                "n": int(mask_bool.sum()),
                "acc": Metrics.acc_score(c_sub, y_sub, n_clusters),
                "nmi": Metrics.nmi_score(c_sub, y_sub),
                "ari": Metrics.ari_score(c_sub, y_sub),
            }

        metrics_report = {
            "dataset": dataset,
            "split_name": split_name,
            "test_ratio": test_ratio,
            "val_ratio": val_ratio,
            "seed": seed,
            "n_clusters": int(n_clusters),
            "output_dim": int(output_dim),
            "test": _score(test_mask_bool),
            "val": _score(val_mask_bool),
            "train": _score(train_mask_bool),
            "full": _score(np.ones_like(test_mask_bool, dtype=bool)),
        }

        print("[metrics] dataset={}  n_clusters={}  output_dim={}".format(
            dataset, n_clusters, output_dim))
        for split in ("test", "val", "train", "full"):
            m = metrics_report[split]
            print("[metrics] {:5s}  n={:6d}  ACC={:.4f}  NMI={:.4f}  ARI={:.4f}".format(
                split, m["n"], m["acc"], m["nmi"], m["ari"]))
    except Exception as exc:
        print(f"[metrics] Failed to compute classification metrics: {exc}")
        metrics_report = {"error": str(exc)}

    print('Saving results...')
    # Optional result suffix: CLI ``result_suffix`` wins over
    # ``SPARC_RESULT_SUFFIX``. Leading ``_`` on the CLI value appends a
    # single-underscore segment (e.g. ``_test_binary`` → ``...seed42_test_binary``);
    # other non-empty values keep the legacy ``__suffix`` layout for harnesses.
    env_suffix = os.environ.get("SPARC_RESULT_SUFFIX", "").strip()
    extra = ""
    if result_suffix is not None and str(result_suffix).strip() != "":
        extra = str(result_suffix).strip()
    elif env_suffix:
        extra = env_suffix
    run_dir_name = "{}_test{}_seed{}".format(split_name, test_ratio, seed)
    embed_copy_name = None
    if extra:
        if result_suffix is not None and str(result_suffix).strip() != "":
            s = str(result_suffix).strip()
            if s.startswith("_"):
                run_dir_name = run_dir_name + s
            else:
                run_dir_name = "{}__{}".format(run_dir_name, s)
        else:
            run_dir_name = "{}__{}".format(run_dir_name, extra)
        stem = extra.lstrip("_").replace(os.sep, "_")
        if stem:
            embed_copy_name = "embeddings_{}.npy".format(stem)
    directory = os.path.join(
        _SCRIPT_DIR, '..', 'sparc_results', dataset, run_dir_name)
    directory = os.path.normpath(directory)
    os.makedirs(directory, exist_ok=True)

    np.save(os.path.join(directory, 'embeddings.npy'), Y)
    if embed_copy_name:
        np.save(os.path.join(directory, embed_copy_name), Y)
        print("[main] Also saved embedding copy: {}".format(embed_copy_name))
    np.save(os.path.join(directory, 'features.npy'), X)
    np.save(os.path.join(directory, 'labels.npy'), labels)
    np.save(os.path.join(directory, 'test_mask.npy'), test_mask)
    np.save(os.path.join(directory, 'val_mask.npy'), val_mask)
    np.save(os.path.join(directory, 'train_mask.npy'), train_mask)
    np.save(os.path.join(directory, 'X_embedded.npy'), X_embedded)

    try:
        with open(os.path.join(directory, 'metrics.json'), 'w') as f:
            json.dump(metrics_report, f, indent=2)
    except Exception as exc:
        print(f"[metrics] Could not save metrics.json: {exc}")

    # Persist the train adjacency so the diagnostic can be re-run offline.
    try:
        save_npz(os.path.join(directory, 'train_adj.npz'), train_adj.tocsr())
    except Exception as exc:
        print(f"[main] Could not save train_adj.npz: {exc}")

    # Persist the full (symmetric, self-looped) adjacency for downstream
    # consumers that need the full graph (e.g. SPARCphormer --space computed).
    try:
        save_npz(os.path.join(directory, 'full_adj.npz'), full_adj.tocsr())
    except Exception as exc:
        print(f"[main] Could not save full_adj.npz: {exc}")

    # -----------------------------------------------------------------
    # Eigenspace-agreement diagnostic
    # -----------------------------------------------------------------
    diag_cfg = config.get("diagnostics", {}) or {}
    if diag_cfg.get("eigenspace_agreement", True):
        max_nodes = int(diag_cfg.get("max_nodes", 150_000))
        n_train = int(train_mask.sum())
        if n_train > max_nodes:
            print(f"[diagnostics] Skipping eigenspace agreement: "
                  f"{n_train} train nodes > max_nodes={max_nodes}. "
                  f"Set diagnostics.max_nodes higher to force it.")
        else:
            try:
                k_diag = diag_cfg.get("k")
                k_diag = int(k_diag) if k_diag else int(output_dim)
                aff_order = diag_cfg.get("affinity_order")
                aff_order = int(aff_order) if aff_order else int(
                    config["spectral"].get("affinity_walk_order", 3))
                report = run_diagnostic(
                    Y_full=Y,
                    adj_train=train_adj,
                    train_mask=train_mask,
                    k=k_diag,
                    laplacian_type=diag_cfg.get("laplacian_type", "sym_adj"),
                    affinity_order=aff_order,
                    drop_first=bool(diag_cfg.get("drop_first", True)),
                    method=diag_cfg.get("eig_method", "auto"),
                    tol=float(diag_cfg.get("eig_tol", 1e-6)),
                    restrict_to_lcc=bool(
                        diag_cfg.get("restrict_to_lcc", False)),
                    verbose=True,
                )
                with open(os.path.join(directory,
                                       "eigenspace_agreement.json"), "w") as f:
                    json.dump(report, f, indent=2)
            except Exception as exc:
                print(f"[diagnostics] Eigenspace agreement failed: {exc}")


if __name__ == "__main__":
    _epilog = """
Examples
--------
Cora, random cold-start split, seed 42, 10% test nodes, task-aware CE off
(spectral.task_aware.enabled=false and lambda=0):

  python main.py --dataset cora --seed 42 --test_ratio 0.1 --split_name random \\
      --no_task_aware

Same, but use symmetric feature kNN (k=10, cosine) as A instead of graph edges:

  python main.py --dataset cora --seed 42 --test_ratio 0.1 --split_name random \\
      --no_task_aware --adjacency feature_knn --feature_adj_k 10

Equivalent task-aware off via JSON override (merge on top of config):

  {"spectral": {"task_aware": {"enabled": false, "lambda": 0.0}}}

  python main.py --dataset cora --seed 42 --test_ratio 0.1 --split_name random \\
      --config_overrides config_ablation/no_taskaware.json
""".strip()
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_epilog)
    parser.add_argument("--dataset", type=str, default="cora")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--test_ratio", type=float, default=DEFAULT_TEST_RATIO)
    parser.add_argument("--val_ratio", type=float, default=DEFAULT_VAL_RATIO)
    parser.add_argument("--split_name", type=str, default="random")
    parser.add_argument(
        "--config_overrides", type=str, default=None,
        help=("Optional path (or colon-separated list of paths) to JSON "
              "files whose contents are deep-merged on top of the dataset "
              "config. When multiple paths are given, later files override "
              "earlier ones. Used by the ablation harness to swap between "
              "original / post-fixes / post-all-tuning and to stack delta "
              "overrides on top of a base iteration."))
    parser.add_argument(
        "--adjacency", type=str, default="graph",
        choices=["graph", "feature_knn"],
        help=("graph: use dataset edges (default). feature_knn: replace A "
              "with symmetric feature kNN + self-loops (inductive train mask "
              "unchanged); see build_symmetric_feature_knn_adjacency."))
    parser.add_argument(
        "--feature_adj_k", type=int, default=10,
        help="k for --adjacency feature_knn (neighbors per node, excluding self).")
    parser.add_argument(
        "--feature_adj_metric", type=str, default="cosine",
        help=("Distance metric for --adjacency feature_knn "
              "(sklearn.neighbors.NearestNeighbors)."))
    parser.add_argument(
        "--no_task_aware", action="store_true",
        help=("Force spectral.task_aware off (enabled=false, lambda=0). "
              "Use for spectral-only training when configs enable the CE head."))

    # ------------------------------------------------------------------
    # Wang-Davidson constrained-spectral-clustering loss (default: ON).
    # ``--constrained_loss`` / ``--no_constrained_loss`` toggle it
    # explicitly; without either flag, the dataset config (or the
    # trainer's own default of True) decides.
    # ------------------------------------------------------------------
    wd_group = parser.add_mutually_exclusive_group()
    wd_group.add_argument(
        "--constrained_loss", dest="constrained_loss",
        action="store_true", default=None,
        help=("Enable the Wang-Davidson constrained-spectral-clustering "
              "regularizer during SpectralNet training. This is the new "
              "default; the flag is provided for completeness and to "
              "override an 'enabled: false' setting in the config."))
    wd_group.add_argument(
        "--no_constrained_loss", dest="constrained_loss",
        action="store_false",
        help=("Disable the Wang-Davidson regularizer and reproduce the "
              "vanilla SpectralNet objective."))
    parser.add_argument(
        "--wd_mu", type=float, default=None,
        help=("Weight of the Wang-Davidson constraint term in the total "
              "loss. Overrides config.spectral.constrained_loss.mu. "
              "Trainer default is 0.5."))
    parser.add_argument(
        "--wd_cl_weight", type=float, default=None,
        help=("Scalar multiplier applied to the cannot-link pair-sum "
              "inside the WD term. If omitted, auto-balances per batch "
              "so the ML and CL contributions have comparable total "
              "mass."))
    parser.add_argument(
        "--wd_min_train_nodes", type=int, default=None,
        help=("Skip the WD term on batches with fewer than this many "
              "labeled nodes. Default: 2."))
    parser.add_argument(
        "--wd_mu_mode", type=str, default=None,
        choices=["static", "auto"],
        help=("Scaling policy for the WD mu multiplier. 'static' uses "
              "the fixed --wd_mu value (trainer default). 'auto' "
              "EMA-rescales mu per-batch so that |mu * wd_reward| "
              "targets --wd_target_ratio * |spec_loss|; this is the "
              "recommended setting on large graphs where the "
              "pair-averaged cosine reward is orders of magnitude "
              "smaller than the Rayleigh-quotient spectral loss."))
    parser.add_argument(
        "--wd_target_ratio", type=float, default=None,
        help=("In auto mu_mode, target |mu * wd_reward| / |spec_loss|. "
              "Default: 0.2."))
    parser.add_argument(
        "--wd_mu_min", type=float, default=None,
        help="Lower clamp on auto-scaled mu. Default: 0.0.")
    parser.add_argument(
        "--wd_mu_max", type=float, default=None,
        help="Upper clamp on auto-scaled mu. Default: 1e6.")
    parser.add_argument(
        "--wd_mu_ema_decay", type=float, default=None,
        help=("Per-batch EMA decay for the auto-scaled mu. Larger "
              "values smooth more aggressively. Default: 0.9."))

    # ------------------------------------------------------------------
    # LR-scheduler knobs. These feed ``ReduceLROnPlateau`` parameters:
    # ``threshold`` + ``threshold_mode`` define how "significant"
    # val-loss improvement must be, and ``cooldown`` suppresses
    # back-to-back decays on noisy metrics. Needed on big inductive
    # graphs where the default ``threshold=1e-4, cooldown=0`` turns
    # epoch-to-epoch val-loss noise into spurious LR drops.
    # ------------------------------------------------------------------
    parser.add_argument(
        "--lr", type=float, default=None,
        help="Override config.spectral.lr (initial learning rate).")
    parser.add_argument(
        "--lr_decay", type=float, default=None,
        help=("Override config.spectral.lr_decay. Multiplicative factor "
              "applied to the LR when a plateau is detected."))
    parser.add_argument(
        "--lr_patience", type=int, default=None,
        help=("Override config.spectral.patience (number of bad "
              "epochs before the scheduler reduces the LR)."))
    parser.add_argument(
        "--lr_min", type=float, default=None,
        help=("Override config.spectral.min_lr (training stops once "
              "the LR falls at or below this value)."))
    parser.add_argument(
        "--lr_threshold", type=float, default=None,
        help=("Override config.spectral.lr_threshold. Minimum "
              "improvement (relative or absolute, per --lr_threshold_mode) "
              "required for an epoch to count as 'improving'. Default "
              "1e-4 is PyTorch's default; on noisy big-graph val loss, "
              "~0.01 (rel) works better."))
    parser.add_argument(
        "--lr_threshold_mode", type=str, default=None,
        choices=["rel", "abs"],
        help="Override config.spectral.lr_threshold_mode. Default: 'rel'.")
    parser.add_argument(
        "--lr_cooldown", type=int, default=None,
        help=("Override config.spectral.lr_cooldown. Number of epochs "
              "to wait after a decay before resuming 'bad epoch' "
              "counting. Default: 0."))

    # ------------------------------------------------------------------
    # Post-training eigenspace-agreement (Grassmann) diagnostic. Enabled
    # by default; can be disabled on large graphs where the full
    # Laplacian eigendecomposition is too expensive.
    # ------------------------------------------------------------------
    diag_group = parser.add_mutually_exclusive_group()
    diag_group.add_argument(
        "--eigenspace_diagnostic", dest="eigenspace_diagnostic",
        action="store_true", default=None,
        help=("Force-enable the post-training eigenspace-agreement "
              "diagnostic (Grassmann distance vs. the exact Laplacian "
              "spectrum). Overrides config.diagnostics.eigenspace_agreement."))
    diag_group.add_argument(
        "--no_eigenspace_diagnostic", dest="eigenspace_diagnostic",
        action="store_false",
        help=("Skip the post-training eigenspace-agreement diagnostic. "
              "Recommended on large graphs (e.g. ogbn-arxiv) where the "
              "eigendecomposition of the training Laplacian is the "
              "dominant cost."))
    parser.add_argument(
        "--label_q_pair_encoding", type=str, default=None,
        choices=["binary", "signed"],
        help=("When label-Q augmentation is enabled in the merged config, "
              "how to encode train-labeled pairs in Q before A+Q: 'binary' "
              "uses {0, weight} (same-label +weight only); 'signed' uses "
              "{-weight, 0, +weight} (+same / -different). Overrides "
              "spectral.label_q_augmentation.pair_encoding."))
    parser.add_argument(
        "--result_suffix", type=str, default=None,
        help=("Append to the sparc_results/<dataset>/ run folder name and "
              "write embeddings_<stem>.npy (stem = suffix without leading "
              "underscores). Use e.g. --result_suffix _test_binary for folder "
              "..._seed42_test_binary and embeddings_test_binary.npy. "
              "Overrides SPARC_RESULT_SUFFIX when set."))

    args = parser.parse_args()
    main(
        dataset=args.dataset,
        seed=args.seed,
        test_ratio=args.test_ratio,
        val_ratio=args.val_ratio,
        split_name=args.split_name,
        config_overrides_path=args.config_overrides,
        constrained_loss=args.constrained_loss,
        wd_mu=args.wd_mu,
        wd_cl_weight=args.wd_cl_weight,
        wd_min_train_nodes=args.wd_min_train_nodes,
        wd_mu_mode=args.wd_mu_mode,
        wd_target_ratio=args.wd_target_ratio,
        wd_mu_min=args.wd_mu_min,
        wd_mu_max=args.wd_mu_max,
        wd_mu_ema_decay=args.wd_mu_ema_decay,
        lr=args.lr,
        lr_decay=args.lr_decay,
        lr_patience=args.lr_patience,
        lr_min=args.lr_min,
        lr_threshold=args.lr_threshold,
        lr_threshold_mode=args.lr_threshold_mode,
        lr_cooldown=args.lr_cooldown,
        eigenspace_diagnostic=args.eigenspace_diagnostic,
        label_q_pair_encoding=args.label_q_pair_encoding,
        result_suffix=args.result_suffix,
        adjacency_source=args.adjacency,
        feature_adj_k=args.feature_adj_k,
        feature_adj_metric=args.feature_adj_metric,
        no_task_aware=args.no_task_aware,
    )
