# coding=utf-8
"""Disk cache for METIS graph partitions (avoids recomputation across runs)."""

import hashlib
import os

import numpy as np

_CACHE_FORMAT_VERSION = 1
_METIS_SEED_TAG = b"metis_seed_42"


def _sparse_sample_fingerprint(adj):
    """Lightweight fingerprint of a sparse matrix (not cryptographically unique)."""
    adj = adj.tocsr()
    h = hashlib.sha256()
    h.update(np.asarray(adj.shape, dtype=np.int64).tobytes())
    h.update(np.int64(adj.nnz).tobytes())
    if adj.nnz > 0:
        n = min(4096, len(adj.data))
        h.update(np.ascontiguousarray(adj.data[:n]).tobytes())
        h.update(np.ascontiguousarray(adj.indices[:n]).tobytes())
        m = min(len(adj.indptr), 1024)
        h.update(np.ascontiguousarray(adj.indptr[:m]).tobytes())
    return h.hexdigest()


def fingerprint_train(dataset, split_name, seed, test_ratio, val_ratio, num_clusters,
                    train_adj, idx_train):
    """Cache key for partitioning the inductive train subgraph."""
    h = hashlib.sha256()
    h.update(b"v%d" % _CACHE_FORMAT_VERSION)
    h.update(_METIS_SEED_TAG)
    h.update(dataset.encode("utf-8"))
    h.update(split_name.encode("utf-8"))
    h.update(str(seed).encode("ascii"))
    h.update(str(test_ratio).encode("ascii"))
    h.update(str(val_ratio).encode("ascii"))
    h.update(str(num_clusters).encode("ascii"))
    h.update(b"train")
    h.update(_sparse_sample_fingerprint(train_adj).encode("ascii"))
    idx = np.asarray(idx_train, dtype=np.int64)
    h.update(idx.tobytes())
    return h.hexdigest()


def fingerprint_full_graph(dataset, num_clusters, kind, full_adj):
    """Cache key for partitioning the full graph (val/test preprocessing).

    Does not depend on train/val/test split: only the loaded adjacency matters.
    """
    h = hashlib.sha256()
    h.update(b"v%d" % _CACHE_FORMAT_VERSION)
    h.update(_METIS_SEED_TAG)
    h.update(dataset.encode("utf-8"))
    h.update(str(num_clusters).encode("ascii"))
    h.update(kind.encode("ascii"))
    h.update(_sparse_sample_fingerprint(full_adj).encode("ascii"))
    return h.hexdigest()


def cache_file_path(cache_root, fingerprint, role):
    """Return path to a single .npz cache file."""
    os.makedirs(cache_root, exist_ok=True)
    return os.path.join(cache_root, "%s_%s.npz" % (fingerprint[:32], role))


def save_partition_cache(path, adj, idx_nodes, groups, num_all_nodes, num_clusters):
    """Write METIS group labels and metadata for later reload."""
    if not path:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    adj_fp = _sparse_sample_fingerprint(adj)
    np.savez_compressed(
        path,
        idx_nodes=np.asarray(idx_nodes, dtype=np.int64),
        groups=np.asarray(groups, dtype=np.int32),
        num_all_nodes=np.int32(num_all_nodes),
        num_clusters=np.int32(num_clusters),
        adj_fp=np.array(adj_fp, dtype="S64"),
        fmt=np.int32(_CACHE_FORMAT_VERSION),
    )


def try_load_partition_cache(path, adj, idx_nodes, num_clusters):
    """Return groups as int32 ndarray, or None if cache miss / invalid."""
    if not path or not os.path.isfile(path):
        return None
    try:
        z = np.load(path, allow_pickle=False)
        if int(z["fmt"]) != _CACHE_FORMAT_VERSION:
            return None
        idx_s = np.asarray(z["idx_nodes"], dtype=np.int64)
        groups = np.asarray(z["groups"], dtype=np.int32)
        n_all = int(z["num_all_nodes"])
        k = int(z["num_clusters"])
        adj_fp = z["adj_fp"].item()
        if isinstance(adj_fp, bytes):
            adj_fp = adj_fp.decode("ascii").rstrip("\x00")
        else:
            adj_fp = str(adj_fp)
    except (IOError, OSError, KeyError, ValueError, TypeError):
        return None

    idx_nodes = np.asarray(idx_nodes, dtype=np.int64)
    if adj.shape[0] != n_all or k != num_clusters:
        return None
    if idx_s.shape != idx_nodes.shape or not np.array_equal(idx_s, idx_nodes):
        return None
    if adj_fp != _sparse_sample_fingerprint(adj):
        return None
    if groups.shape[0] != len(idx_nodes):
        return None
    if num_clusters > 1 and (groups.max() >= num_clusters or groups.min() < 0):
        return None
    return groups
