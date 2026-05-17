import json
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import torch


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
    )
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(indices, values, shape).coalesce()


def load_graphsage_graph(dataset_name, graph_dir="./data"):
    dataset_dir = Path(graph_dir) / dataset_name

    with open(dataset_dir / f"{dataset_name}-G.json", "r") as f:
        graph_dict = json.load(f)

    with open(dataset_dir / f"{dataset_name}-feats.npy", "rb") as f:
        features = np.load(f)

    with open(dataset_dir / f"{dataset_name}-id_map.json", "r") as f:
        id_map = json.load(f)

    with open(dataset_dir / f"{dataset_name}-class_map.json", "r") as f:
        class_map = json.load(f)

    num_nodes = len(id_map)

    # labels
    labels = np.zeros(num_nodes, dtype=np.int64)
    for k, v in class_map.items():
        labels[id_map[k]] = int(v)

    # adjacency
    links = graph_dict["links"]
    row = [int(link["source"]) for link in links]
    col = [int(link["target"]) for link in links]
    data = np.ones(len(row), dtype=np.float32)

    adj = sp.coo_matrix((data, (row, col)), shape=(
        num_nodes, num_nodes), dtype=np.float32).tocsr()
    adj[adj > 1] = 1

    return adj, features, labels


def load_split(dataset_name, split_name, split_dir="./data_splits"):
    split_path = Path(split_dir) / dataset_name / split_name
    idx_train = np.load(split_path / "idx_train.npy")
    idx_val = np.load(split_path / "idx_val.npy")
    idx_test = np.load(split_path / "idx_test.npy")
    return idx_train, idx_val, idx_test


def build_inductive_adj(adj_full, idx_train):
    """
    Keep only train-train edges.
    This makes both val and test inductive.
    """
    train_mask = np.zeros(adj_full.shape[0], dtype=bool)
    train_mask[idx_train] = True

    adj_coo = adj_full.tocoo()
    keep = train_mask[adj_coo.row] & train_mask[adj_coo.col]

    adj_train = sp.coo_matrix(
        (adj_coo.data[keep], (adj_coo.row[keep], adj_coo.col[keep])),
        shape=adj_full.shape,
        dtype=np.float32,
    ).tocsr()

    adj_train[adj_train > 1] = 1
    return adj_train


def load_graph_and_split_for_training(
    dataset_name,
    split_name,
    graph_dir="./data",
    split_dir="./data_splits",
    add_self_loops=True,
):
    adj_full, features, labels = load_graphsage_graph(
        dataset_name, graph_dir=graph_dir)
    idx_train, idx_val, idx_test = load_split(
        dataset_name, split_name, split_dir=split_dir)

    # inductive graph: only train-train edges
    adj_train = build_inductive_adj(adj_full, idx_train)

    if add_self_loops:
        adj_train = adj_train + \
            sp.eye(adj_train.shape[0], dtype=np.float32, format="csr")
        adj_train[adj_train > 1] = 1

    features = torch.tensor(features, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.long)

    idx_train = torch.tensor(idx_train, dtype=torch.long)
    idx_val = torch.tensor(idx_val, dtype=torch.long)
    idx_test = torch.tensor(idx_test, dtype=torch.long)

    adj_full_torch = sparse_mx_to_torch_sparse_tensor(adj_full)
    adj_train_torch = sparse_mx_to_torch_sparse_tensor(adj_train)

    return {
        "adj_full": adj_full_torch,     # optional for analysis/debug
        "adj_train": adj_train_torch,   # use this for training
        "features": features,
        "labels": labels,
        "idx_train": idx_train,
        "idx_val": idx_val,
        "idx_test": idx_test,
    }
