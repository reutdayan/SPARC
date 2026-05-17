import json
import os
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.datasets import (
    HeterophilousGraphDataset, Planetoid, Reddit, WikiCS, WikipediaNetwork,
)
from torch_geometric.utils import to_undirected


def mask_to_idx(mask):
    mask = mask.view(-1).cpu().numpy().astype(bool)
    return np.where(mask)[0].astype(np.int64)


def edge_index_to_adj(edge_index, num_nodes, add_self_loops=False):
    edge_index = to_undirected(edge_index, num_nodes=num_nodes)
    row = edge_index[0].cpu().numpy()
    col = edge_index[1].cpu().numpy()
    data = np.ones(len(row), dtype=np.float32)

    adj = sp.coo_matrix((data, (row, col)), shape=(
        num_nodes, num_nodes), dtype=np.float32).tocsr()
    adj[adj > 1] = 1

    if add_self_loops:
        adj = adj + sp.eye(num_nodes, dtype=np.float32, format="csr")
        adj[adj > 1] = 1

    return adj


def load_pyg_dataset(dataset_name, base_dir="./data", wikics_split_idx=0):
    name = dataset_name.lower()

    if name in {"cora", "citeseer", "pubmed"}:
        pretty = {"cora": "Cora", "citeseer": "CiteSeer",
                  "pubmed": "PubMed"}[name]
        ds = Planetoid(root=str(Path(base_dir) / pretty), name=pretty)
        data = ds[0]
        x = data.x.cpu().numpy()
        y = data.y.cpu().numpy().reshape(-1)
        adj = edge_index_to_adj(data.edge_index, data.num_nodes)
        split = {
            "idx_train": mask_to_idx(data.train_mask),
            "idx_val": mask_to_idx(data.val_mask),
            "idx_test": mask_to_idx(data.test_mask),
        }
        return adj, x, y, split

    elif name == "reddit":
        ds = Reddit(root=str(Path(base_dir) / "Reddit"))
        data = ds[0]
        x = data.x.cpu().numpy()
        y = data.y.cpu().numpy().reshape(-1)
        adj = edge_index_to_adj(data.edge_index, data.num_nodes)
        split = {
            "idx_train": mask_to_idx(data.train_mask),
            "idx_val": mask_to_idx(data.val_mask),
            "idx_test": mask_to_idx(data.test_mask),
        }
        return adj, x, y, split

    elif name == "wikics":
        ds = WikiCS(root=str(Path(base_dir) / "WikiCS"))
        data = ds[0]
        x = data.x.cpu().numpy()
        y = data.y.cpu().numpy().reshape(-1)
        adj = edge_index_to_adj(data.edge_index, data.num_nodes)
        split = {
            "idx_train": mask_to_idx(data.train_mask[:, wikics_split_idx]),
            "idx_val": mask_to_idx(data.val_mask[:, wikics_split_idx]),
            "idx_test": mask_to_idx(data.test_mask),
        }
        return adj, x, y, split

    elif name in {"chameleon", "squirrel"}:
        pretty = name.capitalize()
        ds = WikipediaNetwork(
            root=str(Path(base_dir) / pretty), name=name, geom_gcn_preprocess=True
        )
        data = ds[0]
        x = data.x.cpu().numpy()
        y = data.y.cpu().numpy().reshape(-1)
        adj = edge_index_to_adj(data.edge_index, data.num_nodes)
        split_idx = wikics_split_idx
        if hasattr(data, "train_mask") and data.train_mask.dim() == 2:
            split = {
                "idx_train": mask_to_idx(data.train_mask[:, split_idx]),
                "idx_val": mask_to_idx(data.val_mask[:, split_idx]),
                "idx_test": mask_to_idx(data.test_mask[:, split_idx]),
            }
        else:
            split = {
                "idx_train": mask_to_idx(data.train_mask),
                "idx_val": mask_to_idx(data.val_mask),
                "idx_test": mask_to_idx(data.test_mask),
            }
        return adj, x, y, split

    elif name == "minesweeper":
        ds = HeterophilousGraphDataset(
            root=str(Path(base_dir) / "Minesweeper"), name="Minesweeper"
        )
        data = ds[0]
        x = data.x.cpu().numpy()
        y = data.y.cpu().numpy().reshape(-1)
        adj = edge_index_to_adj(data.edge_index, data.num_nodes)
        split_idx = wikics_split_idx
        if hasattr(data, "train_mask") and data.train_mask.dim() == 2:
            split = {
                "idx_train": mask_to_idx(data.train_mask[:, split_idx]),
                "idx_val": mask_to_idx(data.val_mask[:, split_idx]),
                "idx_test": mask_to_idx(data.test_mask[:, split_idx]),
            }
        else:
            split = {
                "idx_train": mask_to_idx(data.train_mask),
                "idx_val": mask_to_idx(data.val_mask),
                "idx_test": mask_to_idx(data.test_mask),
            }
        return adj, x, y, split

    elif name in {"ogbn-products", "ogbn-arxiv", "ogbn-mag"}:
        ds = PygNodePropPredDataset(
            name=name, root=str(Path(base_dir) / "OGB"))
        data = ds[0]
        x = data.x.cpu().numpy()
        y = data.y.view(-1).cpu().numpy()
        adj = edge_index_to_adj(data.edge_index, data.num_nodes)
        split_idx = ds.get_idx_split()
        split = {
            "idx_train": split_idx["train"].cpu().numpy().astype(np.int64),
            "idx_val": split_idx["valid"].cpu().numpy().astype(np.int64),
            "idx_test": split_idx["test"].cpu().numpy().astype(np.int64),
        }
        return adj, x, y, split

    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")


def save_original_split(dataset_name, split, out_dir="./data/graphsage_splits", split_name="original"):
    split_dir = Path(out_dir) / dataset_name / split_name
    split_dir.mkdir(parents=True, exist_ok=True)
    np.save(split_dir / "idx_train.npy", split["idx_train"])
    np.save(split_dir / "idx_val.npy", split["idx_val"])
    np.save(split_dir / "idx_test.npy", split["idx_test"])
    print(f"Saved original split for {dataset_name} to {split_dir}")


def save_graphsage_format(dataset_name, base_dir="./data", out_dir="./data/graphsage", split_out_dir="./data/graphsage_splits", wikics_split_idx=0):
    adj, features, labels, split = load_pyg_dataset(
        dataset_name, base_dir=base_dir, wikics_split_idx=wikics_split_idx)

    dataset_dir = Path(out_dir) / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)

    graph_dict = {
        "directed": False,
        "graph": [],
        "nodes": [],
        "links": [],
    }

    # nodes: no split info here; split is saved separately
    graph_dict["nodes"] = [
        {"id": str(i), "test": False, "val": False} for i in range(adj.shape[0])]

    adj_coo = adj.tocoo()
    links = []
    for s, t in zip(adj_coo.row, adj_coo.col):
        links.append({"source": int(s), "target": int(t)})
    graph_dict["links"] = links

    with open(dataset_dir / f"{dataset_name}-G.json", "w") as f:
        json.dump(graph_dict, f)

    with open(dataset_dir / f"{dataset_name}-feats.npy", "wb") as f:
        np.save(f, features)

    id_map = {str(i): i for i in range(adj.shape[0])}
    with open(dataset_dir / f"{dataset_name}-id_map.json", "w") as f:
        json.dump(id_map, f)

    class_map = {str(i): int(labels[i]) for i in range(len(labels))}
    with open(dataset_dir / f"{dataset_name}-class_map.json", "w") as f:
        json.dump(class_map, f)

    save_original_split(dataset_name, split,
                        out_dir=split_out_dir, split_name="original")
    print(f"Saved GraphSAGE format for {dataset_name} to {dataset_dir}")


if __name__ == "__main__":
    # Use the same base directory convention as in download_data.py:
    # this points to the folder where datasets have already been downloaded.
    base_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(base_dir, "data")
    split_out_dir = os.path.join(base_dir, "data_splits")

    for dataset in ["cora"]: #, "citeseer", "pubmed", "chameleon", "squirrel", "reddit", "ogbn-products", "ogbn-arxiv", "wikics"]:
        save_graphsage_format(
            dataset,
            base_dir=base_dir,
            out_dir=out_dir,
            split_out_dir=split_out_dir,
        )
        print(f"Saved GraphSAGE format for {dataset}")
    print("All datasets saved")
