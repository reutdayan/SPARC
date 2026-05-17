# SPARC data

Scripts to download graph datasets, convert them to GraphSAGE format, create train/val/test splits, and load them for training.

## Layout

```
data/
├── download_data.py   # PyG/OGB → GraphSAGE files + original split
├── load_data.py       # load graphs and splits
├── split_data.py      # create random or original cold-start splits
├── data/              # GraphSAGE graphs (gitignored)
│   └── <dataset>/
│       ├── <dataset>-G.json
│       ├── <dataset>-feats.npy
│       ├── <dataset>-id_map.json
│       └── <dataset>-class_map.json
└── data_splits/       # split indices (gitignored)
    └── <dataset>/
        ├── original/
        └── <split_name>/
            ├── idx_train.npy
            ├── idx_val.npy
            └── idx_test.npy
```

Run commands from this directory (`SPARC/data/`) so default paths resolve correctly.

## Download (GraphSAGE format)

Requires `torch`, `torch_geometric`, and `ogb` (for OGB datasets).

```bash
python download_data.py
```

Edit the `dataset` list in `download_data.py` (`__main__`) to choose datasets. Supported names include `cora`, `citeseer`, `pubmed`, `chameleon`, `squirrel`, `wikics`, `reddit`, `ogbn-arxiv`, `ogbn-products`, `ogbn-mag`, `minesweeper`.

This writes GraphSAGE files under `data/<dataset>/` and the dataset’s canonical split under `data_splits/<dataset>/original/`.

## Load

```python
from load_data import load_graphsage_graph, load_split, load_graph_and_split_for_training

# graph only
adj, features, labels = load_graphsage_graph("cora", graph_dir="data")

# split indices
idx_train, idx_val, idx_test = load_split("cora", "original", split_dir="data_splits")

# graph + split, inductive train adjacency, torch tensors (for training)
batch = load_graph_and_split_for_training(
    "cora",
    split_name="original",
    graph_dir="data",
    split_dir="data_splits",
)
```

`load_graph_and_split_for_training` builds an inductive adjacency (train–train edges only) for training; val and test nodes are cold-start at inference time.

## Split

Splits are **random** (cold-start, uniform node sampling) or **original** (canonical train/val/test from the source dataset).

```python
import scipy.sparse as sp
from load_data import load_graphsage_graph
from split_data import make_cold_start_split, save_split

adj, features, labels = load_graphsage_graph("cora", graph_dir="data")
adj_sym = adj + adj.T
adj_sym[adj_sym > 1] = 1

# random cold-start split
idx_train, idx_val, idx_test = make_cold_start_split(
    adj_sym,
    test_frac=0.10,
    val_frac=0.10,
    test_strategy="random",
    seed=42,
)
save_split("cora", "random_test0.10_val0.10_seed42", idx_train, idx_val, idx_test)

# canonical split from download
idx_train, idx_val, idx_test = make_cold_start_split(
    adj_sym,
    test_strategy="original",
    dataset_name="cora",
)
save_split("cora", "original", idx_train, idx_val, idx_test)
```

When `test_strategy="original"`, `test_frac`, `val_frac`, and `seed` are ignored; val/test come from `data_splits/<dataset>/original/` (created by `download_data.py`).
