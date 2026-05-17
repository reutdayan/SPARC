# SPARC spectral embeddings

Train **inductive cold-start spectral embeddings** for graph nodes. This directory is the core SPARC training entry point: it loads GraphSAGE-formatted graphs, builds an inductive train graph, partitions it with METIS, and fits a SpectralNet MLP that maps node features into a low-dimensional embedding space aligned with the graph Laplacian spectrum.

Downstream code (e.g. SPARCphormer, GraphSAGE fusion) consumes the saved `embeddings.npy` under `../sparc_results/`.

## Prerequisites

1. **Graph data** in GraphSAGE format. See [`../../data/README.md`](../../data/README.md) for download and layout. By default, `main.py` reads from `../../data/data/<dataset>/`.

2. **Python packages**: PyTorch, NumPy, SciPy, scikit-learn, NetworkX, Matplotlib, and a METIS binding (`metis` / `pymetis`). Install METIS itself via conda when possible:

   ```bash
   conda install -c conda-forge metis -y
   ```

3. **GPU** (optional): CUDA is used automatically when available.

## Quick start

From this directory (`SPARC/SPARC/src`):

```bash
python main.py --dataset cora --seed 42 --test_ratio 0.10 --val_ratio 0.10 --split_name random
```

Use the dataset’s canonical split instead of a random cold-start split:

```bash
python main.py --dataset cora --split_name original
```

Spectral-only training (no task-aware classification head):

```bash
python main.py --dataset cora --seed 42 --test_ratio 0.10 --split_name random --no_task_aware
```

Full CLI help:

```bash
python main.py --help
```

## Training pipeline

```
GraphSAGE graph  →  cold-start split  →  inductive train_adj + full_adj
        →  optional augmentations (feature kNN, label-Q, …)
        →  METIS partition (train)  →  batched preprocess
        →  SpectralNet / SpectralTrainer  →  node embeddings Y
        →  KMeans metrics + save sparc_results/<dataset>/<run>/
```

**Cold-start split.** Train nodes are the only nodes with labels used for supervision. The train adjacency keeps **train–train edges only**; val/test nodes are isolated at training time (inductive setting). Splits are created with `data/split_data.py` (`random` or `original`).

**SpectralNet.** An MLP projects features into `spectral.architecture.output_dim` dimensions. Training minimizes a Rayleigh-quotient style spectral loss on multihop affinities built from each METIS batch, optionally plus a Wang–Davidson constrained-clustering term and/or a task-aware CE head on labeled train nodes.

## Source layout

| File | Role |
|------|------|
| `main.py` | CLI, data loading, augmentations, partitioning, training, metrics, artifact export |
| `SpectralNet.py` | Thin wrapper: constructs `SpectralTrainer` and runs `fit` / `predict` |
| `SpectralTrainer.py` | SpectralNet model, losses, training loop, LR scheduling |
| `utils.py` | Batching, affinity matrices, feature kNN / bridge augmentation helpers |
| `partition_utils.py` | METIS graph partitioning |
| `partition_cache.py` | On-disk cache for partition results |
| `metrics.py` | NMI / ARI / ACC for clustering evaluation |
| `eigenspace_diagnostics.py` | Optional post-hoc Grassmann agreement with Laplacian eigenvectors |
| `config/<dataset>.json` | Per-dataset hyperparameters and augmentation defaults |

## Configuration

Each dataset has a JSON file under `config/`. Important sections:

- **`spectral`**: MLP architecture, learning rate, epochs, batch size (`bsize`), METIS cluster counts (`num_clusters`, `num_clusters_val`, `num_clusters_test`), `affinity_walk_order`, optional `task_aware`, `constrained_loss`, `label_q_augmentation`.
- **`augmentation.feature_knn`**: Optional extra edges from feature similarity (`knn` or `bridge` mode).
- **`diagnostics`**: Post-training eigenspace agreement (`eigenspace_agreement`, `max_nodes`).

Override any field without editing the base config:

```bash
python main.py --dataset cora --config_overrides path/to/overrides.json
```

Colon-separated paths merge left-to-right; later files win.

## Common CLI flags

| Flag | Description |
|------|-------------|
| `--dataset` | Dataset name (`cora`, `citeseer`, `pubmed`, `wikics`, `ogbn-arxiv`, …) |
| `--seed` | Random seed (default `42`) |
| `--test_ratio`, `--val_ratio` | Cold-start fractions (ignored for `--split_name original`) |
| `--split_name` | `random` or `original` |
| `--adjacency` | `graph` (default) or `feature_knn` |
| `--no_task_aware` | Disable the task-aware CE auxiliary loss |
| `--no_constrained_loss` | Disable Wang–Davidson regularizer (vanilla spectral loss) |
| `--no_eigenspace_diagnostic` | Skip expensive post-training diagnostic |
| `--result_suffix` | Tag for output folder and optional `embeddings_<tag>.npy` copy |

Environment variables:

- `SPARC_RESULT_SUFFIX` — append to run directory name (if `--result_suffix` is not set).
- `SPARC_NO_PARTITION_CACHE=1` — disable METIS partition disk cache.

## Outputs

Results are written to:

```
SPARC/SPARC/sparc_results/<dataset>/<split>_test<ratio>_seed<seed>[/suffix]/
```

Typical files:

| File | Contents |
|------|----------|
| `embeddings.npy` | Learned spectral embeddings `Y` (shape `N × output_dim`, row-normalized by `1/√N`) |
| `features.npy` | Scaled input features |
| `labels.npy` | Integer class labels |
| `train_mask.npy`, `val_mask.npy`, `test_mask.npy` | Boolean masks |
| `metrics.json` | KMeans clustering ACC / NMI / ARI on train, val, test, full |
| `train_adj.npz`, `full_adj.npz` | Adjacency used for training / full graph |
| `eigenspace_agreement.json` | Optional diagnostic report |

## Supported datasets

Config files exist for: `cora`, `citeseer`, `pubmed`, `chameleon`, `squirrel`, `wikics`, `reddit`, `ogbn-arxiv`, `ogbn-products`, `minesweeper`.

Add a new dataset by placing GraphSAGE files under `../../data/data/<name>/` and creating `config/<name>.json` (copy an existing config and tune `n_clusters` / `spectral` blocks).

## Programmatic use

```python
from main import load_data, set_seed
from SpectralNet import SpectralNet
import json, os

set_seed(42)
graph_dir = os.path.join("..", "..", "data", "data")
# ... load config, partition, preprocess (see main.main) ...
# spectralnet = SpectralNet(n_clusters=7, config=config)
# spectralnet.fit(train, val)
# Y = spectralnet.predict(torch.tensor(features, dtype=torch.float32))
```

For a full run with partitioning and I/O, calling `main.main(...)` or the CLI is recommended.
