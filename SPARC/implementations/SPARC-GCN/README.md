# SPARC-GCN

Inductive **Graph Convolution Network** baseline that classifies nodes using **raw features** and **SPARC spectral embeddings** (`embeddings.npy`) as a second input channel. Training uses METIS-partitioned minibatches on the **train–train** adjacency (cold-start safe); val/test use the full graph.

## Prerequisites

1. **SPARC embeddings** from [`../../src/README.md`](../../src/README.md):

   ```bash
   cd ../../src
   python main.py --dataset cora --seed 42 --test_ratio 0.03 --split_name random
   ```

   Outputs land in `../../sparc_results/<dataset>/<split>_test<ratio>_seed<seed>/`.

2. **TensorFlow 1.x** (`tensorflow.compat.v1`), NumPy, SciPy, scikit-learn, NetworkX, and **METIS** (`metis` Python package + `libmetis`).

## Quick start

From this directory (`implementations/SPARC-GCN`):

```bash
# SPARC run only (recommended): features, embeddings, masks, adj from sparc_results
python train.py \
  --dataset cora \
  --use_sparc_only True \
  --epochs 75
# Resolves sparc_results/cora/random_test0.03_seed42/ by default (split/seed/test_ratio).
# Or pass --sparc_run random_test0.03_seed42 explicitly.
```

With GraphSAGE files as fallback for structure (embeddings still from SPARC):

```bash
python train.py \
  --dataset cora \
  --data_prefix ../../../data/data \
  --sparc_run random_test0.03_seed42
```

## What the model sees

| Input | Source |
|-------|--------|
| Node features | `features.npy` in the SPARC run dir (or GraphSAGE `-feats.npy`) |
| Spectral channel | `embeddings.npy` (SPARC spectral encoding) |
| Train adjacency | Inductive train graph (`train_adj` / train–train edges) |
| Masks | `train_mask.npy`, `val_mask.npy`, `test_mask.npy` from the SPARC run |

Non-train rows are zeroed in **training** feature/spectral tensors so the loss never reads val/test coordinates; evaluation still uses full tensors on the full adjacency.

## Useful flags

| Flag | Description |
|------|-------------|
| `--dataset` | Dataset name (`cora`, `citeseer`, …) |
| `--use_sparc_only` | Load everything from `sparc_results` only |
| `--sparc_results_root` | Default: `../../sparc_results` |
| `--sparc_run` | Run folder (e.g. `random_test0.03_seed42`); empty = use `--split_name` / `--test_ratio` / `--seed` |
| `--split_name` | Split strategy when `--sparc_run` is empty (default `random`) |
| `--test_ratio` | Test fraction when `--sparc_run` is empty (default `0.03`, matches `main.py`) |
| `--seed` | Seed when `--sparc_run` is empty (default `42`) |
| `--num_layers` | GCN depth (default `5`) |
| `--hidden1` | Hidden width (default `2048`) |
| `--num_clusters` | METIS partitions for training batches |
| `--random_batches` | Random node shards instead of METIS |
| `--adj_identity` | Replace adjacency with identity (no message passing) |
| `--multilabel` | Sigmoid + multilabel metrics (e.g. PPI) |

## Layout

| File | Role |
|------|------|
| `train.py` | Entry point, data loading, training loop |
| `models.py` | GCN model (dual inputs: features + spectral) |
| `layers.py` | Graph convolution layers |
| `utils.py` | Preprocessing, SPARC/GraphSAGE loaders, feed dicts |
| `partition_utils.py` | METIS partitioning |
| `metrics.py` | Accuracy / F1 |

## Outputs

Training logs accuracy and micro/macro F1 on train, val, and test via TensorFlow logging. No separate results CSV is written by default; capture stdout or extend `train.py` if you need structured logs.
