# SPARC-SAGE

**GraphSAGE** baseline where the graph is **not** the original citation/network edges. Each node connects to its top-**K** nearest **training** nodes in **SPARC embedding space** (synthetic kNN graph). Val/test nodes are cold-start at training time but aggregate from their SPARC neighbors at evaluation.

Also called **GraphSAGESPARC** / **SPARC-KNN** in logs.

## Prerequisites

1. Train SPARC embeddings first ([`../../src/README.md`](../../src/README.md)):

   ```bash
   cd ../../src
   python main.py --dataset cora --seed 42 --test_ratio 0.10 --split_name random
   ```

2. **TensorFlow 2.x** with TF 1.x compatibility (`graphsage/tf_compat.py`). **NetworkX 2.x+** (uses `G.nodes[n]`, not legacy `G.node[n]`). NumPy, SciPy, scikit-learn.

## Quick start

### Single run (Python API)

From `implementations/SPARC-SAGE`:

```python
from graphsage.supervised_train import main

results = main(
    dataset="cora",
    seed=42,
    test_ratio=0.10,
    val_ratio=0.10,
    split_name="random",
    sparc_topk=10,
)
print(results)  # train/val/test acc and F1
```

### CLI

Run from **`implementations/SPARC-SAGE`** (recommended):

```bash
cd implementations/SPARC-SAGE
python -m graphsage.supervised_train \
  --cli_dataset cora \
  --cli_seed 42 \
  --cli_test_ratio 0.10 \
  --cli_split_name random \
  --sparc_topk 10
```

Or from `graphsage/` (also works after the path fix):

```bash
cd implementations/SPARC-SAGE/graphsage
python supervised_train.py \
  --cli_dataset cora \
  --cli_seed 42 \
  --cli_test_ratio 0.10 \
  --cli_split_name random \
  --sparc_topk 10
```

### Batch benchmark

Edit dataset/seed lists in `run_all_experiments.py`, then:

```bash
python run_all_experiments.py
```

Appends rows to `random_split_results_graphsagesparc_all_datasets.csv`.

## SPARC run directory

Resolved automatically under:

```
../../sparc_results/<dataset>/<split>_test<test_ratio>_seed<seed>/
```

Example: `random_test0.1_seed42` for `--cli_split_name random --cli_test_ratio 0.1 --cli_seed 42`.

Required files: `embeddings.npy`, `features.npy`, `labels.npy`, `train_mask.npy`, `val_mask.npy`, `test_mask.npy`.

### Matching an existing SPARC run

**Option 1 — pass the same `test_ratio` as SPARC `main.py`:**

```bash
python -m graphsage.supervised_train \
  --cli_dataset cora \
  --cli_seed 42 \
  --cli_test_ratio 0.03 \
  --cli_split_name random \
  --sparc_topk 10
```

**Option 2 — point at the folder by name (`--sparc_run`):**

```bash
python -m graphsage.supervised_train \
  --cli_dataset cora \
  --sparc_run random_test0.03_seed42 \
  --sparc_topk 10
```

(`cli_test_ratio` / `cli_seed` are ignored for path resolution when `--sparc_run` is set.)

**Option 3 — create a new SPARC run with the ratio you want:**

```bash
cd ../../src
python main.py --dataset cora --seed 42 --test_ratio 0.10 --split_name random
```

Then run GraphSAGE with `--cli_test_ratio 0.10` (or `--sparc_run random_test0.1_seed42`).

| Flag | Description |
|------|-------------|
| `--cli_test_ratio` | Must match the SPARC run folder (`0.03` → `random_test0.03_seed42`) |
| `--cli_seed`, `--cli_split_name` | Must match SPARC run as well |
| `--sparc_results_root` | Default: `../../sparc_results` |
| `--sparc_run` | Exact run folder name under `sparc_results/<dataset>/` |
| `--sparc_topk` | Neighbors per node in embedding kNN (default `10`) |
| `--sparc_knn_metric` | `sklearn` distance metric (default `minkowski`) |

## How the synthetic graph works

1. Fit `NearestNeighbors` on **train** nodes in SPARC embedding space.
2. Each node gets edges to its `sparc_topk` closest train nodes (val/test are never neighbors of anyone during training).
3. `construct_adj` keeps train–train edges only for training; `construct_test_adj` keeps all kNN edges for val/test inference.

Raw `features.npy` from the SPARC run are used as GraphSAGE input features; structure comes entirely from SPARC kNN.

## Layout

```
SPARC-SAGE/
├── run_all_experiments.py    # grid runner → CSV
└── graphsage/
    ├── supervised_train.py   # main entry + SPARC kNN graph builder
    ├── supervised_models.py
    ├── models.py, layers.py, aggregators.py
    ├── minibatch.py, neigh_samplers.py
    └── metrics.py, utils.py
```

## Outputs

- **Programmatic `main()`**: dict with `train_acc`, `val_acc`, `test_acc`, micro/macro F1, etc.
- **`run_all_experiments.py`**: CSV with one row per (dataset, seed, split, test_ratio, topk).
- **TensorFlow logs** under `sup-sparc-<dataset>/` (see `base_log_dir` flag).

## Notes

- `--cli_val_ratio` is recorded for parity but **masks come from the SPARC run**; val ratio must match what SPARC used when creating that run.
- `--cli_split_name` should be `random` or `original` (matching [`../../data/split_data.py`](../../data/split_data.py)).
- Default model: `graphsage_mean` with two layers (`samples_1=25`, `samples_2=10`); tune via TensorFlow flags in `supervised_train.py`.
