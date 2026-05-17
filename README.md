# SPARC: Leveraging Graph Geometry for Cold-Start Node Prediction

Reference implementation for the paper

> **Leveraging Graph Geometry for Cold-Start Node Prediction.**

SPARC learns an **inductive spectral encoder** that maps node features into a low-dimensional space aligned
with the graph Laplacian spectrum. The encoder is trained on a *cold-start* split where
validation and test nodes are isolated from the graph during training, and is then used at
inference time to embed previously unseen nodes from features alone. The resulting
embeddings serve as drop-in geometry for downstream classifiers — we provide three:
**SPARC-GCN**, **SPARC-SAGE**, and **SPARCphormer**.

---

## Highlights

- **Inductive spectral embeddings.** A small MLP (`SpectralNet`) trained with a
  Rayleigh-quotient style loss on multihop affinities of METIS minibatches. At test time
  the encoder produces embeddings for cold-start nodes from features alone — no retraining,
  no eigendecomposition of the full graph.
- **Cold-start by construction.** The training adjacency keeps train–train edges only;
  val/test nodes are never observed during message passing. Splits are either *random*
  (uniform cold-start sampling) or *original* (canonical dataset split).
- **Three downstream classifiers.**
  - `SPARC-GCN` — GCN on the real graph with raw features + SPARC embeddings as a second
    input channel.
  - `SPARC-SAGE` — GraphSAGE on a *synthetic* kNN graph built in SPARC embedding space.
  - `SPARCphormer` — Transformer over multi-hop token sequences in a configurable retrieval
    space (SPARC, raw features, Laplacian eigenvectors, Cold-Brew, or fusion).
- **Benchmarked datasets.** `cora`, `citeseer`, `pubmed`, `chameleon`, `squirrel`,
  `wikics`, `reddit`, `ogbn-arxiv`, `ogbn-products`, `ogbn-mag`, `minesweeper`.

---

## Repository layout

```
SPARC/
├── data/                            # Dataset download, splitting, loading
│   ├── download_data.py             # PyG/OGB → GraphSAGE format + canonical split
│   ├── split_data.py                # Random / original cold-start splits
│   ├── load_data.py                 # Loaders + inductive adjacency builder
│   └── README.md
│
├── SPARC/
│   ├── src/                         # SPARC spectral encoder (training entry point)
│   │   ├── main.py                  # CLI: data → partition → train → metrics → save
│   │   ├── SpectralNet.py           # MLP wrapper around SpectralTrainer
│   │   ├── SpectralTrainer.py       # Model, losses, training loop
│   │   ├── neighborhood_prediction.py
│   │   ├── eigenspace_diagnostics.py
│   │   ├── partition_utils.py / partition_cache.py
│   │   ├── metrics.py / utils.py
│   │   ├── config/<dataset>.json    # Per-dataset hyperparameters
│   │   └── README.md
│   │
│   ├── implementations/
│   │   ├── SPARC-GCN/               # GCN on real graph + SPARC channel  (TF 1.x compat)
│   │   ├── SPARC-SAGE/              # GraphSAGE on SPARC-kNN synthetic graph (TF 2.x compat)
│   │   └── SPARCphormer/            # Transformer on multi-hop token sequences (PyTorch)
│   │
│   └── sparc_results/<dataset>/<run>/   # Trained embeddings + masks + metrics (gitignored)
│
├── environment.yml                  # Conda env (Python 3.9 + METIS)
├── requirements.txt                 # Pinned pip dependencies
└── README.md                        # (this file)
```

Each component has its own README with full CLI flags and details:

- [`data/README.md`](data/README.md)
- [`SPARC/src/README.md`](SPARC/src/README.md)
- [`SPARC/implementations/SPARC-GCN/README.md`](SPARC/implementations/SPARC-GCN/README.md)
- [`SPARC/implementations/SPARC-SAGE/README.md`](SPARC/implementations/SPARC-SAGE/README.md)
- [`SPARC/implementations/SPARCphormer/README.md`](SPARC/implementations/SPARCphormer/README.md)

---

## Installation

### 1. Conda environment (recommended)

The conda recipe installs Python 3.9 and the native **METIS** library, which is required
for graph partitioning during training.

```bash
conda env create -f environment.yml
conda activate SPARC
pip install -r requirements.txt
```

If you prefer pip-only:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# Install METIS separately (system package or `conda install -c conda-forge metis`)
```

### 2. Component-specific dependencies

The downstream classifiers use heterogeneous frameworks. Install only the ones you need:

| Component       | Framework                                       |
|-----------------|-------------------------------------------------|
| `SPARC/src`     | PyTorch + NumPy / SciPy / scikit-learn / METIS  |
| `SPARC-GCN`     | TensorFlow 1.x (`tensorflow.compat.v1`) + METIS |
| `SPARC-SAGE`    | TensorFlow 2.x (with TF 1.x compat) + NetworkX  |
| `SPARCphormer`  | PyTorch                                         |

A CUDA-enabled GPU is optional but recommended for the larger datasets
(`reddit`, `ogbn-arxiv`, `ogbn-products`).

---

## Quick start (Cora, end-to-end)

The pipeline runs in three stages: **download → train SPARC → run a downstream classifier**.

```bash
# 1. Download Cora and convert to GraphSAGE format
cd data
python download_data.py            # edit __main__ to choose datasets
cd ..

# 2. Train SPARC spectral embeddings (random cold-start split, seed 42)
cd SPARC/src
python main.py --dataset cora --seed 42 --test_ratio 0.10 --val_ratio 0.10 --split_name random
# → SPARC/sparc_results/cora/random_test0.10_seed42/{embeddings,features,labels,*_mask}.npy

# 3a. Downstream: SPARCphormer (PyTorch transformer)
cd ../implementations/SPARCphormer
python train.py --dataset cora --space spectral --hops 5 \
    --split_name random --test_ratio 0.10 --sparc_seed 42

# 3b. Downstream: SPARC-SAGE (synthetic kNN GraphSAGE)
cd ../SPARC-SAGE
python -m graphsage.supervised_train \
    --cli_dataset cora --cli_seed 42 --cli_test_ratio 0.10 \
    --cli_split_name random --sparc_topk 10

# 3c. Downstream: SPARC-GCN (GCN with dual feature/spectral input)
cd ../SPARC-GCN
python train.py --dataset cora --use_sparc_only True --epochs 75
```

Run each script from its own directory so default relative paths resolve correctly.

---

## Step 1 — Data

`data/download_data.py` converts PyTorch Geometric and OGB datasets into the
**GraphSAGE-style** layout used throughout the repo:

```
data/
├── data/<dataset>/
│   ├── <dataset>-G.json
│   ├── <dataset>-feats.npy
│   ├── <dataset>-id_map.json
│   └── <dataset>-class_map.json
└── data_splits/<dataset>/
    ├── original/           # canonical split from the source dataset
    └── <split_name>/       # random cold-start splits, e.g. random_test0.10_val0.10_seed42
        ├── idx_train.npy
        ├── idx_val.npy
        └── idx_test.npy
```

Splits are created with `data/split_data.py`. The default cold-start split samples
val/test uniformly at random over nodes; `--split_name original` falls back to the
dataset's canonical split. See [`data/README.md`](data/README.md) for full details.

---

## Step 2 — Train the SPARC encoder

`SPARC/src/main.py` is the single training entry point. It loads the dataset, builds an
inductive train adjacency (train–train edges only), partitions it with METIS, and trains
`SpectralNet` with a Rayleigh-quotient style spectral loss — optionally combined with a
Wang–Davidson constrained-clustering term and a task-aware cross-entropy head over labeled
train nodes.

Common runs:

```bash
# Random cold-start split
python main.py --dataset cora --seed 42 --test_ratio 0.10 --val_ratio 0.10 --split_name random

# Canonical dataset split
python main.py --dataset cora --split_name original

# Spectral-only (no task-aware CE head)
python main.py --dataset cora --no_task_aware
```

Outputs land in:

```
SPARC/sparc_results/<dataset>/<split>_test<ratio>_seed<seed>[/suffix]/
├── embeddings.npy       # spectral embeddings Y (N × output_dim)
├── features.npy         # scaled input features
├── labels.npy           # integer class labels
├── train_mask.npy / val_mask.npy / test_mask.npy
├── train_adj.npz / full_adj.npz
├── metrics.json         # KMeans ACC / NMI / ARI on train, val, test, full
└── eigenspace_agreement.json   # optional Grassmann-distance diagnostic
```

Per-dataset hyperparameters live in `SPARC/src/config/<dataset>.json`. Override any field
without editing the base config:

```bash
python main.py --dataset cora --config_overrides path/to/overrides.json
```

See [`SPARC/src/README.md`](SPARC/src/README.md) for the full CLI, configuration
reference, and supported datasets.

---

## Step 3 — Downstream classifiers

All three classifiers consume a SPARC run directory at
`SPARC/sparc_results/<dataset>/<run>/`. The folder name is derived from the SPARC CLI
flags, e.g. `random_test0.10_seed42`.

### SPARCphormer (Transformer)

Transformer over multi-hop kNN sequences in a configurable token space (default:
SPARC spectral embeddings). PyTorch.

```bash
cd SPARC/implementations/SPARCphormer
python train.py --dataset cora --space spectral --hops 5 \
    --split_name random --test_ratio 0.10 --sparc_seed 42 \
    --hidden_dim 512 --n_layers 1 --n_heads 8 --batch_size 2000
```

Other token spaces: `features`, `computed`, `computed_symmetric_multihop_laplace`,
`real_graph`, `hops`, `cold-brew`, `fusion`. See
[`SPARCphormer/README.md`](SPARC/implementations/SPARCphormer/README.md) and
[`SPARCphormer/commands.txt`](SPARC/implementations/SPARCphormer/commands.txt) for
per-dataset hyperparameters.

### SPARC-SAGE (synthetic-kNN GraphSAGE)

GraphSAGE on a synthetic kNN graph built in SPARC embedding space — the original edges are
discarded entirely. TensorFlow 2.x (with TF 1.x compat).

```bash
cd SPARC/implementations/SPARC-SAGE
python -m graphsage.supervised_train \
    --cli_dataset cora --cli_seed 42 --cli_test_ratio 0.10 \
    --cli_split_name random --sparc_topk 10
```

Batch benchmark over (dataset, seed, split, topk) → CSV:

```bash
python run_all_experiments.py
# → random_split_results_graphsagesparc_all_datasets.csv
```

See [`SPARC-SAGE/README.md`](SPARC/implementations/SPARC-SAGE/README.md).

### SPARC-GCN (GCN with spectral channel)

GCN on the real graph using raw features and SPARC embeddings as a second input channel.
TensorFlow 1.x.

```bash
cd SPARC/implementations/SPARC-GCN
python train.py --dataset cora --use_sparc_only True --epochs 75
# Resolves SPARC/sparc_results/cora/random_test0.03_seed42 by default
# (or pass --sparc_run <run_folder> explicitly).
```

See [`SPARC-GCN/README.md`](SPARC/implementations/SPARC-GCN/README.md).

---

## Reproducibility

- All scripts accept a `--seed` flag and call `set_seed` before any randomness.
- METIS partitions are cached on disk under `data/partition_cache/`. Disable with
  `SPARC_NO_PARTITION_CACHE=1`.
- Tag a SPARC run with `--result_suffix <tag>` (or `SPARC_RESULT_SUFFIX`) to keep multiple
  configurations side by side in `sparc_results/`.
- Downstream classifiers resolve the SPARC run folder from
  `(--split_name, --test_ratio, --seed)` so reproducing a row in the paper means matching
  these flags between SPARC training and the downstream call.

---

## Adding a new dataset

1. Add a downloader branch in `data/download_data.py` (or place GraphSAGE files manually
   under `data/data/<name>/`).
2. Create `SPARC/src/config/<name>.json` (copy an existing config and tune `n_clusters`
   and the `spectral` block).
3. Run the standard pipeline:

   ```bash
   cd SPARC/src
   python main.py --dataset <name> --split_name random --test_ratio 0.10 --seed 42
   ```

---

## Citation

If you use this code, please cite:

```bibtex
@article{sparc2026,
  title   = {Leveraging Graph Geometry for Cold-Start Node Prediction},
  author  = {<authors>},
  journal = {<venue>},
  year    = {2026}
}
```

(Update the BibTeX entry once the paper is published.)

---

## Acknowledgments

SPARC builds on prior work in spectral graph learning and inductive representation
learning, including SpectralNet, GraphSAGE, Cluster-GCN (METIS partitioning), Cold-Brew,
and Graphormer-style transformers. Dataset loaders rely on
[PyTorch Geometric](https://pytorch-geometric.readthedocs.io/) and
[OGB](https://ogb.stanford.edu/).
