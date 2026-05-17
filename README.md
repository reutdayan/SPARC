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

- **Inductive spectral embeddings.** An MLP trained with a
  Rayleigh-quotient style loss on multihop affinities. At test time
  the encoder produces embeddings for cold-start nodes from features alone — no retraining,
  no eigendecomposition of the full graph.
- **Cold-start by construction.** The training adjacency keeps train–train edges only;
  val/test nodes are never observed during message passing.
- **Three downstream classifiers.**
  - `SPARC-GCN` — GCN on the real graph with raw features + SPARC embeddings as a second
    input channel.
  - `SPARC-SAGE` — GraphSAGE on a *synthetic* kNN graph built in SPARC embedding space.
  - `SPARCphormer` — Transformer over multi-hop token sequences in the sparc space.
- **Benchmarked datasets.** `cora`, `citeseer`, `pubmed`, `chameleon`, `squirrel`,
  `wikics`, `reddit`, `ogbn-arxiv`, `ogbn-products`.

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
```

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
