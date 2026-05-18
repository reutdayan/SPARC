# SPARC: Leveraging Graph Geometry for Cold-Start Node Prediction

Official implementation for:

> **Leveraging Graph Geometry for Cold-Start Node Prediction**

SPARC is a framework for **strict cold-start node prediction**, where test nodes arrive with features but without observed edges. The method learns an inductive feature-to-spectral mapping that places unseen nodes in a graph-geometry-aware space, enabling pseudo-neighborhood retrieval without test-time adjacency.

The retrieved pseudo-neighborhoods can be used as a plug-in structural context for downstream graph models, including GraphSAGE, Spectral-GCN, and NAGphormer-style graph transformers.

---

## Overview

Most graph neural networks require neighborhood information at inference time. This assumption breaks in the strict cold-start setting, where newly arriving nodes have no observed incident edges.

SPARC addresses this by learning a parametric encoder that maps node features to a low-dimensional spectral representation approximating the graph Laplacian eigenspace. At inference time, a cold-start node is embedded from its features alone, and its pseudo-neighborhood is retrieved using k-nearest neighbors in the learned SPARC space.

---

## Key features

- **Strict cold-start setting**  
  Test nodes are unseen during training and have no observed edges at inference time.

- **Inductive spectral representation**  
  SPARC learns a feature-to-spectral encoder, avoiding full-graph eigendecomposition at inference.

- **Pseudo-neighborhood retrieval**  
  Cold-start nodes retrieve structurally meaningful neighbors in SPARC space.

- **Backbone-agnostic design**  
  SPARC modifies the neighborhood construction step, not the downstream architecture.

- **Supported downstream models**  
  - `SPARC-GCN`
  - `SPARC-SAGE`
  - `SPARCphormer`

- **Benchmarked datasets**  
  `Cora`, `Citeseer`, `Pubmed`, `Chameleon`, `Squirrel`, `Wiki-CS`, `Reddit`, and `ogbn-products`.
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
