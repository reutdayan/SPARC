# SPARCphormer

**Transformer classifier** on multi-hop node sequences built from a chosen **token space** (SPARC embeddings, raw features, Laplacian eigenvectors, Cold-Brew vectors, or fused signals). Designed for the same **inductive cold-start** masks as [`../../src/main.py`](../../src/main.py).

## Prerequisites

1. **SPARC run** (for most `--space` modes):

   ```bash
   cd ../../src
   python main.py --dataset cora --seed 42 --test_ratio 0.10 --split_name random
   ```

   Expects `../../sparc_results/<dataset>/<split>_test<ratio>_seed<seed>/` with at least `embeddings.npy`, `features.npy`, `labels.npy`, and `*_mask.npy`. Adjacency-based spaces also need `full_adj.npz`.

2. **PyTorch**, NumPy, SciPy, scikit-learn. Optional: **Cold-Brew** `.npz` embeddings for `--space cold-brew` or `fusion`.

## Quick start

From this directory (`implementations/SPARCphormer`):

```bash
# Classify using SPARC spectral embeddings as the retrieval space (default)
python train.py \
  --dataset cora \
  --space spectral \
  --hops 5 \
  --split_name random \
  --test_ratio 0.10 \
  --sparc_seed 42 \
  --seed 3407 \
  --batch_size 2000 \
  --hidden_dim 512 \
  --n_layers 1 \
  --n_heads 8 \
  --peak_lr 0.01 \
  --weight_decay 1e-5
```

Match the SPARC run directory tags if you used a suffix:

```bash
python train.py --dataset cora --space spectral --sparc_result_suffix _mytag ...
```

See [`commands.txt`](commands.txt) for per-dataset hyperparameter examples (cora, citeseer, pubmed, reddit, …).

## Token spaces (`--space`)

| Value | Description |
|-------|-------------|
| `spectral` | kNN sequences in **SPARC** `embeddings.npy` (default) |
| `features` | kNN in raw GraphSAGE features + cold-start split (no `sparc_results` required) |
| `computed` | Laplacian eigenvectors of `full_adj.npz`, then kNN sequences |
| `computed_symmetric_multihop_laplace` | Eigenvectors of symmetric multihop Laplacian (set `--multihop_walk_order` to match SPARC config) |
| `real_graph` | Message-passing hops on the **real** graph adjacency |
| `hops` | Same adjacency, hop-style feature construction |
| `cold-brew` | Cold-Brew `.npz` embeddings + GraphSAGE split |
| `fusion` | Concatenate SPARC spectral + Cold-Brew (+ optional raw features) |

## Important flags

| Flag | Description |
|------|-------------|
| `--dataset` | Dataset name |
| `--space` | Token / retrieval space (see table above) |
| `--hops` | Number of hop levels in the sequence model |
| `--split_name`, `--test_ratio`, `--val_ratio`, `--sparc_seed` | Must match the SPARC `main.py` run that produced embeddings |
| `--sparc_result_suffix` | Run folder suffix (same as SPARC `--result_suffix`) |
| `--pe_dim` | Positional encoding size for Laplacian-based spaces |
| `--feature_concat` | `spectral`, `X_embedded`, or `none` (fusion / ablations) |
| `--cold_brew_*` | Paths and hyperparameters for Cold-Brew `.npz` layout |

## Pipeline sketch

```
sparc_results/  →  load embeddings, features, masks [, full_adj]
        →  build multi-hop token sequences (utils.re_features_*)
        →  TransformerModel  →  node classification (CE)
        →  train / val / test accuracy (+ ROC-AUC for multilabel)
```

## Layout

| File | Role |
|------|------|
| `train.py` | CLI, data loading, feature construction, training |
| `model.py` | Transformer classifier |
| `utils.py` | Hop sequences, kNN cold-start helpers, fusion loaders |
| `data.py` | Legacy DGL dataset helpers (some code paths) |
| `lr.py`, `early_stop.py` | LR schedule and early stopping |

## Notes

- **`--sparc_seed`** defaults to `--seed` when omitted; use `--sparc_seed` to match SPARC’s data split seed independently of the Transformer seed.
- Large graphs: use `--space spectral` without loading `full_adj` when possible; `computed` / `hops` need `full_adj.npz` and more memory.
- If the preferred `sparc_results` folder is missing, `train.py` may fall back to the first subdirectory found — pass `--sparc_result_suffix` or align folder names explicitly.
