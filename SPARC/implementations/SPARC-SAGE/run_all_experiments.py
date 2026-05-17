"""
Run all GraphSAGESPARC cold-start experiments.

This is the SPARC-KNN variant: instead of the real graph, every node
aggregates from its top-K nearest train-set neighbors in SPARC embedding
space. See ``graphsage/supervised_train.py`` for details.

Combinations:
    datasets   : cora, citeseer, pubmed, chameleon, squirrel, reddit,
                 ogbn-products, ogbn-arxiv, wikics
    seeds      : 42
    splits     : random
    test_ratios: 0.10
    val_ratio  : 0.10 (kept for parity; actual masks come from SPARC run dir)

For each (dataset, seed, split, test_ratio) we resolve the SPARC run
directory at:

    SPARC_project/SPARC/sparc_results/<dataset>/<split>_test<tr>_seed<seed>/

and load embeddings.npy / features.npy / labels.npy / *_mask.npy.

Results are appended to a single CSV after every run so progress is never lost.

Usage:
    python run_all_experiments.py
"""

import csv
import os
import sys
import time
import traceback

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from graphsage.supervised_train import main as run_experiment  # noqa: E402

DATASETS = ["ogbn-products"]
SEEDS = [42]
SPLIT_NAMES = ["random"]
TEST_RATIOS = [0.10]
VAL_RATIO = 0.10
SPARC_TOPK = 10

RESULTS_FILE = os.path.join(
    SCRIPT_DIR, "random_split_results_graphsagesparc_all_datasets.csv")

CSV_COLUMNS = [
    "dataset", "seed", "split_name", "test_ratio", "val_ratio", "sparc_topk",
    "train_acc", "train_f1_mic", "train_f1_mac",
    "val_acc", "val_f1_mic", "val_f1_mac",
    "test_acc", "test_f1_mic", "test_f1_mac",
    "lr_test_acc", "lr_val_acc",
    "status", "error",
]


def load_completed_runs():
    """Return a set of (dataset, seed, split_name, test_ratio, sparc_topk) already done."""
    completed = set()
    if not os.path.exists(RESULTS_FILE):
        return completed
    with open(RESULTS_FILE, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("status") == "ok":
                key = (row["dataset"], int(row["seed"]),
                       row["split_name"], float(row["test_ratio"]),
                       int(row.get("sparc_topk", SPARC_TOPK)))
                completed.add(key)
    return completed


def append_row(row_dict):
    file_exists = os.path.exists(RESULTS_FILE)
    with open(RESULTS_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row_dict)


def run_all():
    completed = load_completed_runs()
    total = len(DATASETS) * len(SEEDS) * len(SPLIT_NAMES) * len(TEST_RATIOS)
    done = len(completed)
    print("=" * 70)
    print("Total experiments: {}  |  Already completed: {}  |  Remaining: {}".format(
        total, done, total - done))
    print("Results file: {}".format(RESULTS_FILE))
    print("SPARC top-K = {}".format(SPARC_TOPK))
    print("=" * 70)

    run_idx = 0
    for dataset in DATASETS:
        for split_name in SPLIT_NAMES:
            for test_ratio in TEST_RATIOS:
                for seed in SEEDS:
                    run_idx += 1
                    key = (dataset, seed, split_name, test_ratio, SPARC_TOPK)
                    if key in completed:
                        continue

                    print("\n" + "#" * 70)
                    print("RUN {}/{}: dataset={}, split={}, test_ratio={}, seed={}, topk={}".format(
                        run_idx, total, dataset, split_name, test_ratio, seed, SPARC_TOPK))
                    print("#" * 70)

                    row = {
                        "dataset": dataset,
                        "seed": seed,
                        "split_name": split_name,
                        "test_ratio": test_ratio,
                        "val_ratio": VAL_RATIO,
                        "sparc_topk": SPARC_TOPK,
                    }

                    t0 = time.time()
                    try:
                        results = run_experiment(
                            dataset=dataset,
                            seed=seed,
                            test_ratio=test_ratio,
                            val_ratio=VAL_RATIO,
                            split_name=split_name,
                            sparc_topk=SPARC_TOPK,
                        )
                        row.update(results)
                        row["status"] = "ok"
                        row["error"] = ""
                        elapsed = time.time() - t0
                        print("Finished in {:.1f}s — test_acc={:.5f}".format(
                            elapsed, results["test_acc"]))
                    except Exception as e:
                        row["status"] = "error"
                        row["error"] = str(e)
                        traceback.print_exc()
                        print("FAILED: {}".format(e))

                    append_row(row)

    print("\n" + "=" * 70)
    print("ALL EXPERIMENTS DONE.  Results saved to: {}".format(RESULTS_FILE))
    print("=" * 70)


if __name__ == "__main__":
    run_all()
