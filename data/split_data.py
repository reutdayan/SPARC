import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent
DEFAULT_SPLIT_DIR = DATA_DIR / "data_splits"

VALID_STRATEGIES = frozenset({"random", "original"})


def load_saved_split(dataset_name, split_name="original", split_dir=None):
    if split_dir is None:
        split_dir = DEFAULT_SPLIT_DIR
    # Preferred layout:
    #   <split_dir>/<dataset_name>/<split_name>/idx_*.npy
    # Backward-compatible legacy layout:
    #   <split_dir>/<dataset_name>/idx_*.npy
    split_path = Path(split_dir) / dataset_name / split_name
    legacy_path = Path(split_dir) / dataset_name

    train_path = split_path / "idx_train.npy"
    val_path = split_path / "idx_val.npy"
    test_path = split_path / "idx_test.npy"

    if not (train_path.exists() and val_path.exists() and test_path.exists()):
        train_path = legacy_path / "idx_train.npy"
        val_path = legacy_path / "idx_val.npy"
        test_path = legacy_path / "idx_test.npy"

    missing = [str(p) for p in [train_path, val_path, test_path] if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Saved split files not found in either "
            f"'{split_path}' or '{legacy_path}'. Missing: {', '.join(missing)}. "
            "Generate them first with download_data.py"
        )

    idx_train_saved = np.load(train_path).astype(np.int64)
    idx_val = np.load(val_path).astype(np.int64)
    idx_test = np.load(test_path).astype(np.int64)
    # Rebuild train as all known nodes that are not in val/test.
    all_nodes = np.union1d(np.union1d(idx_train_saved, idx_val), idx_test)
    idx_train = np.setdiff1d(all_nodes, np.union1d(idx_val, idx_test))
    return idx_train, idx_val, idx_test


def _validate_strategy(strategy, param_name):
    if strategy not in VALID_STRATEGIES:
        raise ValueError(
            f"{param_name} must be one of {sorted(VALID_STRATEGIES)}, got {strategy!r}"
        )


def choose_nodes_by_strategy(
    adj,
    frac,
    strategy="random",
    seed=123,
    eligible_nodes=None,
    original_nodes=None,
):
    _validate_strategy(strategy, "strategy")
    rng = np.random.RandomState(seed)
    n = adj.shape[0]
    k = int(round(frac * n))

    if eligible_nodes is None:
        eligible_nodes = np.arange(n)
    else:
        eligible_nodes = np.array(eligible_nodes)

    if strategy == "random":
        chosen = rng.choice(eligible_nodes, size=k, replace=False)
    elif strategy == "original":
        if original_nodes is None:
            raise ValueError(
                "original_nodes must be provided when strategy='original'")
        chosen = np.intersect1d(
            eligible_nodes,
            np.asarray(original_nodes, dtype=np.int64),
            assume_unique=False,
        )

    return np.array(chosen, dtype=np.int64)


def make_cold_start_split(
    adj,
    test_frac=0.10,
    val_frac=0.10,
    test_strategy="random",
    val_strategy="random",
    seed=123,
    dataset_name=None,
    split_dir=None,
):
    """
    Creates split indices only.
    Cold-start behavior itself will happen later by masking edges in the loader.

    test_strategy / val_strategy: "random" or "original".
    When test_strategy is "original", returns the canonical split from
    data_splits/<dataset>/original/ (requires dataset_name).
    """
    _validate_strategy(test_strategy, "test_strategy")
    _validate_strategy(val_strategy, "val_strategy")

    n = adj.shape[0]
    all_nodes = np.arange(n)

    if test_strategy == "original":
        if dataset_name is None:
            raise ValueError(
                "dataset_name is required when using strategy='original'")
        _, original_val, original_test = load_saved_split(
            dataset_name=dataset_name,
            split_name="original",
            split_dir=split_dir,
        )
        all_nodes = np.arange(n, dtype=np.int64)
        original_train = np.setdiff1d(
            all_nodes, np.union1d(original_val, original_test)
        )
        return original_train, original_val, original_test

    if val_strategy != "random":
        raise ValueError(
            "val_strategy must be 'random' when test_strategy is 'random'"
        )

    idx_test = choose_nodes_by_strategy(
        adj,
        frac=test_frac,
        strategy=test_strategy,
        seed=seed,
        eligible_nodes=all_nodes,
    )

    remaining_after_test = np.setdiff1d(all_nodes, idx_test)

    idx_val = choose_nodes_by_strategy(
        adj,
        frac=val_frac,
        strategy=val_strategy,
        seed=seed + 1,
        eligible_nodes=remaining_after_test,
    )

    idx_train = np.setdiff1d(remaining_after_test, idx_val)

    assert len(np.intersect1d(idx_train, idx_val)) == 0
    assert len(np.intersect1d(idx_train, idx_test)) == 0
    assert len(np.intersect1d(idx_val, idx_test)) == 0
    assert len(np.union1d(np.union1d(idx_train, idx_val), idx_test)) == n

    return idx_train, idx_val, idx_test


def save_split(dataset_name, split_name, idx_train, idx_val, idx_test, out_dir=None):
    if out_dir is None:
        out_dir = DEFAULT_SPLIT_DIR
    split_dir = Path(out_dir) / dataset_name / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    np.save(split_dir / "idx_train.npy", idx_train)
    np.save(split_dir / "idx_val.npy", idx_val)
    np.save(split_dir / "idx_test.npy", idx_test)

    print(f"Saved split {split_name} for {dataset_name} to {split_dir}")
