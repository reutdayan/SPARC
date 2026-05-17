"""Eigenspace-agreement diagnostic for SpectralNet.

Given a learned embedding ``Y`` (n x k) produced by SpectralNet and a graph
adjacency ``A``, this module measures how well the *column space* of ``Y``
approximates the bottom-k eigenspace of the (symmetric normalized) graph
Laplacian on the train graph. Two complementary scores are reported:

1. Chordal Grassmann distance between span(Y) and span(U*), where U* is the
   matrix of true bottom-k eigenvectors.
2. Projection-operator Frobenius error
   ``|| Y (Y^T Y)^{-1} Y^T  -  U* U*^T ||_F``
   (computed via orthonormal bases for numerical stability).

Also reports individual principal angles (deg) and a Rayleigh-Ritz estimate of
the *learned* eigenvalues so you can line them up against the true spectrum.

The module can be used two ways:

    # In-pipeline (see main.py):
    from eigenspace_diagnostics import run_diagnostic
    report = run_diagnostic(Y_full=Y, adj_train=train_adj,
                            train_mask=train_mask, k=output_dim)

    # Standalone CLI (re-run on saved artifacts):
    python eigenspace_diagnostics.py \
        --embeddings path/to/embeddings.npy \
        --adj        path/to/train_adj.npz \
        --train-mask path/to/train_mask.npy \
        --k 25 \
        --out report.json
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Optional

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import eigsh, lobpcg


# ---------------------------------------------------------------------------
# Laplacian construction
# ---------------------------------------------------------------------------

def _symmetrize(A: sp.spmatrix) -> sp.spmatrix:
    return ((A + A.T) * 0.5).tocsr()


def normalized_laplacian(adj: sp.spmatrix,
                         drop_self_loops: bool = True) -> sp.spmatrix:
    """Symmetric normalized Laplacian ``L_sym = I - D^{-1/2} A D^{-1/2}``.

    Self-loops are dropped by default so L is a proper graph Laplacian (trivial
    eigenvalue at 0 with eigenvector proportional to D^{1/2} * 1).
    """
    A = adj.tocsr().astype(np.float64).copy()
    if drop_self_loops:
        A.setdiag(0.0)
        A.eliminate_zeros()
    A = _symmetrize(A)
    deg = np.asarray(A.sum(axis=1)).flatten()
    d_inv_sqrt = np.zeros_like(deg)
    mask = deg > 0
    d_inv_sqrt[mask] = 1.0 / np.sqrt(deg[mask])
    D_inv_sqrt = sp.diags(d_inv_sqrt)
    n = A.shape[0]
    L = sp.eye(n, dtype=np.float64) - D_inv_sqrt @ A @ D_inv_sqrt
    return _symmetrize(L)


def multihop_affinity(adj: sp.spmatrix, order: int = 3,
                      drop_self_loops: bool = True) -> sp.spmatrix:
    """Sum-of-powers affinity W + W^2 + ... + W^order (matches SpectralTrainer).

    Self-loops are dropped before powering. The result is kept sparse when
    possible, but note that W^3 on dense-ish graphs fills in quickly.
    """
    if order < 1:
        raise ValueError("order must be >= 1")
    A = adj.tocsr().astype(np.float64).copy()
    if drop_self_loops:
        A.setdiag(0.0)
        A.eliminate_zeros()
    A = _symmetrize(A)
    W = A
    acc = W
    Wi = W
    for _ in range(1, order):
        Wi = (Wi @ W).tocsr()
        acc = (acc + Wi).tocsr()
    acc.setdiag(0.0)
    acc.eliminate_zeros()
    return _symmetrize(acc)


# ---------------------------------------------------------------------------
# Bottom-k eigenvector computation
# ---------------------------------------------------------------------------

def compute_true_eigenvectors(
    L: sp.spmatrix,
    k: int,
    method: str = "auto",
    drop_first: bool = True,
    tol: float = 1e-8,
    max_iter: int = 2000,
    seed: int = 0,
    sigma_shift: float = -1e-5,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the ``k`` smallest eigenpairs of a symmetric PSD operator ``L``.

    Args:
        L:           (n x n) symmetric sparse matrix (a graph Laplacian).
        k:           number of desired eigenvectors *after* optionally dropping
                     the trivial zero eigenvector.
        method:      "eigsh"  ARPACK shift-invert near ``sigma_shift`` (best
                              quality for sparse PSD operators),
                     "lobpcg" LOBPCG (deterministic with ``seed``; robust for
                              singular Laplacians),
                     "auto"   try eigsh, fall back to LOBPCG on failure.
        drop_first:  drop the smallest eigenvalue (~0) before returning.
        tol:         convergence tolerance.
        max_iter:    max iterations (both solvers).
        seed:        RNG seed used both to build a deterministic starting vector
                     for eigsh and to initialise LOBPCG.
        sigma_shift: shift for eigsh's shift-invert. Must NOT be exactly 0 for
                     Laplacians (L is singular at 0). A small negative value
                     targets eigenvalues near 0 from below and is numerically
                     stable.

    Returns:
        vals (k,), vecs (n, k) sorted in ascending eigenvalue order.
    """
    n = L.shape[0]
    k_req = int(k) + (1 if drop_first else 0)
    k_req = max(1, min(k_req, n - 1))

    rng = np.random.default_rng(seed)
    v0 = rng.standard_normal(n).astype(np.float64)
    v0 /= np.linalg.norm(v0) + 1e-30

    def _eigsh() -> tuple[np.ndarray, np.ndarray]:
        L64 = L.astype(np.float64)
        # Shift-invert near 0 (from below) is robust even when L is singular.
        try:
            vals, vecs = eigsh(L64, k=k_req, sigma=sigma_shift,
                               which="LM", tol=tol, maxiter=max_iter, v0=v0)
        except Exception:
            # Smallest-algebraic is slower but doesn't need a factorisation.
            vals, vecs = eigsh(L64, k=k_req, which="SA",
                               tol=tol, maxiter=max_iter, v0=v0)
        return vals, vecs

    def _lobpcg() -> tuple[np.ndarray, np.ndarray]:
        rng_l = np.random.default_rng(seed)
        X0 = rng_l.standard_normal((n, k_req)).astype(np.float64)
        X0, _ = np.linalg.qr(X0)
        vals, vecs = lobpcg(L.astype(np.float64), X0, tol=tol,
                            largest=False, maxiter=max_iter)
        return vals, vecs

    if method == "eigsh":
        vals, vecs = _eigsh()
    elif method == "lobpcg":
        vals, vecs = _lobpcg()
    elif method == "auto":
        try:
            vals, vecs = _eigsh()
        except Exception as exc:
            print(f"[eigenspace_diagnostics] eigsh failed ({exc}); "
                  f"falling back to LOBPCG.")
            vals, vecs = _lobpcg()
    else:
        raise ValueError(f"Unknown method: {method}")

    # Numerical cleanup: clamp tiny negatives (PSD)
    vals = np.maximum(vals, 0.0)
    order = np.argsort(vals)
    vals = vals[order]
    vecs = vecs[:, order]
    if drop_first:
        vals = vals[1:]
        vecs = vecs[:, 1:]
    return vals[:k], vecs[:, :k]


# ---------------------------------------------------------------------------
# Subspace comparison primitives
# ---------------------------------------------------------------------------

def _orth(A: np.ndarray, *, check: bool = False,
         atol: float = 1e-6) -> np.ndarray:
    """Orthonormal basis for ``col(A)`` via reduced QR.

    Returns a column-orthonormal matrix ``Q`` of shape
    ``(n, k)`` (with ``k = min(n, A.shape[1])``) such that:

    * ``Q.T @ Q == I_k`` (columns are unit-norm and mutually orthogonal,
      up to floating-point precision -- this is the *unit-vector*
      property the Grassmann distance below relies on);
    * ``span(Q) == span(A)`` (so principal angles between ``col(A)`` and
      ``col(B)`` are the same as between ``col(Q_a)`` and ``col(Q_b)``).

    Because QR enforces (i) automatically, callers do **not** need to
    pre-normalise the columns of ``A``. Whether ``A`` has columns at L2
    norm 1 (true Laplacian eigenvectors), L2 norm ``sqrt(m)`` (the
    SpectralNet output ``Y`` with ``Y^T Y = m I``), or arbitrary norms,
    the resulting ``Q`` always has columns at L2 norm 1.

    Set ``check=True`` to assert ``||Q.T Q - I||_F < atol`` post-hoc;
    this is cheap (k×k Frobenius norm) and gives a hard guarantee that
    every vector going into the principal-angle SVD really is unit-
    length.
    """
    if A.ndim != 2:
        raise ValueError("expected 2D matrix")
    Q, _ = np.linalg.qr(A)
    if check:
        k = Q.shape[1]
        gram = Q.T @ Q
        err = float(np.linalg.norm(gram - np.eye(k), ord="fro"))
        if err > atol:
            raise AssertionError(
                f"_orth post-check: Q is not orthonormal "
                f"(||Q^T Q - I||_F = {err:.3e} > {atol:.0e}). "
                f"Q.shape={Q.shape}, A.shape={A.shape}.")
    return Q


def principal_angles(A: np.ndarray, B: np.ndarray, *,
                     check_unit_norm: bool = False) -> np.ndarray:
    """Principal angles (rad) between ``col(A)`` and ``col(B)``.

    The principal-angle definition is: given any orthonormal bases
    ``Q_a``, ``Q_b`` of the two subspaces, the singular values of
    ``Q_a^T Q_b`` are the cosines of the principal angles. This relies
    on **both bases having unit-norm orthogonal columns**; we get that
    here from QR of ``A`` and ``B`` (``_orth``), so the "all vectors are
    unit vectors" requirement is enforced automatically and is
    independent of the column scales of the inputs.

    Set ``check_unit_norm=True`` to make the unit-norm property a
    hard runtime assertion (raises ``AssertionError`` otherwise). The
    computation itself is unchanged either way.

    Returned in ascending order (smallest angle first). Length is
    ``min(A.shape[1], B.shape[1])``; values lie in ``[0, pi/2]``.
    """
    Qa = _orth(A, check=check_unit_norm)
    Qb = _orth(B, check=check_unit_norm)
    # cos(theta_i) = singular values of Q_a^T Q_b (in [0, 1] up to fp
    # noise; we clip just to keep arccos in-domain).
    _, s, _ = np.linalg.svd(Qa.T @ Qb, full_matrices=False)
    s = np.clip(s, -1.0, 1.0)
    angles = np.arccos(s)
    return np.sort(angles)


def chordal_grassmann_distance(A: np.ndarray, B: np.ndarray, *,
                               check_unit_norm: bool = False) -> float:
    """Chordal Grassmann distance ``d_c = sqrt(sum_i sin^2 theta_i)``.

    Operates on the *column spans* of ``A`` and ``B``: orthonormal
    bases are formed internally via QR, so the result is invariant to
    any rescaling of the input columns (e.g. ``A`` having columns at
    norm ``sqrt(m)`` vs. norm 1). Lies in ``[0, sqrt(k)]``, where
    ``k = min(A.shape[1], B.shape[1])``; the ratio ``d_c / sqrt(k)``
    is therefore in ``[0, 1]`` and is what to report when the
    ambient ``k`` differs across runs.

    This metric is, up to a factor of ``sqrt(2)``, the Frobenius
    projection error ``|| P_A - P_B ||_F``; see ``projection_error``.
    """
    angles = principal_angles(A, B, check_unit_norm=check_unit_norm)
    return float(np.sqrt(np.sum(np.sin(angles) ** 2)))


def projection_error(A: np.ndarray, B: np.ndarray) -> float:
    """Frobenius norm of the projector difference ``|| P_A - P_B ||_F``.

    Uses orthonormal bases: if Qa, Qb are orthonormal bases of col(A), col(B),
        || P_A - P_B ||_F^2 = k_a + k_b - 2 || Qa^T Qb ||_F^2.
    Equal to ``sqrt(2) * chordal_grassmann_distance(A, B)`` when col(A) and
    col(B) have equal dimension; reported separately here so the numbers you
    see match the Frobenius-form in the paper text.
    """
    Qa = _orth(A)
    Qb = _orth(B)
    ka = Qa.shape[1]
    kb = Qb.shape[1]
    cross = Qa.T @ Qb
    val = ka + kb - 2.0 * float(np.sum(cross ** 2))
    return float(np.sqrt(max(val, 0.0)))


def rayleigh_ritz_eigenvalues(Y: np.ndarray, L: sp.spmatrix) -> np.ndarray:
    """Rayleigh-Ritz estimate of the eigenvalues represented by span(Y).

    Performs a Rayleigh-Ritz projection: orthonormalize Y to Q, form the small
    matrix M = Q^T L Q (k x k), and return its eigenvalues in ascending order.
    These upper-bound the true bottom-k eigenvalues of L (Cauchy interlacing).
    """
    Q = _orth(Y)
    LQ = L @ Q
    M = Q.T @ LQ
    M = (M + M.T) * 0.5
    vals = np.linalg.eigvalsh(M)
    return np.sort(vals)


# ---------------------------------------------------------------------------
# Connectivity helpers
# ---------------------------------------------------------------------------

def _component_info(adj: sp.spmatrix) -> tuple[int, np.ndarray, np.ndarray]:
    """Return ``(num_components, labels, sizes)`` for a sparse adjacency.

    Self-loops are ignored (they don't affect connectivity).
    """
    A = adj.tocsr().copy()
    A.setdiag(0)
    A.eliminate_zeros()
    nc, labs = connected_components(A, directed=False)
    sizes = np.bincount(labs, minlength=nc)
    return int(nc), labs, sizes


def _largest_component_idx(adj: sp.spmatrix) -> tuple[np.ndarray, int, np.ndarray]:
    """Indices of nodes in the largest CC of ``adj`` plus ``(num_components, sizes)``.

    Breaks ties by component label (``argmax`` picks the first max).
    """
    nc, labs, sizes = _component_info(adj)
    lcc_label = int(np.argmax(sizes))
    idx = np.where(labs == lcc_label)[0]
    return idx, nc, sizes


# ---------------------------------------------------------------------------
# End-to-end
# ---------------------------------------------------------------------------

def eigenspace_agreement(
    Y: np.ndarray,
    adj: sp.spmatrix,
    k: int,
    laplacian_type: str = "sym_adj",
    affinity_order: int = 3,
    drop_first: bool = True,
    method: str = "auto",
    tol: float = 1e-6,
    compute_learned_spectrum: bool = True,
    verbose: bool = True,
    restrict_to_lcc: bool = False,
) -> dict:
    """Compute the eigenspace-agreement report.

    Args:
        Y:                (n x k_y) learned embedding restricted to the node
                          set represented by ``adj``. Must be aligned row-wise
                          with ``adj``.
        adj:              (n x n) sparse adjacency for the reference graph.
        k:                number of spectral components to compare against.
                          If ``Y`` has fewer columns, the smaller count is
                          used.
        laplacian_type:   "sym_adj"      Laplacian of ``adj`` (standard).
                          "sym_multihop" Laplacian of
                              sum_{p=1..affinity_order} W^p
                          which matches the training target of SpectralTrainer
                          when ``affinity_walk_order`` is set to the same value.
        affinity_order:   p for the multihop variant.
        drop_first:       drop the trivial 0-eigenvalue eigenvector.
        method:           eigsh | lobpcg | auto.
        tol:              eigensolver tolerance.
        compute_learned_spectrum:
                          additionally return Rayleigh-Ritz learned eigenvalues.
        verbose:          print a short human-readable summary.

    Returns:
        dict with the report; JSON-serializable.
    """
    if sp.issparse(Y):
        raise TypeError(
            "eigenspace_agreement(Y, adj, ...): first argument must be the "
            "(n, k) embedding Y as a dense numpy array, not a sparse matrix "
            "(did you pass adjacency as Y by mistake?)."
        )
    Y = np.asarray(Y, dtype=np.float64, order="C")
    if Y.ndim != 2:
        raise ValueError(
            "Y must be 2D with shape (n, k); got shape {}. "
            "Arguments are (Y, adj), not (adj, Y).".format(getattr(Y, "shape", None))
        )

    n_y, k_y = Y.shape
    n_a = adj.shape[0]
    if n_y != n_a:
        raise ValueError(
            f"Row-count mismatch: Y has {n_y} rows but adj is {n_a}x{n_a}.")

    kk = int(min(k, k_y))
    if kk < 1:
        raise ValueError("k must be >= 1")

    # Report the connectivity structure of the graph the diagnostic is about
    # to analyse: the number of zero eigenvalues of L_sym equals the number
    # of connected components, so this number bounds how many "true" bottom-k
    # eigenvectors are trivial indicator functions.
    nc_full, _, sizes_full = _component_info(adj)
    lcc_size_full = int(sizes_full.max()) if nc_full > 0 else int(n_a)
    if verbose:
        n_singletons = int((sizes_full == 1).sum())
        print(f"[eigenspace_diagnostics] graph: n={n_a}, "
              f"connected components = {nc_full}, "
              f"largest = {lcc_size_full}, isolated nodes = {n_singletons}")

    # Optionally restrict the evaluation to the largest connected component.
    # The network was trained on the full (fragmented) graph; we merely
    # measure spectral agreement on the part of the graph where the bottom
    # spectrum is not dominated by component-indicator functions.
    lcc_info: dict | None = None
    if restrict_to_lcc and nc_full > 1:
        idx, _, sizes = _largest_component_idx(adj)
        adj = adj.tocsr()[idx][:, idx]
        Y = np.asarray(Y)[idx]
        n_a = adj.shape[0]
        n_y = Y.shape[0]
        lcc_info = {
            "original_n": int(n_y + (nc_full - 1)),  # informational only
            "original_components": int(nc_full),
            "lcc_n": int(n_a),
        }
        if verbose:
            print(f"[eigenspace_diagnostics] restrict_to_lcc=True: "
                  f"slicing to the largest component "
                  f"({n_a} of {int(sizes.sum())} nodes).")

    # Build the reference operator
    if laplacian_type == "sym_adj":
        L = normalized_laplacian(adj)
    elif laplacian_type == "sym_multihop":
        W = multihop_affinity(adj, order=affinity_order)
        L = normalized_laplacian(W)
    else:
        raise ValueError(f"Unknown laplacian_type: {laplacian_type}")

    t0 = time.time()
    true_vals, U = compute_true_eigenvectors(
        L, k=kk, method=method, drop_first=drop_first, tol=tol)
    eig_secs = time.time() - t0

    Y_use = Y[:, :kk]
    angles = principal_angles(Y_use, U)
    g = chordal_grassmann_distance(Y_use, U)
    p_err = projection_error(Y_use, U)

    report: dict = {
        "k": kk,
        "n": int(n_a),
        "laplacian_type": laplacian_type,
        "drop_first": bool(drop_first),
        "num_connected_components": int(nc_full),
        "largest_component_size": int(lcc_size_full),
        "restricted_to_lcc": bool(restrict_to_lcc and nc_full > 1),
        "grassmann_chordal": g,
        "projection_error_frobenius": p_err,
        "principal_angles_deg": np.degrees(angles).tolist(),
        "mean_principal_angle_deg": float(np.degrees(angles).mean()),
        "max_principal_angle_deg": float(np.degrees(angles).max()),
        "true_eigenvalues": true_vals.tolist(),
        "eig_solver_seconds": eig_secs,
    }
    if lcc_info is not None:
        report["lcc_info"] = lcc_info

    if compute_learned_spectrum:
        learned_vals = rayleigh_ritz_eigenvalues(Y_use, L)
        report["learned_eigenvalues_rayleigh_ritz"] = learned_vals.tolist()
        m = min(len(learned_vals), len(true_vals))
        report["eigenvalue_abs_gap"] = np.abs(
            learned_vals[:m] - true_vals[:m]).tolist()

    if verbose:
        tag = "LCC-restricted" if report["restricted_to_lcc"] else "full"
        print("\n=== Eigenspace agreement ({} L, {}) ===".format(
            laplacian_type, tag))
        print(f"  n = {n_a}, k = {kk}, drop_first = {drop_first}, "
              f"components(full) = {nc_full}")
        print(f"  Grassmann (chordal)           : {g:.6f}   "
              f"(max = sqrt(k) = {np.sqrt(kk):.3f})")
        print(f"  Projection error (Frobenius)  : {p_err:.6f}   "
              f"(max = sqrt(2k) = {np.sqrt(2 * kk):.3f})")
        print(f"  Mean principal angle          : "
              f"{report['mean_principal_angle_deg']:.2f} deg")
        print(f"  Max  principal angle          : "
              f"{report['max_principal_angle_deg']:.2f} deg")
        print(f"  Eigensolver time              : {eig_secs:.2f} s")
        if compute_learned_spectrum:
            tv = np.array(report["true_eigenvalues"])
            lv = np.array(report["learned_eigenvalues_rayleigh_ritz"])
            m = min(len(tv), len(lv))
            print("  True vs learned (Rayleigh-Ritz) eigenvalues (first "
                  f"{min(m, 10)}):")
            for i in range(min(m, 10)):
                print(f"    {i:2d}: true={tv[i]:.6f}   "
                      f"learned={lv[i]:.6f}   |diff|={abs(tv[i]-lv[i]):.6f}")
        print()

    return report


def run_diagnostic(
    Y_full: np.ndarray,
    adj_train: sp.spmatrix,
    train_mask: np.ndarray,
    k: int,
    **kwargs,
) -> dict:
    """Convenience wrapper: slice Y/adj to train nodes, then run.

    Args:
        Y_full:     (N, k_y) embedding over the *full* node set.
        adj_train:  (N, N) inductive train adjacency (train-train edges only).
                    Non-train rows/cols are dropped before computing L.
        train_mask: (N,) boolean mask selecting train nodes.
        k:          number of spectral components to compare.
        **kwargs:   forwarded to ``eigenspace_agreement``.
    """
    train_mask = np.asarray(train_mask).astype(bool)
    idx = np.where(train_mask)[0]
    Y_train = Y_full[idx]
    A = adj_train.tocsr()[idx][:, idx]
    return eigenspace_agreement(Y_train, A, k=k, **kwargs)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _load_adj(path: str) -> sp.spmatrix:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npz":
        return sp.load_npz(path)
    if ext == ".npy":
        A = np.load(path)
        return sp.csr_matrix(A)
    raise ValueError(f"Unsupported adjacency format: {ext}")


def _main_cli() -> None:
    p = argparse.ArgumentParser(
        description="Eigenspace-agreement diagnostic for SpectralNet.")
    p.add_argument("--embeddings", required=True,
                   help="Path to embeddings .npy (shape N x k).")
    p.add_argument("--adj", required=True,
                   help="Path to train adjacency (.npz sparse or .npy dense).")
    p.add_argument("--train-mask", default=None,
                   help="Optional path to train_mask.npy (bool, length N). "
                        "If omitted, all rows are assumed to be train nodes.")
    p.add_argument("--k", type=int, default=None,
                   help="Number of spectral components to compare "
                        "(defaults to embeddings.shape[1]).")
    p.add_argument("--laplacian-type", default="sym_adj",
                   choices=["sym_adj", "sym_multihop"])
    p.add_argument("--affinity-order", type=int, default=3)
    p.add_argument("--method", default="auto",
                   choices=["auto", "eigsh", "lobpcg"])
    p.add_argument("--tol", type=float, default=1e-6)
    p.add_argument("--keep-first", action="store_true",
                   help="Do NOT drop the trivial 0 eigenvalue.")
    p.add_argument("--restrict-to-lcc", action="store_true",
                   help="Restrict the evaluation to the largest connected "
                        "component of the (train) graph. Useful on graphs with "
                        "many small/isolated components, where the bottom-k "
                        "eigenspace is dominated by component indicators.")
    p.add_argument("--out", default=None,
                   help="Optional path to write the JSON report.")
    args = p.parse_args()

    Y = np.load(args.embeddings)
    adj = _load_adj(args.adj)
    k = args.k if args.k is not None else Y.shape[1]

    if args.train_mask is not None:
        tm = np.load(args.train_mask).astype(bool)
        report = run_diagnostic(
            Y_full=Y, adj_train=adj, train_mask=tm, k=k,
            laplacian_type=args.laplacian_type,
            affinity_order=args.affinity_order,
            drop_first=not args.keep_first,
            method=args.method,
            tol=args.tol,
            restrict_to_lcc=args.restrict_to_lcc,
        )
    else:
        report = eigenspace_agreement(
            Y=Y, adj=adj, k=k,
            laplacian_type=args.laplacian_type,
            affinity_order=args.affinity_order,
            drop_first=not args.keep_first,
            method=args.method,
            tol=args.tol,
            restrict_to_lcc=args.restrict_to_lcc,
        )

    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Wrote report to {args.out}")


if __name__ == "__main__":
    _main_cli()
