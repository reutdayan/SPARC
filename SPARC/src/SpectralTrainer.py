import random
import torch
import numpy as np
import torch.nn as nn
from sklearn.cluster import KMeans
from torch import sigmoid
from torch.nn import functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler
import os

import utils
from utils import *
from torch.utils.data import DataLoader, random_split, TensorDataset


class SpectralNetModel(nn.Module):
    def __init__(self, architecture: dict, input_dim: int, device: torch.device):
        super(SpectralNetModel, self).__init__()
        self.architecture = architecture
        self.num_of_layers = self.architecture["n_layers"]
        self.layers = nn.ModuleList()
        self.input_dim = input_dim
        self.device = device
        self.orthonorm_weights = torch.eye(self.input_dim).to(self.device)
        self.rotation_matrix = None
        

        current_dim = self.input_dim
        for layer, dim in self.architecture.items():
            next_dim = dim
            if layer == "n_layers":
                continue
            if layer == "output_dim":
                layer = nn.Sequential(nn.Linear(current_dim, next_dim), nn.Tanh())
                self.layers.append(layer)
            else:
                layer = nn.Sequential(nn.Linear(current_dim, next_dim), nn.ReLU())
                self.layers.append(layer)
                current_dim = next_dim


    def _make_orthonorm_weights(self, Y: torch.Tensor) -> torch.Tensor:
        """
        Orthonormalize the output of the network using the Cholesky decomposition.

        Parameters
        ----------
        Y : torch.Tensor
            The output of the network.

        Returns
        -------
        torch.Tensor
            The orthonormalized output.

        Notes
        -----
        This function applies the Cholesky decomposition to orthonormalize the output (`Y`) of the network.
        The orthonormalized output is returned as a tensor.
        """

        m = Y.shape[0]

        _, R = torch.linalg.qr(Y)
        D = torch.diag(torch.sign(torch.diag(R)))
        R = R @ D

        R = R + 1e-6 * torch.eye(R.shape[0], device=R.device)
        orthonorm_weights = np.sqrt(m) * torch.linalg.solve_triangular(
            R, torch.eye(R.shape[0], device=R.device), upper=True)

        return orthonorm_weights


    def forward(self, x: torch.Tensor, should_update_orth_weights: bool = True) -> torch.Tensor:
        """
        Perform the forward pass of the model.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.
        should_update_orth_weights : bool, optional
            Whether to update the orthonormalization weights using the Cholesky decomposition or not.

        Returns
        -------
        torch.Tensor
            The output tensor.

        Notes
        -----
        This function takes an input tensor `x` and computes the forward pass of the model.
        If `should_update_orth_weights` is set to True, the orthonormalization weights are updated
        using the Cholesky decomposition. The output tensor is returned.
        """

        for layer in self.layers:
            x = layer(x)
            
        Y_tilde = x
        if should_update_orth_weights:
            self.orthonorm_weights = self._make_orthonorm_weights(Y_tilde)

        Y = Y_tilde @ self.orthonorm_weights
        return Y


    def set_rotation_matrix(self, rotation_matrix):
        self.rotation_matrix = rotation_matrix


class SpectralNetLoss(nn.Module):
    def __init__(self):
        super(SpectralNetLoss, self).__init__()


    def forward(self, W: torch.Tensor, Y: torch.Tensor, is_normalized: bool = False) -> torch.Tensor:
        """
        This function computes the loss of the SpectralNet model.
        The loss is the rayleigh quotient of the Laplacian matrix obtained from W, 
        and the orthonormalized output of the network.

        Args:
            W (torch.Tensor):               Affinity matrix
            Y (torch.Tensor):               Output of the network
            is_normalized (bool, optional): Whether to use the normalized Laplacian matrix or not.

        Returns:
            torch.Tensor: The loss
        """
        m = Y.size(0)
        
        if is_normalized:
            d = torch.sum(W, dim=1).clamp(min=1e-8)
            Y = Y / torch.sqrt(d[:, None])

        Dy = torch.cdist(Y, Y)
        loss = torch.sum(W * Dy.pow(2)) / (2 * m)
        
        return loss


class ConstrainedSpectralLoss(nn.Module):
    """SpectralNet Rayleigh-quotient loss with an optional Wang-Davidson
    constrained-spectral-clustering regularizer.

    References
    ----------
    Wang & Davidson, "On Constrained Spectral Clustering and Its
    Applications", arXiv:1201.5338 (2012). The paper solves a
    generalized eigenproblem
        ``L_bar v = lambda (Q_bar - (alpha/N) I) v``
    where ``Q in {-1, 0, +1}`` encodes cannot-link / unknown / must-link
    constraints and ``Q_bar = D^{-1/2} Q D^{-1/2}``. Here we use a
    Lagrangian-relaxation form suitable for SGD on a neural spectral
    embedder:
        ``L = L_spec(Y, W) - mu * R_wd(Y, labels)``.
    The first term is unchanged from :class:`SpectralNetLoss`.

    Scale-invariance fix
    --------------------
    A naive ``R_wd = tr(Y^T Q_bar Y) / m`` is **quadratic in Y**, and
    because SpectralNet's QR orthonormalization only constrains
    ``Y^T Y`` on a *different* random batch, there is no strict bound
    on the global scale of Y during training. Once the WD reward
    exceeds the spectral penalty, the optimizer can drive the total
    loss to -inf by simply inflating ``||Y||`` (both terms scale as
    ``c^2``, and ``wd - spec`` can go to -inf). This manifests as
    runaway training: ``spec`` and ``|wd|`` growing together in
    lock-step with the same ratio, losing all task signal.

    To make the regularizer robust in the SGD setting we use
    *direction-only* (cosine-similarity) and *pair-averaged* inner
    products:
        ``R_wd = mean_{(i,j) in ML} cos(y_i, y_j)
                 - cl_weight * mean_{(i,j) in CL} cos(y_i, y_j)``
    where ``y_i`` are L2-row-normalized versions of the network output.
    This keeps the *spirit* of Wang-Davidson (reward same-label
    alignment, penalize cross-label alignment) while being

        * scale-invariant:   cos is invariant under ``Y -> c * Y``;
        * size-invariant:    per-pair averaging removes the ``|ML|``
                             vs ``|CL|`` imbalance that would otherwise
                             make CL dominate on multi-class problems;
        * bounded:           ``R_wd in [-(1+|cl_weight|), 1+|cl_weight|]``,
                             so no degenerate scale-up attractor.

    The within-class pair sum is computed analytically from per-class
    centroid sums on the unit-normalized embeddings, which is
    ``O(n_tr * D)`` per batch.

    Auto-scaled mu (``mu_mode='auto'``)
    -----------------------------------
    The pair-averaged cosine reward lives in ``[-1, 1]``, whereas
    ``L_spec`` scales with graph size and degree (on the big inductive
    splits it sits in the range 2-80). A *fixed* mu therefore makes
    ``|mu * wd_reward|`` a vanishingly small fraction of ``L_spec``,
    which is exactly what the mu=10 run on ogbn-arxiv / reddit /
    ogbn-products showed (WD term ~1-5% of total loss, barely visible
    in gradients).

    In ``auto`` mode mu is rescaled per batch so that the WD term is a
    configurable target fraction of ``|L_spec|``:
        ``mu_inst = target_ratio * |spec_loss| / max(|wd_reward|, eps)``
    and EMA-smoothed (in-module buffer ``mu_eff``) across batches to
    avoid per-batch swings, then clamped into ``[mu_min, mu_max]``.
    The EMA update and the final scalar ``mu_eff`` are computed under
    ``no_grad`` so the auto-scaling never perturbs the gradient beyond
    the pre-computed scalar. When ``wd_reward`` is uninformative
    (|value| below eps) mu_eff is not updated, so a random-init batch
    cannot drive mu to explode.
    """

    def __init__(self, mu: float = 0.0, mu_mode: str = "static",
                 target_ratio: float = 0.2,
                 mu_min: float = 0.0, mu_max: float = 1e6,
                 mu_ema_decay: float = 0.9):
        super(ConstrainedSpectralLoss, self).__init__()
        self.mu = float(mu)
        mode = str(mu_mode).lower()
        if mode not in ("static", "auto"):
            raise ValueError(
                "mu_mode must be 'static' or 'auto', got {!r}".format(
                    mu_mode))
        self.mu_mode = mode
        self.target_ratio = float(target_ratio)
        self.mu_min = float(mu_min)
        self.mu_max = float(mu_max)
        self.mu_ema_decay = float(mu_ema_decay)
        # EMA of the effective mu; exposed as a buffer so it is saved /
        # restored with model state_dict. Initialized to the configured
        # static mu (auto mode then immediately warms it up with the
        # first non-degenerate batch).
        self.register_buffer(
            "mu_eff", torch.tensor(float(mu)))

    def forward(self, W: torch.Tensor, Y: torch.Tensor,
                is_normalized: bool = False,
                train_idx: torch.Tensor = None,
                labels: torch.Tensor = None,
                cl_weight: float = None):
        """Compute ``L_spec + L_wd``.

        Args:
            W, Y, is_normalized:        as in :class:`SpectralNetLoss`.
            train_idx (LongTensor):     batch-local indices of labeled
                                        nodes (shape ``[n_tr]``). When
                                        None or empty, the WD term is 0.
            labels (LongTensor):        class ids for ``train_idx``
                                        (shape ``[n_tr]``).
            cl_weight (float or None):  scalar weight on the cannot-link
                                        mean-cosine term. When None,
                                        defaults to 1.0 (equal weighting
                                        of the two averaged terms).

        Returns:
            (total_loss, spec_loss_detached, wd_loss_detached,
             effective_mu_detached). The detached tensors are returned
            only for logging; ``effective_mu_detached`` is the scalar
            multiplier actually applied this batch (static mu in
            'static' mode, current EMA value in 'auto' mode).
        """
        m = Y.size(0)
        zero = Y.new_zeros(())

        if is_normalized:
            d = torch.sum(W, dim=1).clamp(min=1e-8)
            Y_spec = Y / torch.sqrt(d[:, None])
        else:
            Y_spec = Y

        Dy = torch.cdist(Y_spec, Y_spec)
        spec_loss = torch.sum(W * Dy.pow(2)) / (2 * m)

        # Static-mu-zero fast path reproduces vanilla spectral loss.
        if (self.mu_mode == "static" and self.mu == 0.0):
            return (spec_loss + zero, spec_loss.detach(), zero.detach(),
                    Y.new_tensor(0.0))

        if (train_idx is None or labels is None
                or train_idx.numel() < 2):
            eff_mu_val = (self.mu if self.mu_mode == "static"
                          else float(self.mu_eff.item()))
            return (spec_loss + zero, spec_loss.detach(), zero.detach(),
                    Y.new_tensor(eff_mu_val))

        # L2-row-normalize Y (*not* Y_spec): we want direction only, so
        # the WD term is invariant under global scale of the network
        # output and cannot be exploited by inflating ||Y||.
        Y_unit = F.normalize(Y, p=2, dim=-1, eps=1e-8)
        Y_lab = Y_unit[train_idx]
        n_tr = int(train_idx.numel())

        S_all = Y_lab.sum(0)
        # ||y_i|| = 1 for all labeled i, so sum ||y_i||^2 = n_tr.
        # all_pair_sum = 1/2 * (||S_all||^2 - n_tr)
        #              = sum_{i<j, labeled} cos(y_i, y_j).
        all_pair_sum = 0.5 * (S_all.dot(S_all) - float(n_tr))

        unique_cls, inv = torch.unique(labels, return_inverse=True)
        ml_pair_sum = Y.new_zeros(())
        ml_pair_count = 0.0
        for c_idx in range(unique_cls.numel()):
            mask_c = (inv == c_idx)
            nc = int(mask_c.sum().item())
            if nc < 2:
                continue
            Y_c = Y_lab[mask_c]
            S_c = Y_c.sum(0)
            # sq_c = nc (unit norms).
            ml_pair_sum = ml_pair_sum + 0.5 * (S_c.dot(S_c) - float(nc))
            ml_pair_count += nc * (nc - 1) / 2.0

        total_pair_count = n_tr * (n_tr - 1) / 2.0
        cl_pair_count = total_pair_count - ml_pair_count
        cl_pair_sum = all_pair_sum - ml_pair_sum

        ml_denom = max(ml_pair_count, 1.0)
        cl_denom = max(cl_pair_count, 1.0)
        # Per-pair-averaged cosine similarities, both bounded in [-1, 1].
        ml_avg = ml_pair_sum / ml_denom
        cl_avg = cl_pair_sum / cl_denom

        cl_w = 1.0 if cl_weight is None else float(cl_weight)

        # Reward: class alignment contrast. Positive when same-class
        # embeddings are more aligned than cross-class ones.
        wd_reward = ml_avg - cl_w * cl_avg

        # ---- Resolve effective mu ----
        if self.mu_mode == "auto":
            # Use a *detached* instantaneous target mu; the EMA buffer
            # itself is also updated under no_grad so the
            # auto-scaling does not back-prop into the WD gradient
            # beyond the scalar we apply.
            with torch.no_grad():
                spec_abs = float(spec_loss.detach().abs().item())
                wd_abs = float(wd_reward.detach().abs().item())
                # eps floor: prevents a near-zero wd_reward at
                # random init from producing a stratospheric mu. With
                # pair-averaged cosines in [-1, 1], wd_abs ~ 1e-2 is a
                # reasonable "signal threshold".
                eps = 1e-2
                if wd_abs > eps:
                    mu_inst = self.target_ratio * spec_abs / wd_abs
                    mu_inst = max(self.mu_min,
                                  min(self.mu_max, mu_inst))
                    # In-place EMA on the buffer. No grad flows here.
                    self.mu_eff.mul_(self.mu_ema_decay).add_(
                        (1.0 - self.mu_ema_decay) * mu_inst)
                    self.mu_eff.clamp_(self.mu_min, self.mu_max)
                # else: skip update -> keep previous mu_eff (warm-up
                # stays at the configured static mu for degenerate
                # batches).
            effective_mu = float(self.mu_eff.item())
        else:
            effective_mu = self.mu

        wd_loss = -effective_mu * wd_reward

        return (spec_loss + wd_loss,
                spec_loss.detach(),
                wd_loss.detach(),
                Y.new_tensor(effective_mu))



class SpectralTrainer:
    def __init__(self, config: dict, device: torch.device, is_sparse: bool = False):
        """
        This class is responsible for training the SpectralNet model.

        Args:
            config (dict):                  The configuration dictionary
            device (torch.device):          The device to use for training
            is_sparse (bool, optional):     Whether the graph-laplacian obtained from a mini-batch is sparse or not.
                                            In case it is sparse, we build the batch by taking 1/5 of the original random batch,
                                            and then we add to each sample 4 of its nearest neighbors. Defaults to False.
        """

        self.device = device
        self.is_sparse = is_sparse
        self.spectral_config = config["spectral"]
        self.dataset = config["dataset"]
        # Controls how many-hop walks are used to build the "valid affinity" matrix:
        # 2 => W + W^2, 3 => W + W^2 + W^3, etc.
        self.affinity_walk_order = self.spectral_config.get("affinity_walk_order", 3)
        # Toggle for symmetric unnormalized multihop affinity (current / post-fixes
        # behavior). When False, reproduce the legacy asymmetric L2-row-normalized
        # multihop affinity used before the spectral-loss fixes. This toggle is
        # used to reproduce the "original" config iteration in the NMI/ARI/ACC
        # ablation without having to check out old code.
        self.affinity_symmetric_multihop = bool(
            self.spectral_config.get("affinity_symmetric_multihop", True))
        self.lr = self.spectral_config["lr"]
        self.epochs = self.spectral_config["epochs"]
        self.lr_decay = self.spectral_config["lr_decay"]
        self.patience = self.spectral_config["patience"]
        self.batch_size = self.spectral_config["batch_size"]
        self.architecture = self.spectral_config["architecture"]
        self.n_clusters = self.spectral_config["n_clusters"]
        self.output_dim = int(self.architecture["output_dim"])

        # ------------------------------------------------------------
        # Phase-1 ablation knobs (task-aware loss / spectral-PE
        # pretraining / edge dropout). All default to off so legacy
        # configs are unaffected.
        # ------------------------------------------------------------
        ta_cfg = self.spectral_config.get("task_aware", {}) or {}
        self.task_aware_enabled = bool(ta_cfg.get("enabled", False))
        self.task_aware_lambda = float(ta_cfg.get("lambda", 0.0))
        # Number of classes for the CE head. Defaults to n_clusters
        # (which equals the number of true classes for the small
        # datasets used in phase 1: cora=7, citeseer=6, pubmed=3,
        # wikics=10, chameleon=5, squirrel=5).
        self.task_aware_n_classes = int(
            ta_cfg.get("n_classes", self.n_clusters))

        pe_cfg = self.spectral_config.get("pe_pretrain", {}) or {}
        self.pe_pretrain_enabled = bool(pe_cfg.get("enabled", False))
        self.pe_pretrain_epochs = int(pe_cfg.get("epochs", 50))
        self.pe_pretrain_lr = float(pe_cfg.get("lr", self.spectral_config.get("lr", 1e-3)))
        # Print MSE every N pretrain epochs (1 = every epoch).
        self.pe_pretrain_print_every = int(pe_cfg.get("print_every", 10))

        # Print main-training (Epoch / Train Loss / Valid Loss / LR) every N
        # epochs. 1 (default) reproduces legacy behavior of printing every epoch.
        self.train_print_every = max(
            1, int(self.spectral_config.get("print_every", 1)))

        self.edge_dropout = float(self.spectral_config.get("edge_dropout", 0.0))
        # Set per-call by the train loop so val passes never apply dropout.
        self._is_training_step = False

        # Weight on the spectral (Rayleigh + WD) part of the loss. Default
        # 1.0 reproduces legacy behavior. Setting to 0.0 yields a pure
        # task-aware (CE-only) sanity-check baseline that lets us measure
        # how much the spectral objective is actually contributing to
        # downstream accuracy. When 0, val-loss monitoring switches to the
        # mean training CE so the scheduler / best-checkpoint logic still
        # tracks something the gradient is actually optimizing.
        self.spectral_loss_weight = float(
            self.spectral_config.get("spectral_loss_weight", 1.0))

        # Wang-Davidson constrained-spectral-clustering regularizer.
        # When enabled (default), the spectral loss is augmented with
        #   - mu / m * tr(Y^T Q_bar Y)
        # where Q encodes must-link / cannot-link constraints derived from
        # the training labels available in the current batch. See
        # ConstrainedSpectralLoss for details.
        wd_cfg = self.spectral_config.get("constrained_loss", {}) or {}
        self.wd_enabled = bool(wd_cfg.get("enabled", True))
        self.wd_mu = float(wd_cfg.get("mu", 0.5))
        _wd_cl_w = wd_cfg.get("cl_weight", None)
        self.wd_cl_weight = (float(_wd_cl_w) if _wd_cl_w is not None
                             else None)
        self.wd_min_train_nodes = int(
            wd_cfg.get("min_train_nodes", 2))
        # Auto-scaled mu. When ``mu_mode == 'auto'`` the trainer lets
        # ``ConstrainedSpectralLoss`` rescale mu per-batch so that
        # ``|mu * wd_reward|`` targets ``target_ratio * |spec_loss|``,
        # EMA-smoothed and clamped. This is the recommended setting on
        # large graphs, where the fixed-mu WD term is otherwise too
        # small relative to the ``O(graph-size)`` spectral loss.
        self.wd_mu_mode = str(wd_cfg.get("mu_mode", "static")).lower()
        self.wd_target_ratio = float(wd_cfg.get("target_ratio", 0.2))
        self.wd_mu_min = float(wd_cfg.get("mu_min", 0.0))
        self.wd_mu_max = float(wd_cfg.get("mu_max", 1e6))
        self.wd_mu_ema_decay = float(wd_cfg.get("mu_ema_decay", 0.9))

        # ------------------------------------------------------------
        # LR-scheduler tunables. The default
        # ``torch.optim.lr_scheduler.ReduceLROnPlateau`` uses
        # ``threshold=1e-4, threshold_mode='rel', cooldown=0``, which
        # on noisy validation losses (e.g. big inductive graphs) makes
        # almost every non-global-best epoch count as a "bad" epoch
        # and crashes the LR after ~patience epochs. We expose the
        # three knobs here so datasets can use a larger threshold
        # and/or a cooldown to ride out plateaus without prematurely
        # decaying the LR.
        # ------------------------------------------------------------
        self.lr_threshold = float(
            self.spectral_config.get("lr_threshold", 1e-4))
        self.lr_threshold_mode = str(
            self.spectral_config.get("lr_threshold_mode", "rel"))
        self.lr_cooldown = int(
            self.spectral_config.get("lr_cooldown", 0))
        # Save model weights under ./weights/<dataset>/, creating the directory if needed
        self.weights_path = f"./weights/{self.dataset}/spectralnet_weights.pth"
        os.makedirs(os.path.dirname(self.weights_path), exist_ok=True)


    def _pretrain_pe_init(self):
        """Spectral-PE pretraining: MSE pretrain the MLP toward the bottom-k
        eigenvectors of each batch's L_sym(W_acc).

        For each METIS partition we build the same multihop affinity used
        by the Rayleigh loss, form L_sym, take the smallest output_dim+1
        eigenpairs (drop the trivial constant), and MSE-train Y against
        them with a Procrustes orthogonal alignment per batch:
            R* = U V^T   from SVD(Y^T V_target)
            loss = || Y R* - V_target ||_F^2

        Each batch is small (METIS partition size, typically a few
        hundred to a few thousand nodes), so a dense torch.linalg.eigh is
        fast (~ms). Per-batch eigenvectors are not globally consistent
        across partitions -- this is fine for *initialization*: we just
        want the network to start from "something close to the bottom-k
        spectrum on every local subgraph", and the subsequent Rayleigh
        phase globalizes it.
        """
        n_batches = len(self.features_batches)
        print("[pe-init] Spectral-PE pretraining: epochs={} lr={} batches={} target=L_sym(W_acc) walk_order={}".format(
            self.pe_pretrain_epochs, self.pe_pretrain_lr, n_batches,
            self.affinity_walk_order))

        # Use a separate Adam optimizer for the pretraining phase so the
        # main-phase optimizer state starts fresh.
        pretrain_optim = optim.Adam(self.spectral_net.parameters(),
                                    lr=self.pe_pretrain_lr)

        # Pretraining never uses edge dropout (we want to fit the true
        # operator, not a stochastic version of it).
        prev_drop_flag = self._is_training_step
        self._is_training_step = False

        try:
            for epoch in range(self.pe_pretrain_epochs):
                epoch_loss = 0.0
                for pid in range(n_batches):
                    features_b = self.features_batches[pid].to(self.device)
                    support_b = self.support_batches[pid]
                    W = self._get_support_matrix(support_b)
                    perm = np.random.permutation(len(features_b))
                    features_b = features_b[perm]
                    W = W[perm][:, perm]

                    # Build the same multihop affinity used by the loss.
                    W_acc = self._get_valid_affinity_matrix(W, features_b)

                    # Build L_sym = I - D^{-1/2} W_acc D^{-1/2} (dense; cheap on partition size).
                    with torch.no_grad():
                        deg = W_acc.sum(dim=1)
                        d_inv_sqrt = torch.where(
                            deg > 0,
                            deg.clamp(min=1e-30).pow(-0.5),
                            torch.zeros_like(deg))
                        D = torch.diag(d_inv_sqrt)
                        L = torch.eye(W_acc.shape[0], device=self.device,
                                      dtype=W_acc.dtype) - D @ W_acc @ D
                        L = 0.5 * (L + L.t())
                        # Smallest k+1 eigenpairs, drop trivial (~0).
                        eigvals, eigvecs = torch.linalg.eigh(L)
                        k_take = min(self.output_dim + 1, eigvecs.shape[1])
                        V_target = eigvecs[:, 1:k_take].detach()
                        # If we got fewer than output_dim due to small
                        # partition, pad with zeros (rare).
                        if V_target.shape[1] < self.output_dim:
                            pad = torch.zeros(
                                V_target.shape[0],
                                self.output_dim - V_target.shape[1],
                                device=self.device, dtype=V_target.dtype)
                            V_target = torch.cat([V_target, pad], dim=1)

                    # Refresh the network's orthonormalization weights
                    # (forward in eval mode), then a normal forward for the
                    # gradient.
                    self.spectral_net.eval()
                    self.spectral_net(features_b, should_update_orth_weights=True)
                    self.spectral_net.train()
                    Y = self.spectral_net(features_b, should_update_orth_weights=False)

                    # Procrustes alignment: best orthogonal R rotating Y onto V_target.
                    M = Y.t() @ V_target
                    U_p, _, Vh_p = torch.linalg.svd(M, full_matrices=False)
                    R = U_p @ Vh_p
                    loss = F.mse_loss(Y @ R, V_target)

                    pretrain_optim.zero_grad()
                    loss.backward()
                    pretrain_optim.step()
                    epoch_loss += float(loss.item())

                if (epoch + 1) % self.pe_pretrain_print_every == 0 or epoch == 0:
                    print("[pe-init] epoch {}/{} mse={:.6f}".format(
                        epoch + 1, self.pe_pretrain_epochs,
                        epoch_loss / max(1, n_batches)))
        finally:
            self._is_training_step = prev_drop_flag


    def train(self, train, val) -> SpectralNetModel:
        """
        This function trains the SpectralNet model.

        Args:
            train (tuple):                  The training data
            adj_dict (dict):                The adjacency dictionary
            sparse_adj (torch.sparse.Tensor): The sparse adjacency matrix
        Returns:
            SpectralNetModel: The trained SpectralNet model
        """
        self.parts, self.features_batches, self.support_batches, self.y_train_batches, self.train_mask_batches = train
        self.idx_parts, self.val_features_batches, self.val_support_batches, self.y_val_batches, self.val_mask_batches = val
        self.idx_parts = list(range(len(self.parts)))
        self.counter = 0
        # Use ConstrainedSpectralLoss with mu=0 when WD is disabled; that
        # reproduces the exact vanilla SpectralNet objective (spec only).
        effective_mu = self.wd_mu if self.wd_enabled else 0.0
        effective_mu_mode = (self.wd_mu_mode if self.wd_enabled
                             else "static")
        self.criterion = ConstrainedSpectralLoss(
            mu=effective_mu,
            mu_mode=effective_mu_mode,
            target_ratio=self.wd_target_ratio,
            mu_min=self.wd_mu_min,
            mu_max=self.wd_mu_max,
            mu_ema_decay=self.wd_mu_ema_decay,
        ).to(self.device)
        print(
            "[wd-loss] enabled={} mu_mode={} mu={} cl_weight={} "
            "min_train_nodes={}".format(
                self.wd_enabled, effective_mu_mode, effective_mu,
                (self.wd_cl_weight if self.wd_cl_weight is not None
                 else "auto"),
                self.wd_min_train_nodes))
        if effective_mu_mode == "auto":
            print(
                "[wd-loss] auto-mu: target_ratio={} mu_min={} mu_max={} "
                "mu_ema_decay={}".format(
                    self.wd_target_ratio, self.wd_mu_min, self.wd_mu_max,
                    self.wd_mu_ema_decay))
        self.spectral_net = SpectralNetModel(self.architecture, input_dim=self.features_batches[0].shape[1], device=self.device).to(self.device)

        # Optional task-aware classification head: a single linear layer
        # on top of Y, supervised on labeled train nodes within each
        # batch. Disabled by default; enabled via spectral.task_aware.
        # We derive n_classes from the actual one-hot label tensor (last
        # dim of y_train_b) instead of trusting spectral.n_clusters,
        # since some dataset configs (e.g. wikics, ogbn-arxiv, reddit,
        # ogbn-products) use a clustering-oriented spectral.n_clusters
        # that does NOT match the number of label classes. Trusting that
        # value caused CE to be built with too few output logits, which
        # tripped CUDA device-side asserts as soon as a label index of
        # >= n_clusters appeared in a batch.
        if self.task_aware_enabled:
            data_n_classes = int(self.y_train_batches[0].shape[-1])
            if data_n_classes != self.task_aware_n_classes:
                print(
                    "[task-aware] overriding n_classes: config={} -> "
                    "data={} (derived from y_train.shape[-1])".format(
                        self.task_aware_n_classes, data_n_classes))
            self.task_aware_n_classes = data_n_classes
            self.task_head = nn.Linear(
                self.output_dim, self.task_aware_n_classes
            ).to(self.device)
            optim_params = (list(self.spectral_net.parameters())
                            + list(self.task_head.parameters()))
            print("[task-aware] enabled lambda={} n_classes={}".format(
                self.task_aware_lambda, self.task_aware_n_classes))
        else:
            self.task_head = None
            optim_params = list(self.spectral_net.parameters())

        # Optional Spectral-PE pretraining (MSE pretrain Y toward bottom-k
        # eigenvectors of L_sym(W_acc) per batch). Runs before the main
        # Rayleigh-quotient training loop.
        if self.pe_pretrain_enabled:
            self._pretrain_pe_init()

        # Optional edge dropout (applied per training step inside
        # _get_valid_affinity_matrix); val passes never see it.
        if self.edge_dropout > 0.0:
            print("[edge-dropout] p={} (training steps only)".format(
                self.edge_dropout))

        # Pure-CE / lambda->infinity baseline. When the spectral weight is
        # zeroed, the only learning signal is the task-aware head's CE on
        # labeled train nodes. Print a clear warning if task-aware is off
        # in this regime since there will be no learning signal at all.
        if self.spectral_loss_weight != 1.0:
            print("[spectral-weight] w={}".format(self.spectral_loss_weight))
            if (self.spectral_loss_weight <= 0.0
                    and not self.task_aware_enabled):
                print("[spectral-weight] WARNING: spectral_loss_weight=0 "
                      "without task_aware enabled -> no learning signal.")

        self.optimizer = optim.Adam(optim_params, lr=self.lr)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=self.lr_decay,
            patience=self.patience,
            threshold=self.lr_threshold,
            threshold_mode=self.lr_threshold_mode,
            cooldown=self.lr_cooldown,
        )
        print(
            "[lr-schedule] initial_lr={} min_lr={} decay={} patience={} "
            "threshold={} mode={} cooldown={}".format(
                self.lr, self.spectral_config["min_lr"], self.lr_decay,
                self.patience, self.lr_threshold, self.lr_threshold_mode,
                self.lr_cooldown))


        min_val_loss = 1000000000
        batches = len(self.features_batches)
        batches_val = len(self.val_features_batches)
        pure_task_aware_only = (
            self.spectral_loss_weight <= 0.0
            and self.task_aware_enabled
            and self.task_aware_lambda > 0.0
        )
        grassmans = []
        print("Training SpectralNet:")
        for epoch in range(self.epochs):
            train_loss = 0.0
            train_spec_loss = 0.0
            train_wd_loss = 0.0
            train_mu_eff_sum = 0.0
            train_mu_eff_count = 0
            train_wd_active_batches = 0
            self._train_task_ce_sum = 0.0

            for pid in range(batches):
                pid_2 = np.random.choice(batches)

                # Use preprocessed batch data
                features_b = self.features_batches[pid].to(self.device)                
                support_b = self.support_batches[pid]
                y_train_b = self.y_train_batches[pid].to(self.device)
                train_mask_b_np = self.train_mask_batches[pid]
                train_mask_b = torch.as_tensor(
                    train_mask_b_np, dtype=torch.bool, device=self.device)
                W = self._get_support_matrix(support_b)
                perm = np.random.permutation(len(features_b))
                features_b = features_b[perm]
                W = W[perm][:, perm]
                y_train_b = y_train_b[perm]
                train_mask_b = train_mask_b[perm]
                
                features_b_2 = self.features_batches[pid_2].to(self.device)

                perm = np.random.permutation(len(features_b_2))
                features_b_2 = features_b_2[perm]

                # Orthogonality step
                self.spectral_net.eval()
                self.spectral_net(features_b_2, should_update_orth_weights=True)

                # Gradient step
                self.spectral_net.train()
                self.optimizer.zero_grad()


                Y = self.spectral_net(features_b, should_update_orth_weights=False)

                if pure_task_aware_only:
                    # In pure-CE mode, skip building/normalizing affinity
                    # entirely to avoid OOM from dense W_acc on large batches.
                    loss = torch.zeros((), device=self.device)
                    l_spec = torch.zeros((), device=self.device)
                    l_wd = torch.zeros((), device=self.device)
                    mu_eff_t = torch.zeros((), device=self.device)
                else:
                    # Affinity matrix for loss computation. The training-step
                    # flag enables per-batch edge dropout when configured.
                    self._is_training_step = True
                    W = self._get_valid_affinity_matrix(W, features_b)
                    self._is_training_step = False

                    # Wang-Davidson constraint inputs, built per-batch from
                    # the labeled nodes that happen to fall into this
                    # METIS partition. Skipped (WD term = 0) when fewer than
                    # ``wd_min_train_nodes`` labeled nodes are present, or
                    # when mu == 0.
                    wd_kwargs = {}
                    wd_active = self.wd_enabled and (
                        self.wd_mu != 0.0 or self.wd_mu_mode == "auto")
                    if wd_active:
                        train_idx_local = train_mask_b.nonzero(
                            as_tuple=True)[0]
                        if (train_idx_local.numel()
                                >= self.wd_min_train_nodes):
                            labels_local = y_train_b[train_idx_local].argmax(
                                dim=-1)
                            wd_kwargs = dict(
                                train_idx=train_idx_local,
                                labels=labels_local,
                                cl_weight=self.wd_cl_weight,
                            )

                    loss, l_spec, l_wd, mu_eff_t = self.criterion(
                        W, Y, is_normalized=True, **wd_kwargs)

                    # Scale the spectral (+WD) part by spectral_loss_weight.
                    # Default 1.0 preserves legacy behavior; 0.0 zeros the
                    # spectral gradient signal entirely, leaving only the CE
                    # term added below (pure task-aware baseline).
                    if self.spectral_loss_weight != 1.0:
                        loss = self.spectral_loss_weight * loss

                # Task-aware CE on labeled train nodes within the batch.
                # Skipped when the head is disabled, when no train nodes
                # are present in the partition, or when lambda == 0.
                task_ce_item = 0.0
                if (self.task_aware_enabled and self.task_head is not None
                        and self.task_aware_lambda > 0.0):
                    train_idx_local = train_mask_b.nonzero(as_tuple=True)[0]
                    if train_idx_local.numel() > 0:
                        logits = self.task_head(Y[train_idx_local])
                        labels_local = y_train_b[train_idx_local].argmax(
                            dim=-1)
                        ce = F.cross_entropy(logits, labels_local)
                        loss = loss + self.task_aware_lambda * ce
                        task_ce_item = float(ce.detach().item())

                if loss.requires_grad:
                    loss.backward()
                    self.optimizer.step()
                train_loss += loss.item()
                train_spec_loss += l_spec.item()
                wd_item = l_wd.item()
                train_wd_loss += wd_item
                if wd_item != 0.0:
                    train_wd_active_batches += 1
                    train_mu_eff_sum += float(mu_eff_t.item())
                    train_mu_eff_count += 1
                self._train_task_ce_sum += task_ce_item

            # Validation step. Val loss stays spectral-only (val labels
            # are never used to update parameters, and the scheduler /
            # checkpointing downstream should track the primary Rayleigh
            # objective, not the auxiliary constraint reward).
            valid_loss = 0.0
            self.spectral_net.eval()
            if not pure_task_aware_only:
                with torch.no_grad():
                    for pid in range(batches_val):
                        features_b = self.val_features_batches[pid].to(self.device)
                        support_b = self.val_support_batches[pid]
                        y_val_b = self.y_val_batches[pid]
                        val_mask_b = self.val_mask_batches[pid]

                        W = self._get_support_matrix(support_b)

                        Y = self.spectral_net(features_b, should_update_orth_weights=False)

                        W = self._get_valid_affinity_matrix(W, features_b)

                        _total, l_spec_v, _, _ = self.criterion(
                            W, Y, is_normalized=True)
                        valid_loss += l_spec_v.item()

            # When spectral_loss_weight==0 the spectral objective is not
            # being optimized, so val spectral loss is meaningless for
            # scheduling / checkpointing. Fall back to the mean training
            # CE in that regime (it's still imperfect because train labels
            # are visible, but it tracks the only quantity the gradient
            # is actually moving). monitor_loss is what we feed into the
            # LR scheduler and the best-checkpoint comparison.
            if (self.spectral_loss_weight <= 0.0
                    and self.task_aware_enabled
                    and self.task_aware_lambda > 0.0):
                monitor_loss = self._train_task_ce_sum / max(1, batches)
            else:
                monitor_loss = valid_loss

            self.scheduler.step(monitor_loss)

            current_lr = self.optimizer.param_groups[0]["lr"]
            if current_lr <= self.spectral_config["min_lr"]: break

            # Throttle per-epoch progress prints. Always print on the
            # first epoch, every ``train_print_every`` epochs, and on the
            # final epoch so the log endpoints are visible.
            should_print_epoch = (
                self.train_print_every <= 1
                or epoch == 0
                or ((epoch + 1) % self.train_print_every == 0)
                or ((epoch + 1) == self.epochs)
            )
            if should_print_epoch:
                if self.wd_enabled and (self.wd_mu != 0.0
                                        or self.wd_mu_mode == "auto"):
                    mu_eff_avg = (train_mu_eff_sum / train_mu_eff_count
                                  if train_mu_eff_count > 0 else 0.0)
                    print(
                        "Epoch: {}/{}, Train Loss: {:.7f} (spec={:.7f}, wd={:.7f}, "
                        "mu_eff={:.4g}, wd_active={}/{} batches), "
                        "Valid Loss: {:.7f}, LR: {:.6f}".format(
                            epoch + 1, self.epochs,
                            train_loss / batches,
                            train_spec_loss / batches,
                            train_wd_loss / batches,
                            mu_eff_avg,
                            train_wd_active_batches, batches,
                            valid_loss / batches_val,
                            current_lr))
                else:
                    print(
                        "Epoch: {}/{}, Train Loss: {:.7f}, Valid Loss: {:.7f}, LR: {:.6f}".format(
                            epoch + 1, self.epochs,
                            train_loss / batches,
                            valid_loss / batches_val,
                            current_lr))
                if self.task_aware_enabled and self.task_aware_lambda > 0.0:
                    print("    [task-aware] mean CE / batch: {:.6f}".format(
                        self._train_task_ce_sum / max(1, batches)))


            if monitor_loss <= min_val_loss:
                min_val_loss = monitor_loss
                self.save_model()
                

        # Restore best-val-loss checkpoint before returning. The training
        # loop was already calling ``save_model()`` on every improvement,
        # but the returned network was the *final* state, which can be
        # substantially worse than the best one (and was catastrophic on
        # some datasets when the WD regularizer destabilized training).
        if os.path.exists(self.weights_path):
            try:
                state = torch.load(self.weights_path,
                                   map_location=self.device)
                self.spectral_net.load_state_dict(state)
                print("[checkpoint] restored best val-loss weights from "
                      f"{self.weights_path}")
            except Exception as exc:
                print(f"[checkpoint] could not restore best weights: {exc}")
        return self.spectral_net


    def _get_support_matrix(self, support):
        indices = torch.LongTensor(support[0])
        values = torch.FloatTensor(support[1])
        size = support[2]
        W = torch.sparse.FloatTensor(indices.t(), values, size).to_dense()
        W = W.to(device=self.device)
        return W


    def _get_valid_affinity_matrix(self, W, X):
        """Build the symmetric multihop affinity used by the spectral loss.

        The upstream preprocessing passes the raw symmetric adjacency
        submatrix (no row-stochastic normalization), so ``W`` here should be
        symmetric. We drop self-loops before powering so that ``W^k`` counts
        length-``k`` walks, accumulate ``W + W^2 + ... + W^p`` with
        ``p == affinity_walk_order``, and zero the diagonal of the result.

        The returned affinity is **unnormalized** and **symmetric**. The
        spectral loss, called with ``is_normalized=True``, will then form
        ``d = sum_j W_ij`` from the row-sums of this symmetric matrix and
        optimize the Rayleigh quotient of
        ``L_sym = I - D^{-1/2} W D^{-1/2}``, which is exactly the operator
        the eigenspace diagnostic compares against in ``sym_multihop`` mode.

        When ``spectral.affinity_symmetric_multihop`` is False, fall back to
        the legacy asymmetric L2-row-normalized multihop affinity (used by
        the "original" configuration iteration for the ablation study).
        """
        if not self.affinity_symmetric_multihop:
            # Legacy / pre-fix behavior: operate on W as-is (row-stochastic
            # when upstream preprocess normalizes it), sum multihop walks,
            # zero the diagonal, and row-L2-normalize. Asymmetric.
            edge_mask = W > 0
            W[edge_mask] += 1e-6
            del edge_mask
            if self.affinity_walk_order > 1:
                W2 = W @ W
                if self.affinity_walk_order >= 3:
                    W3 = W2 @ W
                    W = W + W2 + W3
                    del W3
                else:
                    W = W + W2
                del W2
            W.fill_diagonal_(0)
            W = F.normalize(W, p=2, dim=1)
            return W

        # Drop self-loops before powering.
        W.fill_diagonal_(0)
        # Defensive symmetrization (guards against numerical drift in case
        # the upstream submatrix extraction produces tiny asymmetries).
        W = 0.5 * (W + W.t())

        # Edge dropout: randomly drop a Bernoulli fraction of (symmetric)
        # edges before powering. Only active during training steps; val
        # passes always see the full graph.
        if self._is_training_step and self.edge_dropout > 0.0:
            n = W.shape[0]
            keep_p = 1.0 - self.edge_dropout
            # Sample on the upper triangle (excl. diag) so the mask is
            # symmetric and we don't double-drop each undirected edge.
            upper = torch.bernoulli(
                torch.full((n, n), keep_p, device=W.device, dtype=W.dtype)
            ).triu(diagonal=1)
            sym_mask = upper + upper.t()
            W = W * sym_mask

        if self.affinity_walk_order >= 2:
            W2 = W @ W
            if self.affinity_walk_order >= 3:
                W3 = W2 @ W
                W_acc = W + W2 + W3
                del W3
            else:
                W_acc = W + W2
            del W2
        else:
            W_acc = W

        # Remove entries introduced by even-length walks returning to self.
        W_acc.fill_diagonal_(0)
        # Enforce symmetry after matmul (tiny numerical asymmetries can
        # accumulate in fp32).
        W_acc = 0.5 * (W_acc + W_acc.t())
        return W_acc
       
       
    def _get_rotation_matrix(self, batches):
        l = None     
        for pid in range(batches):
            features_b = self.features_batches[pid].to(self.device)                
            support_b = self.support_batches[pid]
            W = self._get_support_matrix(support_b)
            Y = self.spectral_net(features_b, should_update_orth_weights=False)
            
            if l is None:
                l = Y.T @ W @ Y
            else:
                l += Y.T @ W @ Y
        l = l / batches
        l = (l + l.T) / 2
        
        l = l.detach().cpu().numpy()
        ortho_matrix, eigenvalues_pred, _ = np.linalg.svd(l)
        eigenvalues_pred = eigenvalues_pred.real
        self.ortho_matrix = np.array(ortho_matrix.real)
        indices = np.argsort(eigenvalues_pred)
        ortho_matrix = np.array(self.ortho_matrix[:, indices])
        self.spectral_net.set_rotation_matrix(ortho_matrix)


    def _get_affinity_matrix(self, X: torch.Tensor) -> torch.Tensor:
        """
        This function computes the affinity matrix W using the Gaussian kernel.
        Args:
            X (torch.Tensor):   The input data
        Returns:
            torch.Tensor: The affinity matrix W
        """

        is_local = self.spectral_config["is_local_scale"]
        n_neighbors = self.spectral_config["n_neighbors"]
        scale_k = self.spectral_config["scale_k"]
        Dx = torch.cdist(X, X)
        Dis, indices = get_nearest_neighbors(X, k=n_neighbors + 1)
        scale = compute_scale(Dis, k=scale_k, is_local=is_local)
        W = get_gaussian_kernel(Dx, scale, indices, device=self.device, is_local=is_local)
        return W


    def save_model(self):
        """
        This function saves the model.
        """
        torch.save(self.spectral_net.state_dict(), self.weights_path)



class ReduceLROnAvgLossPlateau(_LRScheduler):
    def __init__(self, optimizer, factor=0.1, patience=10, min_lr=0, verbose=False, min_delta=1e-4):
        """
        Custom ReduceLROnPlateau scheduler that uses the average loss instead of the loss of the last epoch.

        Args:
            optimizer (_type_):             The optimizer
            factor (float, optional):       factor by which the learning rate will be reduced. 
                                            new_lr = lr * factor. Defaults to 0.1.
            patience (int, optional):       number of epochs with no average improvement after 
                                            which learning rate will be reduced.
            min_lr (int, optional):         A lower bound on the learning rate of all param groups.
            verbose (bool, optional):       If True, prints a message to stdout for each update.
            min_delta (_type_, optional):   threshold for measuring the new optimum, to only focus on
                                            significant changes. Defaults to 1e-4.
        """

        self.factor = factor
        self.min_delta = min_delta
        self.patience = patience
        self.verbose = verbose
        self.wait = 0
        self.best = 1e5
        self.avg_losses = []
        self.min_lr = min_lr
        super(ReduceLROnAvgLossPlateau, self).__init__(optimizer)

    def get_lr(self):
        return [base_lr * self.factor ** self.num_bad_epochs
                for base_lr in self.base_lrs]

    def step(self, loss=1.0, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch

        current_loss = loss
        if len(self.avg_losses) < self.patience:
            self.avg_losses.append(current_loss)
        else:
            self.avg_losses.pop(0)
            self.avg_losses.append(current_loss)
        avg_loss = sum(self.avg_losses) / len(self.avg_losses)
        if avg_loss < self.best - self.min_delta:
            self.best = avg_loss
            self.wait = 0
        else:
            if self.wait >= self.patience:
                for param_group in self.optimizer.param_groups:
                    old_lr = float(param_group['lr'])
                    if old_lr > self.min_lr:
                        new_lr = old_lr * self.factor
                        new_lr = max(new_lr, self.min_lr)
                        param_group['lr'] = new_lr
                        if self.verbose:
                            print(f'Epoch {epoch}: reducing learning rate to {new_lr}.')
                self.wait = 0
            self.wait += 1
