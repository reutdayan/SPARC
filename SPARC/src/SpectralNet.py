"""High-level API for training and applying the SPARC SpectralNet encoder."""

import numpy as np
import torch
from sklearn.cluster import KMeans

from SpectralTrainer import SpectralTrainer


class SpectralNet:
    """Spectral embedding model: MLP trained with a graph-spectral objective."""

    def __init__(self, n_clusters: int, config: dict):
        self.n_clusters = n_clusters
        self.config = config
        self.embeddings_ = None
        self.spec_net = None
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using {self.device} device")

    def fit(self, train, val):
        """Train SpectralNet on METIS-batched train/val data.

        Args:
            train: ``(parts, features_batches, support_batches,
                     y_train_batches, train_mask_batches)``
            val: ``(idx_parts, val_features_batches, val_support_batches,
                    y_val_batches, val_mask_batches)``
        """
        trainer = SpectralTrainer(self.config, self.device, is_sparse=False)
        self.spec_net = trainer.train(train, val)

    def embed(self, X: torch.Tensor) -> np.ndarray:
        """Return input features on CPU (no forward pass through the MLP)."""
        return np.array(X.to(self.device).detach().cpu())

    def predict(self, X: torch.Tensor) -> np.ndarray:
        """Map node features to spectral embeddings via the trained MLP."""
        X = X.to(self.device)
        self.embeddings_ = (
            self.spec_net(X, should_update_orth_weights=False)
            .detach()
            .cpu()
            .numpy()
        )
        return self.embeddings_

    def cluster_assignments(self, embeddings: np.ndarray) -> np.ndarray:
        """K-means cluster IDs in embedding space."""
        kmeans = KMeans(n_clusters=self.n_clusters).fit(embeddings)
        return kmeans.predict(embeddings)
