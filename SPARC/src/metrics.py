import numpy as np
import sklearn.metrics as metrics

from utils import *
from munkres import Munkres
from sklearn.metrics import normalized_mutual_info_score as nmi
from sklearn.metrics import adjusted_rand_score as ari

class Metrics:
    @staticmethod
    def ari_score(cluster_assignments: np.ndarray, y: np.ndarray) -> float:
        """
        Computes the adjusted Rand index score of the clustering algorithm.

        Args:
            cluster_assignments (np.ndarray):   cluster assignments for each data point
            y (np.ndarray):                     ground truth labels

        Returns:
            float: adjusted Rand index score
        """
        return ari(y, cluster_assignments)

    @staticmethod
    def acc_score(cluster_assignments: np.ndarray, y: np.ndarray, n_clusters: int)  -> float:
        """
        Computes the accuracy score of the clustering algorithm
        Args:
            cluster_assignments (np.ndarray):   cluster assignments for each data point
            y (np.ndarray):                     ground truth labels
            n_clusters (int):                   number of clusters

        Returns:
            float: accuracy score
        """

        labels_range = list(range(n_clusters))
        confusion_matrix = metrics.confusion_matrix(
            y, cluster_assignments, labels=labels_range)
        cost_matrix = calculate_cost_matrix(confusion_matrix, n_clusters=n_clusters)
        indices = Munkres().compute(cost_matrix)
        kmeans_to_true_cluster_labels = get_cluster_labels_from_indices(indices)
        y_pred = kmeans_to_true_cluster_labels[cluster_assignments].astype(int)
        accuracy = float(np.mean(y_pred == y))
        return accuracy
    
    @staticmethod
    def nmi_score(cluster_assignments: np.ndarray, y: np.ndarray) -> float:
        """
        Computes the normalized mutual information score of the clustering algorithm
        Args:
            cluster_assignments (np.ndarray):   cluster assignments for each data point
            y (np.ndarray):                     ground truth labels

        Returns:
            float: normalized mutual information score
        """
        return nmi(cluster_assignments, y)

