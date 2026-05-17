# coding=utf-8
# Copyright 2023 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Collections of partitioning functions."""

import os
import time

# Prefer METIS from the active conda env (override a stale METIS_DLL from another env).
if os.environ.get("CONDA_PREFIX"):
    _metis_lib = os.path.join(os.environ["CONDA_PREFIX"], "lib", "libmetis.so")
    if os.path.isfile(_metis_lib):
        os.environ["METIS_DLL"] = _metis_lib

import metis
import scipy.sparse as sp
import tensorflow.compat.v1 as tf
import random

from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import numpy as np


def balanced_kmeans(X, k, max_iters=100):
    # Initialize centroids randomly
    centroids = X[np.random.choice(len(X), k, replace=False)]
    prev_assignments = np.zeros(len(X))
    
    for _ in range(max_iters):
        # Assign each data point to the nearest centroid
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        assignments = np.argmin(distances, axis=1)
        
        # Check for convergence
        if np.array_equal(assignments, prev_assignments):
            break
            
        prev_assignments = assignments
        
        # Update centroids
        for i in range(k):
            cluster_points = X[assignments == i]
            if len(cluster_points) > 0:
                centroids[i] = np.mean(cluster_points, axis=0)
        
    return centroids, assignments


def equal_size_kmeans(data, k, max_iterations=100):
    n, d = data.shape
    target_size = n // k
    
    # Initialize centroids using k-means++
    kmeans = KMeans(n_clusters=k, init='k-means++')
    kmeans.fit(data)
    centroids = kmeans.cluster_centers_
    
    # Assign initial clusters
    distances = np.linalg.norm(data[:, np.newaxis] - centroids, axis=2)
    clusters = np.argmin(distances, axis=1)
    
    # Initialize sizes of clusters
    cluster_sizes = np.bincount(clusters, minlength=k)
    
    for _ in range(max_iterations):
        # Adjust cluster assignments to achieve equal size
        for i in range(n):
            cluster_sizes[clusters[i]] -= 1
            min_size_cluster = np.argmin(cluster_sizes)
            clusters[i] = min_size_cluster
            cluster_sizes[min_size_cluster] += 1
            
        # Recalculate centroids
        for i in range(k):
            centroids[i] = np.mean(data[clusters == i], axis=0)
        
    return clusters, centroids


def initialize_centroids(data, k):
    """
    Randomly initialize k centroids from the data.

    Parameters:
        data (numpy.ndarray): The data points.
        k (int): Number of centroids.

    Returns:
        numpy.ndarray: The initial centroids.
    """
    centroids_indices = np.random.choice(len(data), size=k, replace=False)
    centroids = data[centroids_indices]
    return centroids


def assign_to_clusters(data, centroids):
    """
    Assign each data point to its nearest centroid.

    Parameters:
        data (numpy.ndarray): The data points.
        centroids (numpy.ndarray): The centroids.

    Returns:
        numpy.ndarray: Cluster assignments for each data point.
    """
    distances = np.linalg.norm(data[:, np.newaxis] - centroids, axis=2)
    cluster_assignments = np.argmin(distances, axis=1)
    return cluster_assignments


def update_centroids(data, cluster_assignments, k):
    """
    Update centroids based on current assignments.

    Parameters:
        data (numpy.ndarray): The data points.
        cluster_assignments (numpy.ndarray): Cluster assignments for each data point.
        k (int): Number of centroids.

    Returns:
        numpy.ndarray: Updated centroids.
    """
    centroids = np.empty((k, data.shape[1]))
    for i in range(k):
        centroids[i] = np.mean(data[cluster_assignments == i], axis=0)
    return centroids


def k_means__(data, k, max_iters=100):
    """
    Perform k-means clustering on the given data.

    Parameters:
        data (numpy.ndarray): The data points.
        k (int): Number of clusters.
        max_iters (int): Maximum number of iterations.

    Returns:
        numpy.ndarray: Final centroids.
        numpy.ndarray: Cluster assignments.
    """
    centroids = initialize_centroids(data, k)
    for _ in range(max_iters):
        cluster_assignments = assign_to_clusters(data, centroids)
        new_centroids = update_centroids(data, cluster_assignments, k)
        if np.allclose(new_centroids, centroids):
            break
        centroids = new_centroids
    return centroids, cluster_assignments


def partition_graph(adj, idx_nodes, num_clusters):
  """partition a graph by METIS."""

  start_time = time.time()
  num_nodes = len(idx_nodes)
  num_all_nodes = adj.shape[0]

  neighbor_intervals = []
  neighbors = []
  edge_cnt = 0
  neighbor_intervals.append(0)
  train_adj_lil = adj[idx_nodes, :][:, idx_nodes].tolil()
  train_ord_map = dict()
  train_adj_lists = [[] for _ in range(num_nodes)]
  for i in range(num_nodes):
    rows = train_adj_lil[i].rows[0]
    # self-edge needs to be removed for valid format of METIS
    if i in rows:
      rows.remove(i)
    train_adj_lists[i] = rows
    neighbors += rows
    edge_cnt += len(rows)
    neighbor_intervals.append(edge_cnt)
    train_ord_map[idx_nodes[i]] = i

  if num_clusters > 1:
    _, groups = metis.part_graph(train_adj_lists, num_clusters, seed=1)
    # random.shuffle(groups)
  else:
    groups = [0] * num_nodes

  part_row = []
  part_col = []
  part_data = []
  parts = [[] for _ in range(num_clusters)]
  for nd_idx in range(num_nodes):
    gp_idx = groups[nd_idx]
    nd_orig_idx = idx_nodes[nd_idx]
    parts[gp_idx].append(nd_orig_idx)
    for nb_orig_idx in adj[nd_orig_idx].indices:
      nb_idx = train_ord_map[nb_orig_idx]
      if groups[nb_idx] == gp_idx:
        part_data.append(1)
        part_row.append(nd_orig_idx)
        part_col.append(nb_orig_idx)
  part_data.append(0)
  part_row.append(num_all_nodes - 1)
  part_col.append(num_all_nodes - 1)
  part_adj = sp.coo_matrix((part_data, (part_row, part_col))).tocsr()

  tf.logging.info('Partitioning done. %f seconds.', time.time() - start_time)

  return part_adj, parts
