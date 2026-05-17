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
from tqdm import tqdm
import time
# Prefer METIS from the active conda env (override METIS_DLL so the active env wins)
if os.environ.get("CONDA_PREFIX"):
    _metis_lib = os.path.join(os.environ["CONDA_PREFIX"], "lib", "libmetis.so")
    if os.path.isfile(_metis_lib):
        os.environ["METIS_DLL"] = _metis_lib
# import metispy as metis
import metis
import scipy.sparse as sp
import random

from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import numpy as np

import partition_cache


def _groups_to_part_adj_and_parts(adj, idx_nodes, groups, num_clusters):
    """Build part_adj and node lists from METIS group ids (local order = idx_nodes)."""
    num_nodes = len(idx_nodes)
    num_all_nodes = adj.shape[0]
    train_ord_map = dict((idx_nodes[i], i) for i in range(num_nodes))
    if not isinstance(groups, np.ndarray):
        groups = np.asarray(groups, dtype=np.int32)
    part_row = []
    part_col = []
    part_data = []
    parts = [[] for _ in range(num_clusters)]
    for nd_idx in range(num_nodes):
        gp_idx = int(groups[nd_idx])
        nd_orig_idx = idx_nodes[nd_idx]
        parts[gp_idx].append(nd_orig_idx)
        for nb_orig_idx in adj[nd_orig_idx].indices:
            nb_idx = train_ord_map[nb_orig_idx]
            if int(groups[nb_idx]) == gp_idx:
                part_data.append(1)
                part_row.append(nd_orig_idx)
                part_col.append(nb_orig_idx)
    part_data.append(0)
    part_row.append(num_all_nodes - 1)
    part_col.append(num_all_nodes - 1)
    part_adj = sp.coo_matrix((part_data, (part_row, part_col))).tocsr()
    return part_adj, parts


def partition_graph(adj, idx_nodes, num_clusters, cache_path=None):
    """Partition a graph by METIS.

    Args:
      adj: Full graph CSR (same convention as original SPARC).
      idx_nodes: Global indices of the subgraph to partition (e.g. train nodes).
      num_clusters: Target number of partitions.
      cache_path: If set, load/save METIS output under this .npz path.
    """

    num_nodes = len(idx_nodes)
    num_all_nodes = adj.shape[0]
    idx_arr = np.asarray(idx_nodes, dtype=np.int64)

    if cache_path:
        cached = partition_cache.try_load_partition_cache(
            cache_path, adj, idx_arr, num_clusters)
        if cached is not None:
            print("Loaded partition from cache: %s" % os.path.basename(cache_path))
            return _groups_to_part_adj_and_parts(adj, idx_arr, cached, num_clusters)

    train_adj_lil = adj[idx_arr, :][:, idx_arr].tolil()
    train_adj_lists = [[] for _ in range(num_nodes)]
    for i in range(num_nodes):
        rows = train_adj_lil[i].rows[0]
        if i in rows:
            rows.remove(i)
        train_adj_lists[i] = rows

    if num_clusters > 1:
        _, groups = metis.part_graph(train_adj_lists, num_clusters, seed=42)
    else:
        groups = [0] * num_nodes

    groups_arr = np.asarray(groups, dtype=np.int32)
    if cache_path:
        partition_cache.save_partition_cache(
            cache_path, adj, idx_arr, groups_arr, num_all_nodes, num_clusters)

    return _groups_to_part_adj_and_parts(adj, idx_arr, groups_arr, num_clusters)
