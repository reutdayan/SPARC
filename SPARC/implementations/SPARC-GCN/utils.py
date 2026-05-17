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

"""Collections of preprocessing functions for different graph formats."""

import json
import os
import time

from networkx.readwrite import json_graph
import numpy as np
import partition_utils
import scipy.sparse as sp
from scipy.sparse import load_npz
import sklearn.metrics
import sklearn.preprocessing
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1 import gfile


def parse_index_file(filename):
  """Parse index file."""
  index = []
  for line in gfile.Open(filename):
    index.append(int(line.strip()))
  return index


def sample_mask(idx, l):
  """Create mask."""
  mask = np.zeros(l)
  mask[idx] = 1
  return np.array(mask, dtype=bool)


def sym_normalize_adj(adj):
  """Normalization by D^{-1/2} (A+I) D^{-1/2}."""
  adj = adj + sp.eye(adj.shape[0])
  rowsum = np.array(adj.sum(1)) + 1e-20
  d_inv_sqrt = np.power(rowsum, -0.5).flatten()
  d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
  d_mat_inv_sqrt = sp.diags(d_inv_sqrt, 0)
  adj = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
  return adj


def normalize_adj(adj):
  rowsum = np.array(adj.sum(1)).flatten()
  d_inv = 1.0 / (np.maximum(1.0, rowsum))
  d_mat_inv = sp.diags(d_inv, 0)
  adj = d_mat_inv.dot(adj)
  return adj


def normalize_adj_diag_enhance(adj, diag_lambda):
  """Normalization by  A'=(D+I)^{-1}(A+I), A'=A'+lambda*diag(A')."""
  adj = adj + sp.eye(adj.shape[0])
  rowsum = np.array(adj.sum(1)).flatten()
  d_inv = 1.0 / (rowsum + 1e-20)
  d_mat_inv = sp.diags(d_inv, 0)
  adj = d_mat_inv.dot(adj)
  adj = adj + diag_lambda * sp.diags(adj.diagonal(), 0)
  return adj


def sparse_to_tuple(sparse_mx):
  """Convert sparse matrix to tuple representation."""

  def to_tuple(mx):
    if not sp.isspmatrix_coo(mx):
      mx = mx.tocoo()
    coords = np.vstack((mx.row, mx.col)).transpose()
    values = mx.data
    shape = mx.shape
    return coords, values, shape

  if isinstance(sparse_mx, list):
    for i in range(len(sparse_mx)):
      sparse_mx[i] = to_tuple(sparse_mx[i])
  else:
    sparse_mx = to_tuple(sparse_mx)

  return sparse_mx


def calc_f1(y_pred, y_true, multilabel):
  if multilabel:
    y_pred[y_pred > 0] = 1
    y_pred[y_pred <= 0] = 0
  else:
    y_true = np.argmax(y_true, axis=1)
    y_pred = np.argmax(y_pred, axis=1)
  return sklearn.metrics.f1_score(
      y_true, y_pred, average='micro'), sklearn.metrics.f1_score(
          y_true, y_pred, average='macro')


def construct_feed_dict(features,eigen_vecs, support, labels, labels_mask, placeholders):
  """Construct feed dictionary."""
  feed_dict = dict()
  feed_dict.update({placeholders['labels']: labels})
  feed_dict.update({placeholders['labels_mask']: labels_mask})
  feed_dict.update({placeholders['features']: features})
  feed_dict.update({placeholders['eigen_vecs']: eigen_vecs})
  feed_dict.update({placeholders['support']: support})
  # ``features`` can be either a sparse tuple (coords, values, shape) or a
  # dense ndarray (SPARC-only/random-batch path used for pubmed).
  if isinstance(features, (tuple, list)) and len(features) > 1:
    num_features_nonzero = np.asarray(features[1].shape, dtype=np.int32)
  else:
    num_features_nonzero = np.asarray(features.shape, dtype=np.int32)
  feed_dict.update({placeholders['num_features_nonzero']: num_features_nonzero})
  return feed_dict


def preprocess_multicluster(adj,
                            parts,
                            features,
                            eigen_vecs,
                            y_train,
                            train_mask,
                            num_clusters,
                            block_size,
                            diag_lambda=-1):
  """Generate the batch for multiple clusters."""

  print('features shape: ' + str(features.shape))
  print('eigen_vecs shape: ' + str(eigen_vecs.shape))
  features_batches = []
  eigen_vecs_batches = []
  support_batches = []
  y_train_batches = []
  train_mask_batches = []
  total_nnz = 0
  np.random.shuffle(parts)
  for _, st in enumerate(range(0, num_clusters, block_size)):
    pt = parts[st]
    for pt_idx in range(st + 1, min(st + block_size, num_clusters)):
      pt = np.concatenate((pt, parts[pt_idx]), axis=0)
    features_batches.append(features[pt, :])
    eigen_vecs_batches.append(eigen_vecs[pt, :])
    y_train_batches.append(y_train[pt, :])
    support_now = adj[pt, :][:, pt]
    if diag_lambda == -1:
      support_batches.append(sparse_to_tuple(normalize_adj(support_now)))
    else:
      support_batches.append(
          sparse_to_tuple(normalize_adj_diag_enhance(support_now, diag_lambda)))
    total_nnz += support_now.count_nonzero()

    train_pt = []
    for newidx, idx in enumerate(pt):
      if train_mask[idx]:
        train_pt.append(newidx)
    train_mask_batches.append(sample_mask(train_pt, len(pt)))
  return (features_batches, eigen_vecs_batches, support_batches, y_train_batches,
          train_mask_batches)


def preprocess(adj,
               features,
               eigen_vecs,
               y_train,
               train_mask,
               visible_data,
               num_clusters,
               diag_lambda=-1):
  """Do graph partitioning and preprocessing for SGD training."""

  # Do graph partitioning
  part_adj, parts = partition_utils.partition_graph(adj, visible_data,
                                                    num_clusters)
  if diag_lambda == -1:
    part_adj = normalize_adj(part_adj)
  else:
    part_adj = normalize_adj_diag_enhance(part_adj, diag_lambda)
  parts = [np.array(pt) for pt in parts]

  features_batches = []
  eigen_vecs_batches = []
  support_batches = []
  y_train_batches = []
  train_mask_batches = []
  total_nnz = 0
  for pt in parts:
    features_batches.append(features[pt, :])
    eigen_vecs_batches.append(eigen_vecs[pt, :])
    now_part = part_adj[pt, :][:, pt]
    total_nnz += now_part.count_nonzero()
    support_batches.append(sparse_to_tuple(now_part))
    y_train_batches.append(y_train[pt, :])

    train_pt = []
    for newidx, idx in enumerate(pt):
      if train_mask[idx]:
        train_pt.append(newidx)
    train_mask_batches.append(sample_mask(train_pt, len(pt)))
  return (parts, features_batches, eigen_vecs_batches, support_batches, y_train_batches,
          train_mask_batches)


def format_test_ratio(test_ratio):
  """Match SPARC ``main.py`` run-dir formatting (e.g. 0.1, 0.03)."""
  return '{}'.format(float(test_ratio))


def list_sparc_run_names(sparc_results_root, dataset_str):
  """Return sorted run subdir names under ``<root>/<dataset>/`` that contain embeddings."""
  base = os.path.normpath(os.path.join(sparc_results_root, dataset_str))
  if not os.path.isdir(base):
    return []
  names = []
  for name in sorted(os.listdir(base)):
    sub = os.path.join(base, name)
    if os.path.isdir(sub) and os.path.isfile(os.path.join(sub, 'embeddings.npy')):
      names.append(name)
  return names


def resolve_sparc_run_dir(sparc_results_root, dataset_str, sparc_run=None,
                          split_name='random', test_ratio=0.03, seed=42):
  """Resolve the SPARC run directory for ``main.py`` artifacts.

  Resolution order:

    1. ``<root>/<dataset>/<sparc_run>/`` when ``sparc_run`` is non-empty.
    2. ``<root>/<dataset>/<split>_test<ratio>_seed<seed>/`` (same naming as ``main.py``).
    3. Flat ``<root>/<dataset>/`` if ``embeddings.npy`` lives there directly.
    4. Auto-pick the sole subdirectory that contains ``embeddings.npy``.

  Args:
    sparc_results_root: Root ``sparc_results`` directory.
    dataset_str: Dataset name (e.g. ``cora``).
    sparc_run: Optional explicit run folder name.
    split_name: Split strategy (used when ``sparc_run`` is empty).
    test_ratio: Test fraction (used when ``sparc_run`` is empty).
    seed: Random seed (used when ``sparc_run`` is empty).

  Returns:
    Absolute path to the run directory.
  """
  base = os.path.normpath(os.path.join(sparc_results_root, dataset_str))
  available = list_sparc_run_names(sparc_results_root, dataset_str)

  def _fail(tried_paths, hint=''):
    msg = ['SPARC run directory not found.']
    for p in tried_paths:
      msg.append('  Tried: {}'.format(p))
    if available:
      msg.append('  Available runs under {}: {}'.format(base, available))
    else:
      msg.append('  No run subdirectories with embeddings.npy under {}.'.format(base))
    if hint:
      msg.append('  {}'.format(hint))
    raise FileNotFoundError('\n'.join(msg))

  tried = []
  if sparc_run:
    run_dir = os.path.join(base, sparc_run)
    tried.append(run_dir)
    if os.path.isdir(run_dir):
      return os.path.abspath(run_dir)
    hint = (
        'Pass --sparc_run <name> matching a folder under sparc_results, or '
        're-run SPARC main.py with matching --test_ratio / --seed.')
    if len(available) == 1:
      hint += ' Closest match: --sparc_run {}'.format(available[0])
    _fail(tried, hint=hint)

  run_name = '{}_test{}_seed{}'.format(
      split_name, format_test_ratio(test_ratio), seed)
  run_dir = os.path.join(base, run_name)
  tried.append(run_dir)
  if os.path.isdir(run_dir):
    return os.path.abspath(run_dir)

  flat_emb = os.path.join(base, 'embeddings.npy')
  if os.path.isfile(flat_emb):
    return os.path.abspath(base)

  if len(available) == 1:
    picked = os.path.join(base, available[0])
    tf.logging.info(
        'Auto-selected sole SPARC run: %s (expected %s was missing).',
        available[0], run_name)
    return os.path.abspath(picked)

  if not os.path.isdir(base):
    raise FileNotFoundError(
        'SPARC dataset directory missing: {}'.format(base))
  if not available:
    _fail(tried, hint=(
        'Run SPARC main.py first, or pass --sparc_run <subdir>.'))
  raise ValueError(
      'Multiple SPARC runs under {}: {}\n'
      'Pass --sparc_run to choose one.'.format(base, available))


def resolve_sparc_embeddings_path(sparc_results_root, dataset_str, sparc_run=None,
                                  split_name='random', test_ratio=0.03, seed=42):
  """Resolve embeddings.npy under ``sparc_results_root/<dataset>/``.

  See ``resolve_sparc_run_dir`` for directory resolution rules.

  Args:
    sparc_results_root: Root ``sparc_results`` directory (absolute path).
    dataset_str: Dataset name (e.g. ``cora``).
    sparc_run: Optional run folder name under ``<dataset>/``.
    split_name: Split strategy when ``sparc_run`` is empty.
    test_ratio: Test fraction when ``sparc_run`` is empty.
    seed: Random seed when ``sparc_run`` is empty.

  Returns:
    Absolute path to ``embeddings.npy``.
  """
  run_dir = resolve_sparc_run_dir(
      sparc_results_root, dataset_str, sparc_run=sparc_run,
      split_name=split_name, test_ratio=test_ratio, seed=seed)
  emb = os.path.join(run_dir, 'embeddings.npy')
  if not os.path.isfile(emb):
    raise FileNotFoundError(
        'SPARC embeddings not found at {} (check run dir contents).'.format(emb))
  tf.logging.info('Using SPARC embeddings: %s', emb)
  return os.path.abspath(emb)


def try_load_sparc_split_indices(embeddings_path, num_data):
  """Load train/val/test node indices from SPARC ``main.py`` artifacts next to embeddings.

  When ``embeddings_path`` is ``.../<run>/embeddings.npy``, SPARC saves boolean masks in the
  same ``<run>/`` directory. If all three exist and length matches ``num_data``, returns
  ``(train_data, val_data, test_data)`` as int32 index arrays; otherwise ``None``.

  Args:
    embeddings_path: Resolved path to ``embeddings.npy``.
    num_data: Expected number of graph nodes (dense indices).

  Returns:
    Tuple ``(train_data, val_data, test_data)`` or ``None``.
  """
  run_dir = os.path.dirname(os.path.abspath(embeddings_path))
  paths = [
      os.path.join(run_dir, name)
      for name in ('train_mask.npy', 'val_mask.npy', 'test_mask.npy')
  ]
  if not all(os.path.isfile(p) for p in paths):
    return None

  def _load_bool_mask(path):
    return np.asarray(np.load(path)).astype(bool).reshape(-1)

  train_m = _load_bool_mask(paths[0])
  val_m = _load_bool_mask(paths[1])
  test_m = _load_bool_mask(paths[2])
  if train_m.shape[0] != num_data or val_m.shape[0] != num_data or test_m.shape[
      0] != num_data:
    tf.logging.warning(
        'SPARC mask length does not match num_data (%d); ignoring masks.', num_data)
    return None

  train_data = np.where(train_m)[0].astype(np.int32)
  val_data = np.where(val_m)[0].astype(np.int32)
  test_data = np.where(test_m)[0].astype(np.int32)
  return train_data, val_data, test_data


def load_sparc_run_bundle(sparc_results_root, dataset_str, sparc_run=None,
                          split_name='random', test_ratio=0.03, seed=42):
  """Load tensors saved by SPARC ``main.py`` without reading GraphSAGE JSON/feats.

  Expected files next to ``embeddings.npy`` in the resolved run directory:

    * ``features.npy``, ``labels.npy`` (class IDs ``(N,)`` or one-hot ``(N, C)``)
    * ``train_mask.npy``, ``val_mask.npy``, ``test_mask.npy``
    * ``train_adj.npz``, ``full_adj.npz`` (optional; identity adjacency if absent)

  Args:
    sparc_results_root: Root ``sparc_results`` directory.
    dataset_str: Dataset name (subfolder under root).
    sparc_run: Run subdirectory name (same rules as ``resolve_sparc_embeddings_path``).
    split_name: Split strategy when ``sparc_run`` is empty.
    test_ratio: Test fraction when ``sparc_run`` is empty.
    seed: Random seed when ``sparc_run`` is empty.

  Returns:
    Same tuple shape as ``load_graphsage_data``:
    ``(num_data, train_adj, full_adj, feats, train_feats, test_feats, labels,
      train_data, val_data, test_data, train_eigen_vecs, test_eigen_vecs)``
  """
  start_time = time.time()
  if sparc_results_root is None:
    sparc_results_root = os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     '..', '..', 'sparc_results'))
  embeddings_path = resolve_sparc_embeddings_path(
      sparc_results_root, dataset_str, sparc_run=sparc_run,
      split_name=split_name, test_ratio=test_ratio, seed=seed)
  run_dir = os.path.dirname(os.path.abspath(embeddings_path))

  def _require(name):
    path = os.path.join(run_dir, name)
    if not os.path.isfile(path):
      raise FileNotFoundError(
          'SPARC-only mode requires {} under {}'.format(name, run_dir))
    return path

  feat_path = _require('features.npy')
  lab_path = _require('labels.npy')
  tm_path = _require('train_mask.npy')
  vm_path = _require('val_mask.npy')
  testm_path = _require('test_mask.npy')

  feats = np.asarray(np.load(feat_path), dtype=np.float32)
  labels_raw = np.asarray(np.load(lab_path))
  eigen_vecs = np.load(gfile.Open(embeddings_path, 'rb')).astype(np.float32)

  train_m = np.asarray(np.load(tm_path)).astype(bool).reshape(-1)
  val_m = np.asarray(np.load(vm_path)).astype(bool).reshape(-1)
  test_m = np.asarray(np.load(testm_path)).astype(bool).reshape(-1)

  num_data = int(feats.shape[0])
  # SPARC main.py saves labels = np.argmax(y_test, axis=1) (1d class ids); GCN expects one-hot.
  if labels_raw.ndim == 1:
    labs = labels_raw.astype(np.int64).reshape(-1)
    if labs.shape[0] != num_data:
      raise ValueError('labels length {} != num_data {}'.format(
          labs.shape[0], num_data))
    num_classes = int(labs.max()) + 1
    labels = np.zeros((num_data, num_classes), dtype=np.float32)
    labels[np.arange(num_data, dtype=np.int64), labs] = 1.0
  elif labels_raw.ndim == 2:
    labels = labels_raw.astype(np.float32)
    if labels.shape[0] != num_data:
      raise ValueError('labels rows {} != num_data {}'.format(
          labels.shape[0], num_data))
  else:
    raise ValueError(
        'labels.npy must be 1d class indices or 2d one-hot; got shape {}'.format(
            labels_raw.shape))

  if eigen_vecs.shape[0] != num_data:
    raise ValueError(
        'features/embeddings row mismatch: feats {}, eigen {}'.format(
            feats.shape[0], eigen_vecs.shape[0]))
  if train_m.shape[0] != num_data:
    raise ValueError('train_mask length {} != num_data {}'.format(
        train_m.shape[0], num_data))

  train_adj_path = os.path.join(run_dir, 'train_adj.npz')
  full_adj_path = os.path.join(run_dir, 'full_adj.npz')
  eye = sp.eye(num_data, dtype=np.float32, format='csr')
  if os.path.isfile(train_adj_path):
    train_adj = load_npz(train_adj_path).tocsr().astype(np.float32)
    if train_adj.shape != (num_data, num_data):
      raise ValueError('train_adj shape {} expected ({}, {})'.format(
          train_adj.shape, num_data, num_data))
  else:
    tf.logging.warning(
        'train_adj.npz missing under %s; using identity (no graph edges).', run_dir)
    train_adj = eye.copy()

  if os.path.isfile(full_adj_path):
    full_adj = load_npz(full_adj_path).tocsr().astype(np.float32)
    if full_adj.shape != (num_data, num_data):
      raise ValueError('full_adj shape {} expected ({}, {})'.format(
          full_adj.shape, num_data, num_data))
  else:
    tf.logging.warning(
        'full_adj.npz missing under %s; using identity (no graph edges).', run_dir)
    full_adj = eye.copy()

  train_data = np.where(train_m)[0].astype(np.int32)
  val_data = np.where(val_m)[0].astype(np.int32)
  test_data = np.where(test_m)[0].astype(np.int32)

  tf.logging.info(
      'Loaded SPARC run bundle from %s (%d nodes; train=%d val=%d test=%d) in %f s.',
      run_dir, num_data, len(train_data), len(val_data), len(test_data),
      time.time() - start_time)

  train_feats = feats
  test_feats = feats
  train_eigen_vecs = eigen_vecs
  test_eigen_vecs = eigen_vecs
  return (num_data, train_adj, full_adj, feats, train_feats, test_feats, labels,
          train_data, val_data, test_data, train_eigen_vecs, test_eigen_vecs)


def preprocess_random_batches(adj,
                              features,
                              eigen_vecs,
                              y_labels,
                              node_indices,
                              batch_size,
                              diag_lambda=-1,
                              rng=None):
  """Minibatches by random shards of ``node_indices``; local subgraph adjacency per batch.

  Avoids METIS/graph partitioning. Each batch uses ``adj[pt,:][:, pt]`` only (edges among
  nodes in the minibatch).
  """
  if rng is None:
    rng = np.random
  node_indices = np.asarray(node_indices, dtype=np.int64).reshape(-1)
  if node_indices.size == 0:
    return [], [], [], [], [], []

  rng.shuffle(node_indices)
  bs = max(int(batch_size), 1)
  parts = []
  features_batches = []
  eigen_vecs_batches = []
  support_batches = []
  y_batches = []
  mask_batches = []

  for start in range(0, len(node_indices), bs):
    pt = node_indices[start:start + bs]
    if pt.size == 0:
      continue
    parts.append(pt)
    features_batches.append(features[pt, :])
    eigen_vecs_batches.append(eigen_vecs[pt, :])
    y_batches.append(y_labels[pt, :])
    support_now = adj[pt, :][:, pt]
    if diag_lambda == -1:
      support_batches.append(sparse_to_tuple(normalize_adj(support_now)))
    else:
      support_batches.append(
          sparse_to_tuple(normalize_adj_diag_enhance(support_now, diag_lambda)))
    mask_batches.append(np.ones(len(pt), dtype=bool))

  return (parts, features_batches, eigen_vecs_batches, support_batches, y_batches,
          mask_batches)


def load_graphsage_data(dataset_path, dataset_str, normalize=True,
                       sparc_results_root=None, sparc_run=None,
                       split_name='random', test_ratio=0.03, seed=42):
  """Load GraphSAGE data."""
  start_time = time.time()

  if sparc_results_root is None:
    sparc_results_root = os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     '..', '..', 'sparc_results'))
  embeddings_path = resolve_sparc_embeddings_path(
      sparc_results_root, dataset_str, sparc_run=sparc_run,
      split_name=split_name, test_ratio=test_ratio, seed=seed)

  graph_json = json.load(
      gfile.Open('{}/{}/{}-G.json'.format(dataset_path, dataset_str,
                                          dataset_str)))
  graph_nx = json_graph.node_link_graph(graph_json)

  # print(graph_nx.nodes['2hk2a7'])
  # print(graph_nx.nodes[13])

  id_map = json.load(
      gfile.Open('{}/{}/{}-id_map.json'.format(dataset_path, dataset_str,
                                               dataset_str)))
  is_digit = list(id_map.keys())[0].isdigit()
  # id_map = {(int(k) if is_digit else k): int(v) for k, v in id_map.items()}
  id_map = {k: int(v) for k, v in id_map.items()}
  id_map_re = {value: key for key, value in id_map.items()}
  class_map = json.load(
      gfile.Open('{}/{}/{}-class_map.json'.format(dataset_path, dataset_str,
                                                  dataset_str)))

  is_instance = isinstance(list(class_map.values())[0], list)
  
  # class_map = {(int(k) if is_digit else k): (v if is_instance else int(v))
  #              for k, v in class_map.items()}
  # class_map = {k: (v if is_instance else int(v)) 
  #              for k, v in class_map.items()}

  # co = 0
  # for node in graph_nx.nodes():
  #   if 'val' in graph_nx.nodes[node]:
  #     co += 1
  # print(co)

  print(len(graph_nx.nodes()))
  # broken_count = 0
  # to_remove = []
  # for node in graph_nx.nodes():
  #   if node not in id_map and node not in id_map_re:
  #     to_remove.append(node)
  #     broken_count += 1
  # for node in to_remove:
  #   graph_nx.remove_node(node)
  # tf.logging.info(
  #     'Removed %d nodes that lacked proper annotations due to networkx versioning issues',
  #     broken_count)

  feats = np.load(
      gfile.Open(
          '{}/{}/{}-feats.npy'.format(dataset_path, dataset_str, dataset_str),
          'rb')).astype(np.float32)
  
  
  eigen_vecs = np.load(gfile.Open(embeddings_path, 'rb')).astype(np.float32)
  

  
  print('eigem_vecs shape: ' + str(eigen_vecs.shape))
  print('feats shape: ' + str(feats.shape))

  tf.logging.info('Loaded data (%f seconds).. now preprocessing..',
                  time.time() - start_time)
  start_time = time.time()

  print('num of nodes: ' + str(len(graph_nx.nodes())))

  # co = 0
  # for node in graph_nx.nodes():
  #   if 'val' in graph_nx.nodes[node]:
  #     co += 1
  # print(co)

  # co = 0
  # for node in graph_nx.nodes():
  #   print( graph_nx.nodes[node])
  #   if co == 10:
  #     break
  #   co += 1

  edges = []
  
  for edge in graph_nx.edges():
    if edge[0] in id_map_re and edge[1] in id_map_re:
      edges.append((edge[0], edge[1]))
  num_data = len(id_map)
  
  print('num of edges: ' + str(len(graph_nx.edges())))
  
  # print(graph_nx.nodes['2hk2a7'])
  # print(graph_nx.nodes[13])
  sparc_splits = try_load_sparc_split_indices(embeddings_path, num_data)
  if sparc_splits is not None:
    train_data, val_data, test_data = sparc_splits
    tf.logging.info(
        'Using train/val/test splits from SPARC run dir (same folder as embeddings): '
        'train=%d, val=%d, test=%d.',
        len(train_data), len(val_data), len(test_data))
    is_train = np.zeros((num_data), dtype=bool)
    is_train[train_data] = True
  else:
    val_data = np.array(
        [id_map[n] for n in graph_nx.nodes()
         if 'val' in graph_nx.nodes[n] and graph_nx.nodes[n]['val']],
        dtype=np.int32)
    print('done val')
    print(len(val_data))
    test_data = np.array(
        [id_map[n] for n in graph_nx.nodes()
         if 'test' in graph_nx.nodes[n] and graph_nx.nodes[n]['test']],
        dtype=np.int32)
    is_train = np.ones((num_data), dtype=bool)
    is_train[val_data] = False
    is_train[test_data] = False
    train_data = np.array([n for n in range(num_data) if is_train[n]],
                          dtype=np.int32)

  train_edges = [
      (e[0], e[1]) for e in edges if is_train[e[0]] and is_train[e[1]]
  ]
  edges = np.array(edges, dtype=np.int32)
  train_edges = np.array(train_edges, dtype=np.int32)

  print('edges', len(edges))
  print('train_edges', len(train_edges))


  # Process labels
  if isinstance(list(class_map.values())[0], list):
    num_classes = len(list(class_map.values())[0])
    labels = np.zeros((num_data, num_classes), dtype=np.float32)
    for k in class_map.keys():
      labels[id_map[k], :] = np.array(class_map[k])
  else:
    num_classes = len(set(class_map.values()))
    labels = np.zeros((num_data, num_classes), dtype=np.float32)
    for k in class_map.keys():
      labels[id_map[k], class_map[k]] = 1

  if normalize:
    if sparc_splits is not None:
      train_ids = train_data
    else:
      train_ids = np.array([
          id_map[n]
          for n in graph_nx.nodes()
          if 'val' in graph_nx.nodes[n] and
          not graph_nx.nodes[n]['val'] and not graph_nx.nodes[n]['test']
      ])
    train_feats = feats[train_ids]
    scaler = sklearn.preprocessing.StandardScaler()
    scaler.fit(train_feats)
    feats = scaler.transform(feats)

  def _construct_adj(edges):
    print(edges.shape)
    adj = sp.csr_matrix((np.ones(
        (edges.shape[0]), dtype=np.float32), (edges[:, 0], edges[:, 1])),
                        shape=(num_data, num_data))
    adj += adj.transpose()
    return adj

  train_adj = _construct_adj(train_edges)
  full_adj = _construct_adj(edges)
  

  train_feats = feats
  test_feats = feats
  
  train_eigen_vecs = eigen_vecs
  test_eigen_vecs = eigen_vecs

  tf.logging.info('Data loaded, %f seconds.', time.time() - start_time)
  return num_data, train_adj, full_adj, feats, train_feats, test_feats, labels, train_data, val_data, test_data, train_eigen_vecs, test_eigen_vecs