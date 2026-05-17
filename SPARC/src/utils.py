import json
import os
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import scipy.sparse as sp
import sklearn
from networkx.readwrite import json_graph
from scipy.sparse.csgraph import connected_components
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors
import partition_utils


def sample_mask(idx, l):
  """Create mask."""
  mask = np.zeros(l)
  mask[idx] = 1
  return np.array(mask, dtype=bool)


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


def to_tuple(mx):
    if not sp.isspmatrix_coo(mx):
        mx = mx.tocoo()
    coords = np.vstack((mx.row, mx.col)).transpose()
    values = mx.data
    shape = mx.shape
    return coords, values, shape


def sparse_to_tuple(sparse_mx):
    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def build_feature_knn_adj(features: np.ndarray,
                          k: int,
                          metric: str = "cosine",
                          mutual: bool = True) -> sp.csr_matrix:
    """Build a symmetric binary kNN adjacency from feature similarity.

    Parameters
    ----------
    features:
        (n, d) dense feature matrix. Rows are nodes, columns are features.
        The features are used *as given*; apply whatever normalization you want
        upstream (e.g. StandardScaler) before calling this.
    k:
        Number of feature neighbors to link to. ``k`` does *not* include the
        self-neighbor.
    metric:
        Any metric accepted by ``sklearn.neighbors.NearestNeighbors``
        (``"cosine"``, ``"euclidean"``, ...). Cosine is the usual choice for
        TF-IDF-style features.
    mutual:
        If True (default), keep an edge ``(i, j)`` only when ``j`` is in
        ``kNN(i)`` *and* ``i`` is in ``kNN(j)``. This is the standard
        ``mutual-kNN`` graph used in spectral-clustering-on-features pipelines;
        it strongly suppresses false edges to outliers. If False, return the
        union ``kNN(i) ∪ kNN(j)`` instead.

    Returns
    -------
    scipy.sparse.csr_matrix
        (n, n) symmetric 0/1 matrix with zeros on the diagonal. Useful as an
        augmentation to a structural adjacency via elementwise max.
    """
    n = features.shape[0]
    k_eff = int(max(1, min(k, n - 1)))
    nn = NearestNeighbors(n_neighbors=k_eff + 1, metric=metric)
    nn.fit(features)
    _, idx = nn.kneighbors(features)
    # Drop the self-neighbor in column 0.
    rows = np.repeat(np.arange(n, dtype=np.int64), k_eff)
    cols = idx[:, 1:].reshape(-1).astype(np.int64)
    data = np.ones(rows.shape[0], dtype=np.float32)
    A = sp.csr_matrix((data, (rows, cols)), shape=(n, n))
    if mutual:
        # Elementwise product of directed kNN and its transpose keeps exactly
        # the entries where both i->j and j->i are present.
        A = A.multiply(A.T)
    else:
        A = A.maximum(A.T)
    A = A.tocsr()
    A.setdiag(0.0)
    A.eliminate_zeros()
    # Re-binarize (mutual via product can leave 1.0 but keep it defensive).
    if A.nnz:
        A.data = np.ones_like(A.data, dtype=np.float32)
    return A


def build_component_bridge_adj(features: np.ndarray,
                               adj: sp.spmatrix,
                               metric: str = "cosine",
                               restrict_idx: np.ndarray = None,
                               initial_k: int = None,
                               max_k: int = None,
                               verbose: bool = False
                               ) -> "tuple[sp.csr_matrix, int, int]":
    """Add the *minimum* feature-similarity edges to make ``adj`` connected.

    Computes the connected components of ``adj`` (ignoring its diagonal /
    self-loops) and adds exactly ``n_components - 1`` undirected edges --
    the provable minimum to merge everything into a single component. The
    bridging edges are picked greedily as a Minimum Spanning Tree over the
    components, where the weight between two components is the smallest
    feature distance (under ``metric``) over all cross-component node-pairs.

    The returned matrix is binary (0/1) and symmetric, so it composes with
    a binary structural adjacency via elementwise ``maximum`` exactly the
    way the existing kNN augmentation does -- bridging edges carry the
    same weight as real structural edges.

    Parameters
    ----------
    features:
        ``(n, d)`` dense feature matrix. Must align row-wise with ``adj``.
    adj:
        ``(n, n)`` sparse adjacency. Self-loops are ignored when computing
        components. Edge weights are not used (only the sparsity pattern).
    metric:
        Any metric accepted by ``sklearn.neighbors.NearestNeighbors`` /
        ``scipy.spatial.distance.cdist``. Cosine is the usual choice for
        text-style features; euclidean for dense embeddings.
    restrict_idx:
        Optional 1-D array of node indices. When given, components are
        computed on the induced subgraph ``adj[restrict_idx][:, restrict_idx]``
        and bridging edges are restricted to node-pairs both inside
        ``restrict_idx``. Use this to keep inductive splits honest -- e.g.
        when bridging the train graph, pass ``idx_train`` so val/test
        nodes (which are isolated singletons in ``train_adj``) are never
        linked into the train component graph.
    initial_k, max_k:
        Lower / upper bounds on the kNN query size used to generate
        bridge-edge candidates. Defaults: ``initial_k = max(32, 2 * n_components)``,
        ``max_k = m - 1`` where ``m`` is the working node count. ``k`` is
        doubled until the candidate set's MST connects every component;
        any still-unbridged component pairs are merged via an exhaustive
        ``cdist`` fallback.

    Returns
    -------
    bridge_adj : ``scipy.sparse.csr_matrix``
        ``(n, n)`` binary symmetric matrix containing only the bridging
        edges. nnz equals ``2 * (n_components - 1)`` (each undirected edge
        stored as two directed entries).
    n_components_before : int
        Connected-component count of the input graph (after restriction,
        if any).
    n_bridges_added : int
        Number of undirected edges added (= ``n_components_before - 1``
        when bridging succeeds; less if the input is already connected).
    """
    n_total = features.shape[0]
    A_full = adj.tocsr()
    if A_full.shape[0] != n_total:
        raise ValueError(
            f"adj shape {A_full.shape} does not match features rows {n_total}")

    if restrict_idx is None:
        sub_idx = np.arange(n_total, dtype=np.int64)
    else:
        sub_idx = np.asarray(restrict_idx, dtype=np.int64).ravel()
    m = sub_idx.shape[0]

    # Work on the induced subgraph so val/test isolated nodes don't pollute
    # the component count when bridging the train adjacency.
    A_sub = A_full[sub_idx][:, sub_idx].tocsr().copy()
    A_sub.setdiag(0)
    A_sub.eliminate_zeros()

    n_comp, comp_labels = connected_components(A_sub, directed=False)
    if verbose:
        print(f"[bridge] working over m={m} nodes, found "
              f"{n_comp} connected components")
    if n_comp <= 1:
        return sp.csr_matrix((n_total, n_total), dtype=np.float32), n_comp, 0

    feats_sub = np.ascontiguousarray(features[sub_idx])

    # ------------------------------------------------------------------
    # Stage 1: kNN-based candidate generation.
    # For each node i in the subgraph, find its closest neighbor in a
    # *different* component. This yields up to m candidate cross-component
    # edges. Sort them and run union-find Kruskal over component IDs.
    # ------------------------------------------------------------------
    if initial_k is None:
        initial_k = min(m - 1, max(32, 2 * n_comp))
    if max_k is None:
        max_k = m - 1
    k = max(1, min(initial_k, m - 1))

    # union-find over component IDs (0 .. n_comp - 1)
    parent = np.arange(n_comp, dtype=np.int64)

    def _find(x: int) -> int:
        # iterative path-compression
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    bridge_rows: list = []
    bridge_cols: list = []
    bridges_added = 0

    while bridges_added < n_comp - 1:
        nn = NearestNeighbors(n_neighbors=min(k + 1, m), metric=metric)
        nn.fit(feats_sub)
        dists, idx = nn.kneighbors(feats_sub)

        candidates = []  # (dist, sub_i, sub_j)
        for i in range(m):
            ci = comp_labels[i]
            for col in range(1, idx.shape[1]):
                j = int(idx[i, col])
                if comp_labels[j] != ci:
                    candidates.append((float(dists[i, col]), i, j))
                    break

        if candidates:
            candidates.sort(key=lambda t: t[0])
            for _d, i, j in candidates:
                ri = _find(int(comp_labels[i]))
                rj = _find(int(comp_labels[j]))
                if ri == rj:
                    continue
                parent[ri] = rj
                bridge_rows.append(int(sub_idx[i]))
                bridge_cols.append(int(sub_idx[j]))
                bridges_added += 1
                if bridges_added == n_comp - 1:
                    break

        if bridges_added == n_comp - 1 or k >= max_k:
            break
        k = min(max_k, k * 2)
        if verbose:
            print(f"[bridge] expanding kNN to k={k} "
                  f"(bridges so far: {bridges_added}/{n_comp - 1})")

    # ------------------------------------------------------------------
    # Stage 2: brute-force fallback for any still-unbridged component
    # pairs. Group nodes by their MST-cluster (the union-find root of
    # their original component) and merge the closest pair until one
    # cluster remains. Uses ``cdist`` over the pair of node sets so the
    # cost is proportional to |C_a| * |C_b|; rare in practice.
    # ------------------------------------------------------------------
    if bridges_added < n_comp - 1:
        roots = np.array([_find(int(c)) for c in range(n_comp)])
        cluster_nodes: dict = {}
        for local_idx in range(m):
            r = int(roots[comp_labels[local_idx]])
            cluster_nodes.setdefault(r, []).append(local_idx)
        clusters = [np.asarray(v, dtype=np.int64)
                    for v in cluster_nodes.values()]
        if verbose and len(clusters) > 1:
            print(f"[bridge] kNN candidates left {len(clusters)} clusters; "
                  f"finishing with cdist fallback")

        while len(clusters) > 1:
            best = None
            best_pair = (0, 1)
            for a in range(len(clusters)):
                for b in range(a + 1, len(clusters)):
                    Xa = feats_sub[clusters[a]]
                    Xb = feats_sub[clusters[b]]
                    D = cdist(Xa, Xb, metric=metric)
                    flat = int(np.argmin(D))
                    ra, cb = np.unravel_index(flat, D.shape)
                    d = float(D[ra, cb])
                    if best is None or d < best[0]:
                        best = (d,
                                int(clusters[a][ra]),
                                int(clusters[b][cb]))
                        best_pair = (a, b)

            if best is None:
                break
            _d, i_local, j_local = best
            bridge_rows.append(int(sub_idx[i_local]))
            bridge_cols.append(int(sub_idx[j_local]))
            bridges_added += 1
            a, b = best_pair
            merged = np.concatenate([clusters[a], clusters[b]])
            clusters = ([clusters[i] for i in range(len(clusters))
                         if i != a and i != b]
                        + [merged])

    # Symmetrize: store each undirected edge as two directed entries.
    rows = np.array(bridge_rows + bridge_cols, dtype=np.int64)
    cols = np.array(bridge_cols + bridge_rows, dtype=np.int64)
    data = np.ones(rows.shape[0], dtype=np.float32)
    bridge_adj = sp.csr_matrix(
        (data, (rows, cols)), shape=(n_total, n_total), dtype=np.float32)
    bridge_adj.sum_duplicates()
    if bridge_adj.nnz:
        bridge_adj.data = np.ones_like(bridge_adj.data, dtype=np.float32)
    return bridge_adj, int(n_comp), int(bridges_added)


def preprocess_multicluster(adj,
                            parts,
                            features,
                            y_train,
                            train_mask,
                            num_clusters,
                            block_size,
                            diag_lambda=-1,
                            normalize=True):
    """Generate the batch for multiple clusters.

    When ``normalize=False`` the raw symmetric submatrix ``adj[pt][:, pt]``
    is passed through without row-stochastic / diag-enhance normalization.
    This is what the spectral loss expects: it needs a *symmetric* affinity
    so that the Rayleigh-quotient objective targets the symmetric normalized
    Laplacian. ``diag_lambda`` is ignored in that case.
    """

    features_batches = []
    support_batches = []
    y_train_batches = []
    train_mask_batches = []
    total_nnz = 0
    np.random.shuffle(parts)
    for _, st in enumerate(range(0, num_clusters, block_size)):
        pt = parts[st]
        for pt_idx in range(st + 1, min(st + block_size, num_clusters)):
            pt = np.concatenate((pt, parts[pt_idx]), axis=0)
            pt = pt.astype(np.int32)
        features_batches.append(torch.tensor(features[pt, :]).float())
        y_train_batches.append(torch.tensor(y_train[pt, :]).float())
        support_now = adj[pt, :][:, pt]
        if not normalize:
            support_batches.append(sparse_to_tuple(support_now))
        elif diag_lambda == -1:
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
    return (features_batches, support_batches, y_train_batches,
          train_mask_batches)


def preprocess(adj,
               features,
               y_train,
               train_mask,
               visible_data,
               num_clusters,
               diag_lambda=-1,
               partition_cache_path=None,
               normalize=True):
    """Do graph partitioning and preprocessing for SGD training.

    When ``normalize=False`` the partitioned adjacency is kept raw and
    symmetric (no row-stochastic / diag-enhance normalization). This is the
    mode used by the spectral pipeline; ``diag_lambda`` is ignored in that
    case.
    """

    # Do graph partitioning
    part_adj, parts = partition_utils.partition_graph(
        adj, visible_data, num_clusters, cache_path=partition_cache_path)
    if normalize:
        if diag_lambda == -1:
            part_adj = normalize_adj(part_adj)
        else:
            part_adj = normalize_adj_diag_enhance(part_adj, diag_lambda)
    parts = [np.array(pt) for pt in parts]

    features_batches = []
    support_batches = []
    y_train_batches = []
    train_mask_batches = []
    total_nnz = 0
    for pt in parts:
        features_batches.append(torch.tensor(features[pt, :]).float())
        now_part = part_adj[pt, :][:, pt]
        total_nnz += now_part.count_nonzero()
        support_batches.append(sparse_to_tuple(now_part))
        y_train_batches.append(torch.tensor(y_train[pt, :]).float())

        train_pt = []
        for newidx, idx in enumerate(pt):
            if train_mask[idx]:
                train_pt.append(newidx)
        train_mask_batches.append(sample_mask(train_pt, len(pt)))
    return (parts, features_batches, support_batches, y_train_batches,
          train_mask_batches)


def load_graphsage_data(dataset_path, dataset_str, normalize=True, file_prefix=None):
    """Load GraphSAGE data.

    dataset_path: base path to data (e.g. '../../data').
    dataset_str: subdirectory under dataset_path (e.g. 'cora' or 'cora/3_3').
    file_prefix: prefix of -G.json, -id_map.json, etc. Defaults to dataset_str.
                 Use when files are named e.g. cora-* but live in cora/3_3/.
    """
    start_time = time.time()
    if file_prefix is None:
        file_prefix = dataset_str

    current_path = os.getcwd()
    base = f"{current_path}/{dataset_path}/{dataset_str}"

    with open(f"{base}/{file_prefix}-G.json", "r") as file:
        graph_json = json.load(file)

    graph_nx = json_graph.node_link_graph(graph_json)

    with open(f"{base}/{file_prefix}-id_map.json", "r") as file:
        id_map = json.load(file)

    is_digit = list(id_map.keys())[0].isdigit()

    id_map = {k: int(v) for k, v in id_map.items()}
    id_map_re = {value: key for key, value in id_map.items()}
    with open(f"{base}/{file_prefix}-class_map.json", "r") as file:
        class_map = json.load(file)

    is_instance = isinstance(list(class_map.values())[0], list)

    broken_count = 0
    to_remove = []
    for node in graph_nx.nodes():
      if node not in id_map and node not in id_map_re:
        to_remove.append(node)
        broken_count += 1
    for node in to_remove:
      graph_nx.remove_node(node)
    # print(f'Removed {broken_count} nodes that lacked proper annotations due to networkx versioning issues')

    with open(f"{base}/{file_prefix}-feats.npy", "rb") as file:
        feats = np.load(file)

    # print('Loaded data in {:.2f} seconds'.format(time.time() - start_time))
    start_time = time.time()

    # print('num of nodes: ' + str(len(graph_nx.nodes())))

    edges = []

    for edge in graph_nx.edges():
        if edge[0] in id_map_re and edge[1] in id_map_re:
            edges.append((edge[0], edge[1]))
    num_data = len(id_map)

    # print('num of edges: ' + str(len(graph_nx.edges())))


    # Map graph nodes to integer indices robustly, handling both string and int node IDs.
    def _node_to_idx(node):
        # If the node label itself is a key in id_map (e.g. string ID), use that.
        if node in id_map:
            return id_map[node]
        # Otherwise, if the node is already an integer index (present in id_map_re),
        # just return it directly.
        if node in id_map_re:
            return int(node)
        raise KeyError(f"Node {node} not found in id_map or id_map_re")

    val_data = np.array(
        [_node_to_idx(n) for n in graph_nx.nodes()
         if 'val' in graph_nx.nodes[n] and graph_nx.nodes[n]['val']],
        dtype=np.int32)

    test_data = np.array(
        [_node_to_idx(n) for n in graph_nx.nodes()
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
        train_ids = np.array([
            _node_to_idx(n)
            for n in graph_nx.nodes()
            if 'val' in graph_nx.nodes[n]
               and not graph_nx.nodes[n]['val']
               and not graph_nx.nodes[n]['test']
        ])
        train_feats = feats[train_ids]
        scaler = sklearn.preprocessing.StandardScaler()
        scaler.fit(train_feats)
        feats = scaler.transform(feats)

    def _construct_adj(edges):
        adj = sp.csr_matrix((np.ones(
            (edges.shape[0]), dtype=np.float32), (edges[:, 0], edges[:, 1])),
            shape=(num_data, num_data))
        adj += adj.transpose()
        adj += sp.eye(num_data)
        # adj = adj @ adj 
        # binarize the adjacency matrix to 1 and 0
        # adj[adj > 0] = 1
        return adj

    train_adj = _construct_adj(train_edges)
    full_adj = _construct_adj(edges)
    
    train_feats = feats
    test_feats = feats

    return num_data, train_adj, full_adj, feats, train_feats, test_feats, labels, train_data, val_data, test_data


def get_number_of_clusters(X: torch.Tensor,  n_samples: int, threshold: float) -> int:
    """
    Computes the number of clusters in the given dataset

    Args:
        X:          dataset
        n_samples:  number of samples to use for computing the number of clusters
        threshold:  threshold for the eigenvalues of the laplacian matrix. This 
                    threshold is used in order to find when the difference between 
                    the eigenvalues becomes large. 

    Returns:
        Number of clusters in the dataset
    """
    indices = torch.randperm(X.shape[0])[:n_samples]
    X = X[indices]
    
    W = get_affinity_matrix(X)
    L = get_laplacian(W)
    vals = get_eigenvalues(L)
    diffs = np.diff(vals)
    cutoff = np.argmax(diffs > threshold)
    num_clusters = cutoff + 1
    return num_clusters


def build_ann(X: torch.Tensor):
    """
    Builds approximate-nearest-neighbors object 
    that can be used to calculate the knn of a data-point

    Args:
        X:  dataset
    """
    X = X.view(X.size(0), -1)
    t = AnnoyIndex(X[0].shape[0], 'euclidean')
    for i, x_i in enumerate(X):
        t.add_item(i, x_i)

    t.build(50)
    t.save('ann_index.ann')


def make_batch_for_sparse_grapsh(batch_x: torch.Tensor) -> torch.Tensor:
    """
    Computes new batch of data points from the given batch (batch_x) 
    in case that the graph-laplacian obtained from the given batch is sparse.
    The new batch is computed based on the nearest neighbors of 0.25
    of the given batch

    Args:
        batch_x:    Batch of data points

    Returns:
        New batch of data points
    """

    batch_size = batch_x.shape[0]
    batch_size //= 5
    new_batch_x = batch_x[:batch_size]
    batch_x = new_batch_x
    n_neighbors = 5

    u = AnnoyIndex(batch_x[0].shape[0], 'euclidean')
    u.load('ann_index.ann')
    for x in batch_x:
        x = x.detach().cpu().numpy()
        nn_indices = u.get_nns_by_vector(x, n_neighbors)
        nn_tensors = [u.get_item_vector(i) for i in nn_indices[1:]]
        nn_tensors = torch.tensor(nn_tensors)
        new_batch_x = torch.cat((new_batch_x, nn_tensors))

    return new_batch_x


def get_laplacian(W: torch.Tensor) -> np.ndarray:
    """
    Computes the un-normalized Laplacian matrix, given the affinity matrix W

    Args:
        W (torch.Tensor):   Affinity matrix
    
    Returns:
        Laplacian matrix
    """

    W = W.detach().cpu().numpy()
    D = np.diag(W.sum(axis=1))
    L = D - W
    return L


def sort_laplacian(L: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Sorts the columns and the rows of the laplacian by the true lablel in order
    to see whether the sorted laplacian is a block diagonal matrix

    Args:
        L:  Laplacian matrix
        y:  labels

    Returns:
        Sorted laplacian
    """

    i = np.argsort(y)
    L = L[i, :]
    L = L[:, i]
    return L


def sort_matrix_rows(A: np.ndarray , y: np.ndarray) -> np.ndarray:
    """
    Sorts the rows of a matrix by a given order y

    Args:
        A:  Numpy ndarray
        y:  True labels
    """

    i = np.argsort(y)
    A = A[i, :]
    return A


def get_eigenvalues(A: np.ndarray) -> np.ndarray:
    """
    Computes the eigenvalues of a given matrix A and sorts them in increasing order

    Args:
        A:  Numpy ndarray

    Returns:
        Sorted eigenvalues
    """

    _, vals, _ = np.linalg.svd(A)
    sorted_vals = vals[np.argsort(vals)]
    return sorted_vals


def get_eigenvectors(A: np.ndarray) -> np.ndarray:
    """
    Computes the eigenvectors of a given matrix A and sorts them by the eigenvalues
    Args:
        A:  Numpy ndarray

    Returns:
        Sorted eigenvectors
    """

    vecs, vals, _ = np.linalg.svd(A)
    vecs = vecs[:, np.argsort(vals)]
    return vecs


def plot_eigenvalues(vals: np.ndarray):
    """
    Plot the eigenvalues of the laplacian

    Args:
        vals:   Eigenvalues
    """

    rang = range(len(vals))
    plt.plot(rang, vals)
    plt.show()


def get_laplacian_eigenvectors(V: torch.Tensor, y: np.ndarray) -> np.ndarray:
    """
    Returns eigenvectors of the laplacian when the data is in increasing order by the true label.
    i.e., the rows of the eigenvectors matrix V are sorted by the true labels in increasing order.

    Args:
        V:  Eigenvectors matrix
        y:  True labels
    """

    V = sort_matrix_rows(V, y)
    rang = range(len(y))
    return V, rang


def plot_laplacian_eigenvectors(V: np.ndarray, y: np.ndarray):
    """
    Plot the eigenvectors of the laplacian when the data is in increasing order by the true label.
    i.e., the rows of the eigenvectors matrix V are sorted by the true labels in increasing order.

    Args:
        V:  Eigenvectors matrix
        y:  True labels
    """

    # sort the rows of V
    V = sort_matrix_rows(V, y)
    rang = range(len(y))
    plt.plot(rang, V)
    plt.show()
    return plt


def plot_sorted_laplacian(W: torch.Tensor, y: np.ndarray):
    """
    Plot the block diagonal matrix that is obtained from the sorted laplacian

    Args:
        W:  Affinity matrix
        y:  True labels
    """
    L = get_laplacian(W)
    L = sort_laplacian(L, y)
    plt.imshow(L, cmap='hot', norm=colors.LogNorm())
    plt.imshow(L, cmap='flag')
    plt.show()


def get_nearest_neighbors(X: torch.Tensor, Y: torch.Tensor = None, k: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes the distances and the indices of the
    k nearest neighbors of each data point

    Args:
        X:              Batch of data points
        Y (optional):   Defaults to None.
        k:              Number of nearest neighbors to calculate. Defaults to 3.

    Returns:
        Distances and indices of each datapoint
    """

    if Y is None:
        Y = X
    if len(X) < k:
        k = len(X)
    X = X.cpu().detach().numpy()
    Y = Y.cpu().detach().numpy()
    nbrs = NearestNeighbors(n_neighbors=k).fit(Y)
    Dis, Ids = nbrs.kneighbors(X)
    return Dis, Ids


def get_grassman_distance(A: np.ndarray, B: np.ndarray) -> float:
    """
    Computes the Grassmann distance between the subspaces spanned by the columns of A and B.

    Parameters
    ----------
    A : np.ndarray
        Numpy ndarray.
    B : np.ndarray
        Numpy ndarray.

    Returns
    -------
    float
        The Grassmann distance.
    """

    A, _ = np.linalg.qr(A)
    B, _ = np.linalg.qr(B)

    M = np.dot(np.transpose(A), B)
    _, s, _ = np.linalg.svd(M, full_matrices=False)
    s = 1 - np.square(s)
    grassmann = np.sum(s)
    return grassmann


def compute_scale(Dis: np.ndarray, k: int = 2, med: bool = True, is_local: bool = True) -> np.ndarray:
    """
    Computes the scale for the Gaussian similarity function

    Args:
        Dis:        Distances of the k nearest neighbors of each data point.
        k:          Number of nearest neighbors. Defaults to 2.
        med:        Scale calculation method. Can be calculated by the median distance
                    from a data point to its neighbors, or by the maximum distance. 
        is_local:   Local distance (different for each data point), or global distance. Defaults to local.

    Returns:
        scale (global or local)
    """

    if is_local:
        if not med:
            scale = np.max(Dis, axis=1)
        else:
            scale = np.median(Dis, axis=1)
    else:
        if not med:
            scale = np.max(Dis[:, k - 1])
        else:
            scale = np.median(Dis[:, k - 1])
    return scale


def get_gaussian_kernel(D: torch.Tensor, scale, Ids: np.ndarray, device: torch.device, is_local: bool = True) -> torch.Tensor:   
    """
    Computes the Gaussian similarity function 
    according to a given distance matrix D and a given scale

    Args:
        D:      Distance matrix 
        scale:  scale
        Ids:    Indices of the k nearest neighbors of each sample
        device: Defaults to torch.device("cpu")
        is_local:  Determines whether the given scale is global or local 

    Returns:
        Matrix W with Gaussian similarities
    """

    if not is_local:
        # global scale
        W = torch.exp(-torch.pow(D, 2) / (scale ** 2))
    else:
        # local scales
        W = torch.exp(-torch.pow(D, 2).to(device) / (torch.tensor(scale).float().to(device).clamp_min(1e-7) ** 2))
    if Ids is not None:
        n, k = Ids.shape
        mask = torch.zeros([n, n]).to(device=device)
        for i in range(len(Ids)):
            mask[i, Ids[i]] = 1
        W = W * mask
    sym_W = (W + torch.t(W)) / 2.
    return sym_W


def plot_data_by_assignmets(X, assignments: np.ndarray):
    """
    Plots the data with the assignments obtained from SpectralNet.
    Relevant only for 2D data

    Args:
        X:                      Data
        cluster_assignments:    Cluster assignments 
    """

    plt.scatter(X[:, 0], X[:, 1], c=assignments)
    plt.show()


def calculate_cost_matrix(C: np.ndarray , n_clusters: int) -> np.ndarray:
    """
    Calculates the cost matrix for the Munkres algorithm

    Args:
        C (np.ndarray):     Confusion matrix
        n_clusters (int):   Number of clusters

    Returns:
        np.ndarray:        Cost matrix
    """
    cost_matrix = np.zeros((n_clusters, n_clusters))
    # cost_matrix[i,j] will be the cost of assigning cluster i to label j
    for j in range(n_clusters):
        s = np.sum(C[:, j])  # number of examples in cluster i
        for i in range(n_clusters):
            t = C[i, j]
            cost_matrix[j, i] = s - t
    return cost_matrix


def get_cluster_labels_from_indices(indices: np.ndarray) -> np.ndarray:
    """
    Gets the cluster labels from their indices

    Args:
        indices (np.ndarray):  Indices of the clusters

    Returns:
        np.ndarray:   Cluster labels
    """

    num_clusters = len(indices)
    cluster_labels = np.zeros(num_clusters)
    for i in range(num_clusters):
        cluster_labels[i] = indices[i][1]
    return cluster_labels


def write_assignmets_to_file(assignments: np.ndarray):
    """
    Saves SpectralNet cluster assignments to a file

    Args:
        assignments (np.ndarray): The assignments that obtained from SpectralNet
    """

    np.savetxt("cluster_assignments.csv", assignments.astype(int), fmt='%i', delimiter=',')


def get_affinity_matrix(X: torch.Tensor) -> torch.Tensor:
    """
    Computes the affinity matrix W

    Args:
        X (torch.Tensor):  Data

    Returns:
        torch.Tensor: Affinity matrix W
    """
    is_local = True
    n_neighbors = 30
    scale_k = 15
    Dx = torch.cdist(X,X)
    Dis, indices = get_nearest_neighbors(X, k=n_neighbors + 1)
    scale = compute_scale(Dis, k=scale_k, is_local=is_local)
    W = get_gaussian_kernel(Dx, scale, indices, device=torch.device("cpu"), is_local=is_local)
    return W

