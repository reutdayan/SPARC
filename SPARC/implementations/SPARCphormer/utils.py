import numpy as np
import scipy.sparse as sp
import torch
# Shim for PyTorch 2.x: torch.utils._import_utils was removed; torchdata still expects it
import sys
if "torch.utils._import_utils" not in sys.modules:
    import types
    _mod = types.ModuleType("torch.utils._import_utils")
    try:
        import dill as _dill

        def _dill_available():
            return True
    except ImportError:
        def _dill_available():
            return False
    _mod.dill_available = _dill_available
    sys.modules["torch.utils._import_utils"] = _mod

import torch.nn.functional as F
import pickle
import os
import re
import copy
import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch as th
from sklearn.model_selection import ShuffleSplit
from tqdm import tqdm
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors
import time


def normalize_features(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def normalize_adj(mx):
    """Row-column-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1/2).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx).dot(r_mat_inv)
    return mx


def normalize_laplacian(mx):
    """Row-column-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1/2).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx).dot(r_mat_inv)
    mx = sp.eye(mx.shape[0]) - mx
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def accuracy_batch(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def torch_sparse_tensor_to_sparse_mx(torch_sparse):
    """Convert a torch sparse tensor to a scipy sparse matrix."""

    m_index = torch_sparse._indices().numpy()
    row = m_index[0]
    col = m_index[1]
    data = torch_sparse._values().numpy()

    sp_matrix = sp.coo_matrix((data, (row, col)), shape=(
        torch_sparse.size()[0], torch_sparse.size()[1]))

    return sp_matrix


def laplacian_positional_encoding(g, pos_enc_dim):
    """
        Graph positional encoding v/ Laplacian eigenvectors
    """
    import dgl  # lazy import: only needed for this DGL-based code path

    # Laplacian

    # adjacency_matrix(transpose, scipy_fmt="csr")
    A = g.adjacency_matrix_scipy(return_edge_ids=False).astype(float)
    N = sp.diags(dgl.backend.asnumpy(
        g.in_degrees()).clip(1) ** -0.5, dtype=float)
    L = sp.eye(g.number_of_nodes()) - N * A * N

    # Eigenvectors with scipy
    # EigVal, EigVec = sp.linalg.eigs(L, k=pos_enc_dim+1, which='SR')
    EigVal, EigVec = sp.linalg.eigs(
        L, k=pos_enc_dim+1, which='SR', tol=1e-2)  # for 40 PEs
    EigVec = EigVec[:, EigVal.argsort()]  # increasing order
    lap_pos_enc = torch.from_numpy(EigVec[:, 1:pos_enc_dim+1]).float()

    return lap_pos_enc


def nor_matrix(adj, a_matrix):

    nor_matrix = torch.mul(adj, a_matrix)
    row_sum = torch.sum(nor_matrix, dim=1, keepdim=True)
    nor_matrix = nor_matrix / row_sum

    return nor_matrix


def re_features_spectral_diffusion_distance_avarage_seq(graph, features, K, pe_dim=10, t=2):
    import dgl  # lazy import: only needed for this DGL-based code path

    # Compute  symmetric Laplacian matrix
    A = graph.adjacency_matrix_scipy(return_edge_ids=False).astype(float)
    D = sp.diags(dgl.backend.asnumpy(
        graph.in_degrees()).clip(1) ** -0.5, dtype=float)
    L = sp.eye(graph.number_of_nodes()) - D * A * D

    # Eigenvectors with scipy
    EigVal, EigVec = sp.linalg.eigs(L, k=pe_dim, which='SR', tol=1e-2)
    EigVec = EigVec[:, EigVal.argsort()]  # increasing order
    EigVal = EigVal[EigVal.argsort()]

    EigVec = np.real(EigVec)
    EigVal = np.real(EigVal)

    # Compute the modified diffusion distances (1 - eigenvalue)^2
    modified_distances = (1 - EigVal) + (1 - EigVal) ** 2 + \
        (1-EigVal)**6 + (1-EigVal)**4

    # Scale each eigenvector column by the corresponding diffusion distance
    scaled_eigenvectors = EigVec * modified_distances

    # Use approximate k-NN to find the most similar nodes based on scaled eigenvectors

    knn = NearestNeighbors(n_neighbors=1024, algorithm='auto').fit(
        scaled_eigenvectors)
    distances, indices = knn.kneighbors(scaled_eigenvectors)

    # Create a tensor to hold the node features
    nodes_features = torch.empty(
        (scaled_eigenvectors.shape[0], K+1, features.shape[1]))
    # Get indices of the most similar nodes (excluding the node itself)
    # skip the first one because it's the node itself
    similar_indices = indices[:, 1:]

    # Stack the features for each node and its K most similar nodes
    nodes_features[:, 0, :] = torch.tensor(features)  # current node's features

    starts_idx = [0, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    # starts_idx = [0,10,20,30,40,50,60,70,80,90,100]

    # Calculate the average of different sets of 10 closest nodes for positions 1 to 10
    for i in range(1, K+1):
        # Get the indices of the next set of 10 closest nodes
        start_idx = starts_idx[i-1]
        end_idx = starts_idx[i]
        # start_idx = (i-1) * seq_len
        # end_idx = (i) * seq_len

        closest_indices = similar_indices[:, :end_idx]
        distances = distances[:, :end_idx]
        distance = distances.mean(axis=1)

        closest_indices = torch.tensor(closest_indices)
        features = torch.tensor(features)

        # Calculate the average features of these 10 closest nodes
        avg_features = features[closest_indices].mean(
            dim=1) / distance[:, np.newaxis]
        nodes_features[:, i, :] = avg_features

    return nodes_features


def re_features_spectral_diffusion_distance_avarage_seq_spectral_encoding(spectral_encoding, features, K, train_mask, val_mask):
    # Create a tensor to hold the node features
    nodes_features = torch.empty(
        (spectral_encoding.shape[0], K+1, features.shape[1]))
    # Get indices of the most similar nodes (excluding the node itself)
    # Stack the features for each node and its K most similar nodes
    nodes_features[:, 0, :] = torch.tensor(features)  # current node's features
    starts_idx = [0, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    # starts_idx = [0,10,20,50,100,200,500,1000,2000,3000,5000]

    # Use approximate k-NN to find the most similar nodes based on euclidean distance on the spectral encoding
    knn = NearestNeighbors(
        n_neighbors=2000, algorithm='auto').fit(spectral_encoding)
    distances, indices = knn.kneighbors(spectral_encoding)

    '''
    for node in tqdm(range(indices.shape[0])):
        for k in range(1,indices.shape[1]):
            if not train_mask[indices[node][k]].any() or not val_mask[indices[node][k]].any():
                np.delete(indices[node], k)
    '''
    # Efficient filtering of indices using numpy vectorized operations
    for node in tqdm(range(indices.shape[0])):
        # Get all valid neighbors for this node by combining the train and val mask conditions
        valid_neighbors = indices[node][(
            train_mask[indices[node]] | val_mask[indices[node]])]
        if len(valid_neighbors) < 2 ** K:
            # If there are not enough valid neighbors, pad with the last valid neighbor
            valid_neighbors = np.pad(
                valid_neighbors, (node, 2**K - len(valid_neighbors)), mode='edge')
        # Update the indices array with the valid neighbors
        indices[node, 1:len(valid_neighbors)
                ] = valid_neighbors[:len(valid_neighbors)-1]

    for i in tqdm(range(1, K+1)):
        end_idx = starts_idx[i]
        closest_indices = indices[:, :end_idx+1]

        closest_indices = torch.tensor(closest_indices)
        features = torch.tensor(features)

        # Calculate the average features of these 10 closest nodes
        avg_features = features[closest_indices].mean(dim=1)
        nodes_features[:, i, :] = avg_features

    return nodes_features


def re_features_hops(adj, features, K):
    features = torch.tensor(features, dtype=torch.float32)
    nodes_features = torch.empty(features.shape[0], 1, K+1, features.shape[1])

    for i in range(features.shape[0]):

        nodes_features[i, 0, 0, :] = features[i]

    x = features + torch.zeros_like(features)

    for i in range(K):

        x = torch.matmul(adj, x)

        for index in range(features.shape[0]):

            nodes_features[index, 0, i + 1, :] = x[index]

    nodes_features = nodes_features.squeeze()

    return nodes_features


def scipy_to_torch_sparse_coo(scipy_matrix):
    """
    Converts a scipy sparse matrix to a PyTorch sparse COO tensor.
    Ensures the matrix is in COO format before conversion.

    Args:
        scipy_matrix (scipy.sparse matrix): The input scipy sparse matrix.

    Returns:
        torch.sparse_coo_tensor: The converted PyTorch sparse COO tensor.
    """
    # Ensure the matrix is in COO format
    if not sp.isspmatrix_coo(scipy_matrix):
        scipy_matrix = scipy_matrix.tocoo()

    # Extract the row, col, and data from the scipy sparse COO matrix
    rows = torch.tensor(scipy_matrix.row, dtype=torch.long)
    cols = torch.tensor(scipy_matrix.col, dtype=torch.long)
    values = torch.tensor(scipy_matrix.data, dtype=torch.float32)

    # Stack the row and column indices to match the format required by PyTorch
    indices = torch.stack([rows, cols])

    # Create the PyTorch sparse COO tensor with the same shape as the scipy matrix
    torch_sparse_coo = torch.sparse_coo_tensor(
        indices, values, scipy_matrix.shape)

    return torch_sparse_coo


def re_features_spectral_diffusion_distance_avarage_seq_spectral_encoding_coldstart(
        spectral_encoding, features, K, train_mask, val_mask, test_mask,
        hop_avg_style="exponential", knn_metric="minkowski",
        retrieval_space="knn", adj=None):

    print("re_features_spectral_diffusion_distance_avarage_seq_spectral_encoding_coldstart")
    print("K: ", K)
    if hop_avg_style == "exponential":
        starts_idx = [1, 2, 4, 8, 16, 32, 64, 128, 256,
                      512, 1024, 2048, 4096, 8192, 16384, 32768]
        max_neighbors = starts_idx[min(K, len(starts_idx) - 1)]
    elif hop_avg_style == "linear":
        step = max(1, min(2**14, len(train_mask)) // (K + 1))
        starts_idx = [step * i for i in range(K + 2)]
        max_neighbors = starts_idx[K]
    else:
        raise ValueError(
            "hop_avg_style must be 'exponential' or 'linear', got '{}'".format(hop_avg_style))

    features_tensor = torch.tensor(features, dtype=torch.float32)
    n_nodes, n_features = features_tensor.shape
    result_shape = (n_nodes, K + 1, n_features)

    nodes_features = torch.empty(result_shape, dtype=torch.float32)
    nodes_features[:, 0, :] = features_tensor
    retrieval_space = str(retrieval_space).lower()
    if retrieval_space == "knn":
        # Cold-start: index only train|val nodes (test nodes are never neighbors).
        # float32 + n_jobs cuts memory and wall time vs indexing the full graph.
        candidate_mask = (
            np.asarray(train_mask, dtype=bool)
            | np.asarray(val_mask, dtype=bool)
        )
        candidate_global_idx = np.flatnonzero(candidate_mask)
        n_candidates = int(candidate_global_idx.shape[0])
        if n_candidates < 2:
            raise ValueError(
                "re_features_spectral_diffusion_distance_avarage_seq_spectral_encoding_coldstart: "
                "need at least two train|val nodes for KNN retrieval.")
        spectral_np = np.ascontiguousarray(
            np.asarray(spectral_encoding, dtype=np.float32)
        )
        n_neighbors_query = min(max(max_neighbors + 1, 2), n_candidates)
        knn = NearestNeighbors(
            n_neighbors=n_neighbors_query,
            algorithm="auto",
            metric=knn_metric,
            n_jobs=-1,
        )
        knn.fit(spectral_np[candidate_mask])
        try:
            _, local_idx = knn.kneighbors(spectral_np, n_jobs=-1)
        except TypeError:
            _, local_idx = knn.kneighbors(spectral_np)
        gidx = candidate_global_idx[local_idx.astype(np.int64, copy=False)]

        node_ids = np.arange(n_nodes, dtype=gidx.dtype)[:, None]
        is_self = gidx == node_ids
        self_pos = np.where(
            is_self.any(axis=1),
            is_self.argmax(axis=1),
            np.int64(n_neighbors_query),
        )
        ncol = min(max_neighbors, n_neighbors_query - 1)
        col_idx = np.arange(ncol, dtype=np.int64)[None, :]
        col_idx = col_idx + (col_idx >= self_pos[:, None]).astype(np.int64)
        col_idx = np.minimum(col_idx, gidx.shape[1] - 1)
        indices = np.take_along_axis(gidx, col_idx, axis=1)

        for j in tqdm(range(K), desc="knn-hop"):
            end_idx = starts_idx[j + 1] if j + \
                1 < len(starts_idx) else indices.shape[1]
            end_idx = min(end_idx, indices.shape[1])
            closest_indices = indices[:, :end_idx]

            for i in range(0, n_nodes, 1024):
                chunk_indices = closest_indices[i: i + 1024]
                chunk_features = torch.mean(
                    features_tensor[chunk_indices], dim=1)
                nodes_features[i: i + 1024, j + 1, :] = chunk_features
    elif retrieval_space == "real_graph":
        if adj is None:
            raise ValueError(
                "retrieval_space='real_graph' requires a graph adjacency matrix (adj).")
        if not sp.issparse(adj):
            raise TypeError(
                "adj must be a scipy sparse matrix for retrieval_space='real_graph'.")
        adj_csr = adj.tocsr()

        candidate_mask = np.asarray(train_mask).astype(
            bool) | np.asarray(val_mask).astype(bool)
        spectral_encoding_np = np.asarray(spectral_encoding)
        mean_fallback = features_tensor.mean(dim=0)

        for node in tqdm(range(n_nodes)):
            row_start = adj_csr.indptr[node]
            row_end = adj_csr.indptr[node + 1]
            neighbors = adj_csr.indices[row_start:row_end]

            if neighbors.size > 0:
                neighbors = neighbors[(neighbors != node)
                                      & candidate_mask[neighbors]]

            if neighbors.size == 0:
                for j in range(K):
                    nodes_features[node, j + 1, :] = mean_fallback
                continue

            distances = pairwise_distances(
                spectral_encoding_np[node:node + 1],
                spectral_encoding_np[neighbors],
                metric=knn_metric
            ).reshape(-1)
            sorted_neighbors = neighbors[np.argsort(distances)]
            sorted_neighbors_t = torch.from_numpy(sorted_neighbors).long()

            for j in range(K):
                end_idx = starts_idx[j + 1] if j + \
                    1 < len(starts_idx) else sorted_neighbors_t.shape[0]
                end_idx = min(end_idx, sorted_neighbors_t.shape[0])
                if end_idx <= 0:
                    nodes_features[node, j + 1, :] = mean_fallback
                else:
                    nodes_features[node, j + 1, :] = torch.mean(
                        features_tensor[sorted_neighbors_t[:end_idx]], dim=0)
    else:
        raise ValueError(
            "retrieval_space must be 'knn' or 'real_graph', got '{}'".format(retrieval_space))

    return nodes_features

    # # Use approximate k-NN to find the most similar nodes based on euclidean distance on the  spectral encoding
    # n_train = train_mask.sum()
    # knn = NearestNeighbors(n_neighbors=2**14, algorithm='auto').fit(spectral_encoding[train_mask])
    # distances, indices = knn.kneighbors(spectral_encoding)
    # # Create a tensor to hold the node features
    # nodes_features = torch.empty((spectral_encoding.shape[0], K+1, features.shape[1] ))
    # # Get indices of the most similar nodes (excluding the node itself)
    # # Stack the features for each node and its K most similar nodes
    # nodes_features[:, 0, :] = torch.tensor(features)  # current node's features
    # starts_idx= [0,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192,16384,32768]
    # # Combine distances and indices for easy slicing
    # def get_closest_features(node_indices, K, starts_idx, features_tensor, train_mask):
    #     closest_features = torch.empty(features_tensor.shape[0], K+1, features_tensor.shape[1])
    #     for j in tqdm(range(K)):
    #         end_idx = starts_idx[j+1]
    #         closest_indices = node_indices[:, :end_idx]
    #         average_features = torch.mean(features_tensor[closest_indices], dim=1)
    #         closest_features[:, j, :] = average_features
    #     return closest_features

    # features_tensor = torch.tensor(features)
    # nodes_features[:, 1:, :] = get_closest_features(indices, K, starts_idx, features_tensor, train_mask)
    # return nodes_features


def re_features_knn_sequence_coldstart(
        spectral_encoding, features, K, train_mask, val_mask, test_mask,
        knn_metric="minkowski"):
    """Build per-node token sequences from the K nearest cold-start-safe neighbors.

    For every node a ``(K+1, n_features)`` token sequence is produced:

        token 0       : the node's own features
        token 1..K    : features of the 1st, 2nd, ..., K-th nearest *other* node

    KNN is computed in the ``spectral_encoding`` space, with neighbor
    candidates restricted to ``train_mask | val_mask`` (cold-start safe: test
    nodes are never picked as neighbors, although they still receive their own
    token sequence). No averaging is performed and no distance weighting is
    applied -- the raw neighbor features are emitted in nearest-first order.

    Signature mirrors
    :func:`re_features_spectral_diffusion_distance_avarage_seq_spectral_encoding_coldstart`
    so it is a drop-in replacement in ``train.py``.

    Parameters
    ----------
    spectral_encoding : np.ndarray, shape (N, d_enc)
        The space in which KNN is computed.
    features : np.ndarray or torch.Tensor, shape (N, n_features)
        Features used to populate the tokens.
    K : int
        Number of neighbor tokens per node (total token length is K+1).
    train_mask, val_mask, test_mask : array-like of bool, shape (N,)
        Node split masks. Candidate neighbors = train OR val.
    knn_metric : str
        Distance metric for :class:`sklearn.neighbors.NearestNeighbors`.

    Returns
    -------
    torch.FloatTensor, shape (N, K+1, n_features)
    """
    train_mask_b = np.asarray(train_mask).astype(bool)
    val_mask_b = np.asarray(val_mask).astype(bool)
    # test_mask is accepted for API symmetry; test nodes are simply never
    # candidates. We keep the argument to make the cold-start intent explicit.
    _ = np.asarray(test_mask).astype(bool)

    candidate_mask = train_mask_b | val_mask_b
    candidate_global_idx = np.nonzero(candidate_mask)[0]
    n_candidates = candidate_global_idx.shape[0]
    if n_candidates == 0:
        raise ValueError(
            "re_features_knn_sequence_coldstart: no candidate neighbors "
            "(train_mask | val_mask is empty).")
    if n_candidates < K + 1:
        raise ValueError(
            "re_features_knn_sequence_coldstart: need at least K+1={} "
            "candidate neighbors to guarantee K distinct neighbors per node, "
            "but only {} are available.".format(K + 1, n_candidates))

    # Query K+1 neighbors so we can drop the query node itself if it happens
    # to be a candidate (train/val node), and still have K remaining.
    n_neighbors = K + 1

    knn = NearestNeighbors(
        n_neighbors=n_neighbors, algorithm="auto", metric=knn_metric, n_jobs=-1
    )
    knn.fit(spectral_encoding[candidate_mask])
    try:
        _, local_indices = knn.kneighbors(spectral_encoding, n_jobs=-1)
    except TypeError:
        _, local_indices = knn.kneighbors(spectral_encoding)
    # Map local (candidate-subset) indices back to global node indices.
    global_indices = candidate_global_idx[local_indices]  # (N, K+1)

    n_nodes = global_indices.shape[0]
    node_ids = np.arange(n_nodes)[:, None]
    is_self = (global_indices == node_ids)  # (N, K+1)

    # For each row, drop "self" if present (at most once) and take the first K
    # of the remaining entries. If self is absent (test nodes), just take the
    # first K.
    self_pos = np.where(is_self.any(axis=1),
                        is_self.argmax(axis=1),
                        n_neighbors)  # K+1 means "not present"
    col_idx = np.arange(K)[None, :]
    col_idx = col_idx + (col_idx >= self_pos[:, None]).astype(col_idx.dtype)
    neighbor_idx = np.take_along_axis(
        global_indices, col_idx, axis=1)  # (N, K)

    if isinstance(features, torch.Tensor):
        features_tensor = features.to(dtype=torch.float32)
    else:
        features_tensor = torch.as_tensor(features, dtype=torch.float32)
    n_features = features_tensor.shape[1]

    nodes_features = torch.empty((n_nodes, K + 1, n_features),
                                 dtype=torch.float32)
    nodes_features[:, 0, :] = features_tensor
    neighbor_idx_t = torch.from_numpy(neighbor_idx).long()
    nodes_features[:, 1:, :] = features_tensor[neighbor_idx_t]

    return nodes_features


def link_prediction_coldstart(adj, spectral_encoding, train_mask, val_mask, test_mask):
    """
    Perform link prediction for cold-start nodes by recalculating adjacency
    based on k-Nearest Neighbors (k-NN).

    Parameters:
    - adj: scipy.sparse.csr_matrix (N x N adjacency matrix)
    - spectral_encoding: np.ndarray (N x D feature matrix for nodes)
    - train_mask: np.ndarray (Boolean mask for training nodes)
    - val_mask: np.ndarray (Boolean mask for validation nodes)
    - test_mask: np.ndarray (Boolean mask for test nodes)

    Returns:
    - adj: scipy.sparse.csr_matrix (Modified adjacency matrix with updated connections)
    """
    # Calculate mean node degree as the target number of neighbors
    mean_node_degree = int(np.round(adj.sum(axis=1).mean()))
    print(f"Mean node degree: {mean_node_degree}")

    # Remove edges involving validation and test nodes
    val_test_mask = val_mask | test_mask
    val_test_indices = np.where(val_test_mask)[0]

    time1 = time.time()

    # Convert adj to LIL format for efficient updates
    adj = adj.tolil()
    for idx in val_test_indices:
        adj[idx, :] = 0  # Remove all outgoing edges
        adj[:, idx] = 0  # Remove all incoming edges

    # Fit k-NN model on training node features
    knn = NearestNeighbors(n_neighbors=mean_node_degree, algorithm='auto')
    knn.fit(spectral_encoding[train_mask])
    distances, indices = knn.kneighbors(spectral_encoding)

    # Add edges for validation and test nodes based on k-NN
    for node_idx in tqdm(val_test_indices, desc="Updating adjacency"):
        neighbors = indices[node_idx]
        adj[node_idx, neighbors] = 1
        adj[neighbors, node_idx] = 1  # Ensure symmetry

    time2 = time.time()
    print(f"Link prediction time: {time2 - time1:.2f}s")

    adj = adj.tocsr()

    save_path = '../data/'

    sp.save_npz(save_path + 'adjacency_matrix_coldstart.npz', adj)

    return adj
