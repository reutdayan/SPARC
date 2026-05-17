from __future__ import print_function

import numpy as np
import random
import json
import sys
import os

import networkx as nx
from networkx.readwrite import json_graph
version_info = list(map(int, nx.__version__.split('.')))
major = version_info[0]
minor = version_info[1]
assert (major <= 1) and (minor <= 11), "networkx major version > 1.11"

WALK_LEN = 5
N_WALKS = 50


def load_data(prefix, normalize=True, load_walks=False, test_ratio=0.03, val_ratio=0.10):
    G_data = json.load(open(prefix + "-G.json"))
    G = json_graph.node_link_graph(G_data)
    if isinstance(G.nodes()[0], int):
        def conversion(n): return int(n)
    else:
        def conversion(n): return n

    if os.path.exists(prefix + "-feats.npy"):
        feats = np.load(prefix + "-feats.npy")
    else:
        print("No features present.. Only identity features will be used.")
        feats = None
    id_map = json.load(open(prefix + "-id_map.json"))
    id_map = {conversion(k): int(v) for k, v in id_map.items()}
    walks = []
    class_map = json.load(open(prefix + "-class_map.json"))
    if isinstance(list(class_map.values())[0], list):
        def lab_conversion(n): return n
    else:
        def lab_conversion(n): return int(n)

    class_map = {conversion(k): lab_conversion(v)
                 for k, v in class_map.items()}

    # ------------------------------------------------------------------
    # Decide whether to preserve existing val/test splits.
    #
    # For SPARC cold-start datasets created via prepare_coldstart_datasets.py,
    # the input GraphSAGE files already:
    #   - mark nodes with 'val' / 'test' flags, AND
    #   - isolate all val/test nodes (no incident edges).
    #
    # In that case we should *not* resample splits or touch edges; we only
    # use the existing flags so that validation/test nodes stay isolated.
    # ------------------------------------------------------------------
    preserve_splits = False
    has_flags_all = True
    for node in G.nodes():
        if 'val' not in G.nodes[node] or 'test' not in G.nodes[node]:
            has_flags_all = False
            break

    if has_flags_all:
        any_flag_edge = False
        for u, v in G.edges():
            if (G.nodes[u].get('val') or G.nodes[u].get('test') or
                    G.nodes[v].get('val') or G.nodes[v].get('test')):
                any_flag_edge = True
                break
        # If no edge ever touches a val/test node, we are already in a
        # cold-start setting; keep the existing flags/splits as-is.
        if not any_flag_edge:
            preserve_splits = True
            print("Detected pre-split cold-start graph: preserving existing "
                  "val/test flags and edge structure from input JSON.")

    if not preserve_splits:
        # Remove all nodes that do not have val/test annotations
        # (necessary because of networkx weirdness with the Reddit data)
        broken_count = 0
        for node in list(G.nodes()):
            if 'val' not in G.nodes[node] or 'test' not in G.nodes[node]:
                G.remove_node(node)
                broken_count += 1
        print("Removed {:d} nodes that lacked proper annotations due to "
              "networkx versioning issues".format(broken_count))

        # Reassign test and val nodes based on the given ratios
        all_nodes = list(G.nodes())
        num_test = max(1, int(len(all_nodes) * test_ratio))
        num_val = max(1, int(len(all_nodes) * val_ratio))
        # random.seed(42)
        test_nodes = set(random.sample(all_nodes, num_test))
        remaining_nodes = [n for n in all_nodes if n not in test_nodes]
        val_nodes = set(random.sample(
            remaining_nodes, min(num_val, len(remaining_nodes))))
        for node in all_nodes:
            G.nodes[node]['test'] = (node in test_nodes)
            G.nodes[node]['val'] = (node in val_nodes)
        print("Selected {:d} test nodes ({:.1f}%) and {:d} val nodes ({:.1f}%) "
              "out of {:d} total nodes".format(
                  num_test, 100.0 * num_test / len(all_nodes),
                  len(val_nodes), 100.0 * len(val_nodes) / len(all_nodes),
                  len(all_nodes)))

        # Check how many nodes are already isolated (degree 0) before our removal
        pre_isolated = [n for n in all_nodes if G.degree(n) == 0]
        print("Already isolated nodes (degree 0) before cold-start removal: {:d} / {:d}".format(
            len(pre_isolated), len(all_nodes)))

        # Cold-start isolation: remove all edges connected to test nodes only
        edges_to_remove = []
        for u, v in list(G.edges()):
            if u in test_nodes or v in test_nodes:
                edges_to_remove.append((u, v))
        G.remove_edges_from(edges_to_remove)
        print("Removed {:d} edges to isolate {:d} test nodes (cold start)".format(
            len(edges_to_remove), num_test))

    print("Loaded data.. now preprocessing..")
    for edge in G.edges():
        if (G.nodes[edge[0]]['val'] or G.nodes[edge[1]]['val'] or
                G.nodes[edge[0]]['test'] or G.nodes[edge[1]]['test']):
            G[edge[0]][edge[1]]['train_removed'] = True
        else:
            G[edge[0]][edge[1]]['train_removed'] = False

    train_removed_count = 0
    for edge in G.edges():
        if G[edge[0]][edge[1]]['train_removed']:
            train_removed_count += 1
    print("Amount of train_removed edges: ", train_removed_count)

    if normalize and not feats is None:
        from sklearn.preprocessing import StandardScaler
        train_ids = np.array([id_map[n] for n in G.nodes(
        ) if not G.nodes[n]['val'] and not G.nodes[n]['test']])
        train_feats = feats[train_ids]
        scaler = StandardScaler()
        scaler.fit(train_feats)
        feats = scaler.transform(feats)

    if load_walks:
        with open(prefix + "-walks.txt") as fp:
            for line in fp:
                walks.append(map(conversion, line.split()))

    return G, feats, id_map, walks, class_map


def run_random_walks(G, nodes, num_walks=N_WALKS):
    pairs = []
    for count, node in enumerate(nodes):
        if G.degree(node) == 0:
            continue
        for i in range(num_walks):
            curr_node = node
            for j in range(WALK_LEN):
                next_node = random.choice(G.neighbors(curr_node))
                # self co-occurrences are useless
                if curr_node != node:
                    pairs.append((node, curr_node))
                curr_node = next_node
        if count % 1000 == 0:
            print("Done walks for", count, "nodes")
    return pairs


if __name__ == "__main__":
    """ Run random walks """
    graph_file = sys.argv[1]
    out_file = sys.argv[2]
    G_data = json.load(open(graph_file))
    G = json_graph.node_link_graph(G_data)
    nodes = [n for n in G.nodes() if not G.nodes[n]["val"]
             and not G.nodes[n]["test"]]
    G = G.subgraph(nodes)
    pairs = run_random_walks(G, nodes)
    with open(out_file, "w") as fp:
        fp.write("\n".join([str(p[0]) + "\t" + str(p[1]) for p in pairs]))
