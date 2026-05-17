"""GraphSAGESPARC supervised trainer.

This is a SPARC-driven variant of the original GraphSAGE supervised trainer.
The real graph is **not** used: it is replaced by a synthetic graph whose
edges are each node's top-K nearest neighbors in the SPARC embedding space.

Key differences vs. the original ``GraphSAGE`` module:

* Inputs are loaded from the SPARC run directory, i.e.
  ``SPARC/sparc_results/<dataset>/<split_name>_test<test_ratio>_seed<seed>/``,
  which provides ``embeddings.npy``, ``features.npy``, ``labels.npy``,
  ``train_mask.npy``, ``val_mask.npy``, ``test_mask.npy``.
* The networkx graph fed into ``NodeMinibatchIterator`` has only KNN edges:
  for every node we take the ``--sparc_topk`` (default 10) closest
  *training* nodes in SPARC embedding space (Minkowski/Euclidean distance).
  Train nodes are the only candidate neighbors -- val and test nodes are
  never anyone's neighbor (cold-start safety).
* Edges touching val/test are flagged ``train_removed=True`` so the
  training adjacency in ``construct_adj`` only sees train-train edges,
  while ``construct_test_adj`` keeps all KNN edges (so val/test nodes
  aggregate from their SPARC neighbors at evaluation time).
"""

from __future__ import division
from __future__ import print_function

import os
import sys

# ---------------------------------------------------------------------------
# Package path: allow ``python supervised_train.py`` from this directory.
# The parent of ``graphsage/`` (SPARC-SAGE) must be on sys.path.
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
_PACKAGE_ROOT = os.path.normpath(os.path.join(_SCRIPT_DIR, os.pardir))
if _PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, _PACKAGE_ROOT)

import time
import numpy as np
from graphsage.tf_compat import tf, flags, FLAGS
import sklearn
from sklearn import metrics
from sklearn.neighbors import NearestNeighbors

import networkx as nx
from sklearn.preprocessing import StandardScaler

from graphsage.supervised_models import SupervisedGraphsage
from graphsage.models import SAGEInfo
from graphsage.minibatch import NodeMinibatchIterator
from graphsage.neigh_samplers import UniformNeighborSampler

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# Default: SPARC/SPARC/sparc_results (three levels up from this file)
_DEFAULT_SPARC_RESULTS_ROOT = os.path.normpath(
    os.path.join(_SCRIPT_DIR, os.pardir, os.pardir, os.pardir, "sparc_results"))

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

# Settings
flags.DEFINE_boolean('log_device_placement', False,
                     """Whether to log device placement.""")
flags.DEFINE_string('model', 'graphsage_mean',
                    'model names. See README for possible values.')
flags.DEFINE_float('learning_rate', 0.01, 'initial learning rate.')
flags.DEFINE_string("model_size", "small",
                    "Can be big or small; model specific def'ns")
flags.DEFINE_string('dataset_name', '',
                    'dataset name (e.g., cora). Set programmatically by main().')

flags.DEFINE_integer('epochs', 10, 'number of epochs to train.')
flags.DEFINE_float('dropout', 0.0, 'dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 0.0,
                   'weight for l2 loss on embedding matrix.')
flags.DEFINE_integer('max_degree', 128, 'maximum node degree.')
flags.DEFINE_integer('samples_1', 25, 'number of samples in layer 1')
flags.DEFINE_integer('samples_2', 10, 'number of samples in layer 2')
flags.DEFINE_integer(
    'samples_3', 0, 'number of users samples in layer 3. (Only for mean model)')
flags.DEFINE_integer(
    'dim_1', 128, 'Size of output dim (final is 2x this, if using concat)')
flags.DEFINE_integer(
    'dim_2', 128, 'Size of output dim (final is 2x this, if using concat)')
flags.DEFINE_boolean('random_context', True,
                     'Whether to use random context or direct edges')
flags.DEFINE_integer('batch_size', 512, 'minibatch size.')
flags.DEFINE_boolean('sigmoid', False, 'whether to use sigmoid loss')
flags.DEFINE_integer(
    'identity_dim', 0, 'Set to positive value to use identity embedding features of that dimension. Default 0.')

flags.DEFINE_string('base_log_dir', '.',
                    'base directory for logging and saving embeddings')
flags.DEFINE_integer('validate_iter', 5000,
                     "how often to run a validation minibatch.")
flags.DEFINE_integer('validate_batch_size', 256,
                     "how many nodes per validation sample.")
flags.DEFINE_integer('gpu', 1, "which gpu to use.")
flags.DEFINE_integer('print_every', 5, "How often to print training info.")
flags.DEFINE_integer('max_total_steps', 10**10,
                     "Maximum total number of iterations")

# CLI flags for the benchmark runner
flags.DEFINE_string('cli_dataset', '', 'Dataset name for CLI mode (overrides hardcoded default)')
flags.DEFINE_integer('cli_seed', 42, 'Random seed for CLI mode')
flags.DEFINE_float('cli_test_ratio', 0.1, 'Test fraction; used to resolve the SPARC run dir')
flags.DEFINE_float('cli_val_ratio', 0.1, 'Val fraction; not used to find the SPARC dir but kept for parity')
flags.DEFINE_string('cli_split_name', 'random', 'Split strategy: random | low_degree | high_degree')

# SPARC-specific flags
flags.DEFINE_integer('sparc_topk', 10,
                     'Number of nearest SPARC neighbors per node (per-node degree in the synthetic KNN graph).')
flags.DEFINE_string('sparc_knn_metric', 'minkowski',
                    'Distance metric passed to sklearn NearestNeighbors.')
flags.DEFINE_string('sparc_results_root', '',
                    'Optional override for SPARC sparc_results root. '
                    'Defaults to SPARC_project/SPARC/sparc_results.')
flags.DEFINE_string('sparc_run', '',
                    'Optional override for the SPARC run sub-folder name '
                    '(default: <split_name>_test<test_ratio>_seed<seed>).')

os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.gpu)

GPU_MEM_FRACTION = 0.8

DEFAULT_SEED = 42
DEFAULT_TEST_RATIO = 0.1
DEFAULT_VAL_RATIO = 0.1

np.random.seed(DEFAULT_SEED)
tf.set_random_seed(DEFAULT_SEED)


# ---------------------------------------------------------------------------
# SPARC data loading + KNN graph construction
# ---------------------------------------------------------------------------

def _format_test_ratio(test_ratio):
    """Match the SPARC main.py run-dir formatting (e.g. 0.1, 0.03)."""
    s = "{}".format(float(test_ratio))
    return s


def resolve_sparc_run_dir(dataset, seed, test_ratio, split_name,
                          sparc_results_root="", sparc_run=""):
    """Find the SPARC run directory for the given experiment.

    Layout expected (matching ``SPARC/src/main.py`` ~ line 577):
        ``<sparc_results_root>/<dataset>/<split_name>_test<test_ratio>_seed<seed>/``

    A user-supplied ``sparc_run`` overrides the default folder name.
    """
    root = sparc_results_root or _DEFAULT_SPARC_RESULTS_ROOT
    if not os.path.isdir(root):
        raise FileNotFoundError(
            "SPARC results root does not exist: {}".format(root))

    if sparc_run:
        run_dir_name = sparc_run
    else:
        run_dir_name = "{}_test{}_seed{}".format(
            split_name, _format_test_ratio(test_ratio), seed)

    run_dir = os.path.normpath(os.path.join(root, dataset, run_dir_name))
    if not os.path.isdir(run_dir):
        # Useful diagnostic listing what IS available in the dataset dir.
        ds_dir = os.path.join(root, dataset)
        if os.path.isdir(ds_dir):
            available = sorted(os.listdir(ds_dir))
        else:
            available = []
        raise FileNotFoundError(
            "SPARC run directory not found: {}\n"
            "Tried: split_name={}, test_ratio={}, seed={}.\n"
            "Available runs under {}: {}".format(
                run_dir, split_name, _format_test_ratio(test_ratio), seed,
                ds_dir, available))
    return run_dir


def load_sparc_run_arrays(run_dir):
    """Load the arrays we need from a SPARC run directory."""
    def _np(name):
        path = os.path.join(run_dir, name)
        if not os.path.isfile(path):
            raise FileNotFoundError(
                "Missing required file in SPARC run dir: {}".format(path))
        return np.load(path, allow_pickle=False)

    embeddings = _np("embeddings.npy")
    features = _np("features.npy")
    labels = _np("labels.npy")
    train_mask = _np("train_mask.npy").astype(bool)
    val_mask = _np("val_mask.npy").astype(bool)
    test_mask = _np("test_mask.npy").astype(bool)
    return embeddings, features, labels, train_mask, val_mask, test_mask


def build_sparc_knn_graphsage_data(embeddings, features, labels,
                                   train_mask, val_mask, test_mask,
                                   topk=10,
                                   knn_metric="minkowski",
                                   normalize=True):
    """Build the ``(G, features, id_map, walks, class_map)`` tuple for
    ``NodeMinibatchIterator`` from SPARC outputs.

    The graph is purely synthetic: every node gets exactly ``topk`` outgoing
    KNN edges into the *train* node pool. Edges touching val/test nodes are
    marked ``train_removed=True`` so they are excluded from the training
    adjacency but included in ``test_adj`` (used at val/test eval time).
    """
    n = int(embeddings.shape[0])
    assert features.shape[0] == n, \
        "features and embeddings have different N ({} vs {})".format(
            features.shape[0], n)
    assert labels.shape[0] == n, \
        "labels and embeddings have different N ({} vs {})".format(
            labels.shape[0], n)
    assert train_mask.shape[0] == n and val_mask.shape[0] == n and \
        test_mask.shape[0] == n, "mask sizes inconsistent with N={}".format(n)

    train_idx = np.where(train_mask)[0].astype(np.int64)
    val_set = set(int(i) for i in np.where(val_mask)[0])
    test_set = set(int(i) for i in np.where(test_mask)[0])

    if train_idx.size == 0:
        raise ValueError("No training nodes available -- cannot build SPARC KNN graph.")

    # Need topk + 1 to handle self-match for query nodes that are themselves
    # train nodes (we drop the self entry below).
    n_neighbors = int(min(topk + 1, train_idx.size))
    if n_neighbors < 2:
        raise ValueError(
            "Not enough train candidates for KNN (have {}, need >=2).".format(
                train_idx.size))

    print("[SPARC-KNN] fitting NearestNeighbors: candidates={}, topk={}, metric={}".format(
        train_idx.size, topk, knn_metric))
    t0 = time.time()
    emb_f32 = np.ascontiguousarray(np.asarray(embeddings, dtype=np.float32))
    knn = NearestNeighbors(
        n_neighbors=n_neighbors, algorithm="auto",
        metric=knn_metric, n_jobs=-1)
    knn.fit(emb_f32[train_idx])
    try:
        _, local_indices = knn.kneighbors(emb_f32, n_jobs=-1)
    except TypeError:
        _, local_indices = knn.kneighbors(emb_f32)
    print("[SPARC-KNN] kneighbors done in {:.1f}s -> {}".format(
        time.time() - t0, local_indices.shape))

    # Map local (train-subset) indices back to global node indices.
    global_indices = train_idx[local_indices]  # (N, n_neighbors)

    # Drop self-matches and trim to exactly topk per node.
    rows_kept = np.empty((n, topk), dtype=np.int64)
    rng_self = np.arange(n)[:, None]
    for i in range(n):
        cand = global_indices[i]
        cand = cand[cand != i]
        if cand.size < topk:
            # Train pool was tiny; pad with re-sampled candidates so we always
            # emit exactly topk neighbours.
            extra = np.random.choice(cand, size=topk - cand.size, replace=True) \
                if cand.size > 0 else \
                np.random.choice(train_idx, size=topk, replace=True)
            cand = np.concatenate([cand, extra])
        rows_kept[i] = cand[:topk]
    del global_indices

    # ------------------------------------------------------------------ build G
    G = nx.Graph()
    for i in range(n):
        G.add_node(int(i),
                   val=(int(i) in val_set),
                   test=(int(i) in test_set))

    edges_added = 0
    for i in range(n):
        i_int = int(i)
        for j in rows_kept[i]:
            j_int = int(j)
            if j_int == i_int:
                continue
            if not G.has_edge(i_int, j_int):
                G.add_edge(i_int, j_int)
                edges_added += 1

    # train_removed flag: any edge touching a non-train node is excluded from
    # training-time adjacency but kept in test_adj for evaluation.
    for u, v in G.edges():
        if (u in val_set or v in val_set or u in test_set or v in test_set):
            G[u][v]['train_removed'] = True
        else:
            G[u][v]['train_removed'] = False

    id_map = {int(i): int(i) for i in range(n)}
    if labels.ndim == 1:
        class_map = {int(i): int(labels[i]) for i in range(n)}
    else:
        class_map = {int(i): labels[i].tolist() for i in range(n)}

    if normalize and features is not None:
        scaler = StandardScaler()
        scaler.fit(features[train_idx])
        features = scaler.transform(features)

    edges_train_removed = sum(1 for u, v in G.edges() if G[u][v]['train_removed'])
    print("[SPARC-KNN] graph built: {} nodes, {} edges (unique), "
          "{} train_removed | {} train, {} val, {} test".format(
              G.number_of_nodes(), G.number_of_edges(),
              edges_train_removed,
              int(train_mask.sum()), int(val_mask.sum()), int(test_mask.sum())))

    walks = []
    return G, features, id_map, walks, class_map


# ---------------------------------------------------------------------------
# Training utilities (unchanged from upstream GraphSAGE supervised_train)
# ---------------------------------------------------------------------------

def calc_f1(y_true, y_pred):
    """Returns micro-F1, macro-F1, and accuracy."""
    if not FLAGS.sigmoid:
        y_true = np.argmax(y_true, axis=1)
        y_pred = np.argmax(y_pred, axis=1)
    else:
        y_pred[y_pred > 0.5] = 1
        y_pred[y_pred <= 0.5] = 0
    mic_f1 = metrics.f1_score(y_true, y_pred, average="micro")
    mac_f1 = metrics.f1_score(y_true, y_pred, average="macro")
    acc = metrics.accuracy_score(y_true, y_pred)
    return mic_f1, mac_f1, acc


def evaluate(sess, model, minibatch_iter, size=None):
    t_test = time.time()
    feed_dict_val, labels = minibatch_iter.node_val_feed_dict(size)
    node_outs_val = sess.run([model.preds, model.loss],
                             feed_dict=feed_dict_val)
    mic, mac, acc = calc_f1(labels, node_outs_val[0])
    return node_outs_val[1], mic, mac, acc, (time.time() - t_test)


def log_dir():
    dataset_name = FLAGS.dataset_name or "unknown"
    log_dir = FLAGS.base_log_dir + "/sup-sparc-" + dataset_name
    log_dir += "/{model:s}_{model_size:s}_{lr:0.4f}_topk{topk:d}/".format(
        model=FLAGS.model,
        model_size=FLAGS.model_size,
        lr=FLAGS.learning_rate,
        topk=FLAGS.sparc_topk)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir


def incremental_evaluate(sess, model, minibatch_iter, size, test=False):
    t_test = time.time()
    finished = False
    val_losses = []
    val_preds = []
    labels = []
    iter_num = 0
    finished = False
    while not finished:
        feed_dict_val, batch_labels, finished, _ = minibatch_iter.incremental_node_val_feed_dict(
            size, iter_num, test=test
        )
        node_outs_val = sess.run([model.preds, model.loss],
                                 feed_dict=feed_dict_val)
        val_preds.append(node_outs_val[0])
        labels.append(batch_labels)
        val_losses.append(node_outs_val[1])
        iter_num += 1
    val_preds = np.vstack(val_preds)
    labels = np.vstack(labels)
    mic, mac, acc = calc_f1(labels, val_preds)
    return np.mean(val_losses), mic, mac, acc, (time.time() - t_test)


def incremental_evaluate_train(sess, model, minibatch_iter, size):
    """Evaluate on the full training set (all train nodes)."""
    t_test = time.time()
    finished = False
    train_losses = []
    train_preds = []
    labels = []
    iter_num = 0
    train_nodes = list(minibatch_iter.train_nodes)

    if len(train_nodes) == 0:
        raise ValueError("No training nodes found for full-train evaluation.")

    while not finished:
        start = iter_num * size
        end = min((iter_num + 1) * size, len(train_nodes))
        batch_nodes = train_nodes[start:end]
        if not batch_nodes:
            break
        feed_dict, batch_labels = minibatch_iter.batch_feed_dict(batch_nodes)
        node_outs_val = sess.run([model.preds, model.loss],
                                 feed_dict=feed_dict)
        train_preds.append(node_outs_val[0])
        labels.append(batch_labels)
        train_losses.append(node_outs_val[1])
        iter_num += 1
        finished = end >= len(train_nodes)

    train_preds = np.vstack(train_preds)
    labels = np.vstack(labels)
    mic, mac, acc = calc_f1(labels, train_preds)
    return np.mean(train_losses), mic, mac, acc, (time.time() - t_test)


def construct_placeholders(num_classes):
    placeholders = {
        'labels': tf.placeholder(tf.float32, shape=(None, num_classes), name='labels'),
        'batch': tf.placeholder(tf.int32, shape=(None), name='batch1'),
        'dropout': tf.placeholder_with_default(0., shape=(), name='dropout'),
        'batch_size': tf.placeholder(tf.int32, name='batch_size'),
    }
    return placeholders


def train(train_data, test_data=None):
    G = train_data[0]
    features = train_data[1]
    id_map = train_data[2]
    class_map = train_data[4]
    if isinstance(list(class_map.values())[0], list):
        num_classes = len(list(class_map.values())[0])
    else:
        num_classes = len(set(class_map.values()))

    if features is not None:
        features = np.vstack([features, np.zeros((features.shape[1],))])

    context_pairs = train_data[3] if FLAGS.random_context else None
    placeholders = construct_placeholders(num_classes)
    minibatch = NodeMinibatchIterator(G,
                                      id_map,
                                      placeholders,
                                      class_map,
                                      num_classes,
                                      batch_size=FLAGS.batch_size,
                                      max_degree=FLAGS.max_degree,
                                      context_pairs=context_pairs)
    adj_info_ph = tf.placeholder(tf.int32, shape=minibatch.adj.shape)
    adj_info = tf.Variable(adj_info_ph, trainable=False, name="adj_info")

    if FLAGS.model == 'graphsage_mean':
        sampler = UniformNeighborSampler(adj_info)
        if FLAGS.samples_3 != 0:
            layer_infos = [SAGEInfo("node", sampler, FLAGS.samples_1, FLAGS.dim_1),
                           SAGEInfo("node", sampler, FLAGS.samples_2, FLAGS.dim_2),
                           SAGEInfo("node", sampler, FLAGS.samples_3, FLAGS.dim_2)]
        elif FLAGS.samples_2 != 0:
            layer_infos = [SAGEInfo("node", sampler, FLAGS.samples_1, FLAGS.dim_1),
                           SAGEInfo("node", sampler, FLAGS.samples_2, FLAGS.dim_2)]
        else:
            layer_infos = [
                SAGEInfo("node", sampler, FLAGS.samples_1, FLAGS.dim_1)]

        model = SupervisedGraphsage(num_classes, placeholders,
                                    features, adj_info, minibatch.deg,
                                    layer_infos,
                                    model_size=FLAGS.model_size,
                                    sigmoid_loss=FLAGS.sigmoid,
                                    identity_dim=FLAGS.identity_dim,
                                    logging=True)

    elif FLAGS.model == 'gcn':
        sampler = UniformNeighborSampler(adj_info)
        layer_infos = [SAGEInfo("node", sampler, FLAGS.samples_1, 2 * FLAGS.dim_1),
                       SAGEInfo("node", sampler, FLAGS.samples_2, 2 * FLAGS.dim_2)]
        model = SupervisedGraphsage(num_classes, placeholders,
                                    features, adj_info, minibatch.deg,
                                    layer_infos=layer_infos,
                                    aggregator_type="gcn",
                                    model_size=FLAGS.model_size, concat=False,
                                    sigmoid_loss=FLAGS.sigmoid,
                                    identity_dim=FLAGS.identity_dim,
                                    logging=True)

    elif FLAGS.model == 'graphsage_seq':
        sampler = UniformNeighborSampler(adj_info)
        layer_infos = [SAGEInfo("node", sampler, FLAGS.samples_1, FLAGS.dim_1),
                       SAGEInfo("node", sampler, FLAGS.samples_2, FLAGS.dim_2)]
        model = SupervisedGraphsage(num_classes, placeholders,
                                    features, adj_info, minibatch.deg,
                                    layer_infos=layer_infos,
                                    aggregator_type="seq",
                                    model_size=FLAGS.model_size,
                                    sigmoid_loss=FLAGS.sigmoid,
                                    identity_dim=FLAGS.identity_dim,
                                    logging=True)

    elif FLAGS.model == 'graphsage_maxpool':
        sampler = UniformNeighborSampler(adj_info)
        layer_infos = [SAGEInfo("node", sampler, FLAGS.samples_1, FLAGS.dim_1),
                       SAGEInfo("node", sampler, FLAGS.samples_2, FLAGS.dim_2)]
        model = SupervisedGraphsage(num_classes, placeholders,
                                    features, adj_info, minibatch.deg,
                                    layer_infos=layer_infos,
                                    aggregator_type="maxpool",
                                    model_size=FLAGS.model_size,
                                    sigmoid_loss=FLAGS.sigmoid,
                                    identity_dim=FLAGS.identity_dim,
                                    logging=True)

    elif FLAGS.model == 'graphsage_meanpool':
        sampler = UniformNeighborSampler(adj_info)
        layer_infos = [SAGEInfo("node", sampler, FLAGS.samples_1, FLAGS.dim_1),
                       SAGEInfo("node", sampler, FLAGS.samples_2, FLAGS.dim_2)]
        model = SupervisedGraphsage(num_classes, placeholders,
                                    features, adj_info, minibatch.deg,
                                    layer_infos=layer_infos,
                                    aggregator_type="meanpool",
                                    model_size=FLAGS.model_size,
                                    sigmoid_loss=FLAGS.sigmoid,
                                    identity_dim=FLAGS.identity_dim,
                                    logging=True)

    else:
        raise Exception('Error: model name unrecognized.')

    config = tf.ConfigProto(log_device_placement=FLAGS.log_device_placement)
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True

    sess = tf.Session(config=config)
    merged = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(log_dir(), sess.graph)

    sess.run(tf.global_variables_initializer(),
             feed_dict={adj_info_ph: minibatch.adj})

    total_steps = 0
    avg_time = 0.0
    epoch_val_costs = []

    train_adj_info = tf.assign(adj_info, minibatch.adj)
    val_adj_info = tf.assign(adj_info, minibatch.test_adj)

    for epoch in range(FLAGS.epochs):
        minibatch.shuffle()

        iter = 0
        print('Epoch: %04d' % (epoch + 1))
        epoch_val_costs.append(0)
        while not minibatch.end():
            feed_dict, labels = minibatch.next_minibatch_feed_dict()
            feed_dict.update({placeholders['dropout']: FLAGS.dropout})

            t = time.time()
            outs = sess.run([merged, model.opt_op, model.loss,
                             model.preds], feed_dict=feed_dict)
            train_cost = outs[2]

            if iter % FLAGS.validate_iter == 0:
                sess.run(val_adj_info.op)
                if FLAGS.validate_batch_size == -1:
                    val_cost, val_f1_mic, val_f1_mac, val_acc, duration = incremental_evaluate(
                        sess, model, minibatch, FLAGS.batch_size)
                else:
                    val_cost, val_f1_mic, val_f1_mac, val_acc, duration = evaluate(
                        sess, model, minibatch, FLAGS.validate_batch_size)
                sess.run(train_adj_info.op)
                epoch_val_costs[-1] += val_cost

            if total_steps % FLAGS.print_every == 0:
                summary_writer.add_summary(outs[0], total_steps)

            avg_time = (avg_time * total_steps +
                        time.time() - t) / (total_steps + 1)

            if total_steps % FLAGS.print_every == 0:
                train_f1_mic, train_f1_mac, train_acc = calc_f1(
                    labels, outs[-1])
                print("Iter:", '%04d' % iter,
                      "train_loss=", "{:.5f}".format(train_cost),
                      "train_f1_mic=", "{:.5f}".format(train_f1_mic),
                      "train_f1_mac=", "{:.5f}".format(train_f1_mac),
                      "train_acc=", "{:.5f}".format(train_acc),
                      "val_loss=", "{:.5f}".format(val_cost),
                      "val_f1_mic=", "{:.5f}".format(val_f1_mic),
                      "val_f1_mac=", "{:.5f}".format(val_f1_mac),
                      "val_acc=", "{:.5f}".format(val_acc),
                      "time=", "{:.5f}".format(avg_time))

            iter += 1
            total_steps += 1

            if total_steps > FLAGS.max_total_steps:
                break

        if total_steps > FLAGS.max_total_steps:
            break

    print("Optimization Finished!")

    sess.run(train_adj_info.op)
    train_cost, train_f1_mic, train_f1_mac, train_acc, train_duration = incremental_evaluate_train(
        sess, model, minibatch, FLAGS.batch_size
    )
    print("Full training stats:",
          "loss=", "{:.5f}".format(train_cost),
          "f1_micro=", "{:.5f}".format(train_f1_mic),
          "f1_macro=", "{:.5f}".format(train_f1_mac),
          "acc=", "{:.5f}".format(train_acc),
          "time=", "{:.5f}".format(train_duration))
    with open(log_dir() + "train_stats.txt", "w") as fp:
        fp.write("loss={:.5f} f1_micro={:.5f} f1_macro={:.5f} acc={:.5f} time={:.5f}".
                 format(train_cost, train_f1_mic, train_f1_mac, train_acc, train_duration))

    sess.run(val_adj_info.op)
    val_cost_final, val_f1_mic_final, val_f1_mac_final, val_acc_final, val_duration = \
        incremental_evaluate(sess, model, minibatch, FLAGS.batch_size)
    print("Full validation stats:",
          "loss=", "{:.5f}".format(val_cost_final),
          "f1_micro=", "{:.5f}".format(val_f1_mic_final),
          "f1_macro=", "{:.5f}".format(val_f1_mac_final),
          "acc=", "{:.5f}".format(val_acc_final),
          "time=", "{:.5f}".format(val_duration))
    with open(log_dir() + "val_stats.txt", "w") as fp:
        fp.write("loss={:.5f} f1_micro={:.5f} f1_macro={:.5f} acc={:.5f} time={:.5f}".
                 format(val_cost_final, val_f1_mic_final, val_f1_mac_final, val_acc_final, val_duration))

    sess.run(val_adj_info.op)
    print("Writing test set stats to file (don't peak!)")
    test_cost, test_f1_mic, test_f1_mac, test_acc, test_duration = \
        incremental_evaluate(sess, model, minibatch,
                             FLAGS.batch_size, test=True)
    print("Full test stats:",
          "loss=", "{:.5f}".format(test_cost),
          "f1_micro=", "{:.5f}".format(test_f1_mic),
          "f1_macro=", "{:.5f}".format(test_f1_mac),
          "acc=", "{:.5f}".format(test_acc),
          "time=", "{:.5f}".format(test_duration))

    dataset_name = FLAGS.dataset_name or "unknown"
    test_results_file = log_dir() + "{}_test_stats.txt".format(dataset_name)
    with open(test_results_file, "w") as fp:
        fp.write("loss={:.5f} f1_micro={:.5f} f1_macro={:.5f} acc={:.5f}".
                 format(test_cost, test_f1_mic, test_f1_mac, test_acc))
    print("Test results saved to: {}".format(test_results_file))

    with open(log_dir() + "test_stats.txt", "w") as fp:
        fp.write("loss={:.5f} f1_micro={:.5f} f1_macro={:.5f} acc={:.5f}".
                 format(test_cost, test_f1_mic, test_f1_mac, test_acc))

    sess.close()

    return {
        "train_acc": float(train_acc),
        "train_f1_mic": float(train_f1_mic),
        "train_f1_mac": float(train_f1_mac),
        "val_acc": float(val_acc_final),
        "val_f1_mic": float(val_f1_mic_final),
        "val_f1_mac": float(val_f1_mac_final),
        "test_acc": float(test_acc),
        "test_f1_mic": float(test_f1_mic),
        "test_f1_mac": float(test_f1_mac),
        "lr_test_acc": None,
        "lr_val_acc": None,
    }


def main(dataset, seed=DEFAULT_SEED, test_ratio=DEFAULT_TEST_RATIO,
         val_ratio=DEFAULT_VAL_RATIO, split_name="random",
         sparc_topk=None, sparc_results_root=None, sparc_run=None,
         knn_metric=None):
    """Run a SPARC-KNN GraphSAGE experiment.

    Resolves a SPARC run dir for the given (dataset, split_name, test_ratio,
    seed), loads its embeddings + features + masks, and trains GraphSAGE on a
    synthetic top-K KNN graph (no real graph used). ``val_ratio`` is accepted
    only for parity with the original GraphSAGE main; the actual val/test
    masks come from the SPARC run dir.
    """
    tf.reset_default_graph()
    np.random.seed(seed)
    tf.set_random_seed(seed)

    FLAGS.dataset_name = dataset

    if sparc_topk is None:
        sparc_topk = FLAGS.sparc_topk
    if sparc_results_root is None:
        sparc_results_root = FLAGS.sparc_results_root or _DEFAULT_SPARC_RESULTS_ROOT
    if sparc_run is None:
        sparc_run = FLAGS.sparc_run
    if knn_metric is None:
        knn_metric = FLAGS.sparc_knn_metric

    print("=" * 60)
    print("[GraphSAGESPARC] dataset={}, split={}, test_ratio={}, seed={}, topk={}, "
          "metric={}".format(dataset, split_name, test_ratio, seed,
                             sparc_topk, knn_metric))
    print("=" * 60)

    run_dir = resolve_sparc_run_dir(
        dataset=dataset, seed=seed, test_ratio=test_ratio,
        split_name=split_name, sparc_results_root=sparc_results_root,
        sparc_run=sparc_run)
    print("[SPARC] run dir: {}".format(run_dir))

    embeddings, features, labels, train_mask, val_mask, test_mask = \
        load_sparc_run_arrays(run_dir)
    print("[SPARC] N={}, emb_dim={}, feat_dim={}, classes={} | "
          "|train|={}, |val|={}, |test|={}".format(
              embeddings.shape[0], embeddings.shape[1], features.shape[1],
              int(np.max(labels)) + 1 if labels.ndim == 1 else labels.shape[1],
              int(train_mask.sum()), int(val_mask.sum()), int(test_mask.sum())))

    train_data = build_sparc_knn_graphsage_data(
        embeddings=embeddings, features=features, labels=labels,
        train_mask=train_mask, val_mask=val_mask, test_mask=test_mask,
        topk=sparc_topk, knn_metric=knn_metric, normalize=True,
    )

    print("Done loading training data — starting training..")
    results = train(train_data)
    return results


if __name__ == '__main__':
    import json as _json
    if FLAGS.cli_dataset:
        _result = main(
            dataset=FLAGS.cli_dataset,
            seed=FLAGS.cli_seed,
            test_ratio=FLAGS.cli_test_ratio,
            val_ratio=FLAGS.cli_val_ratio,
            split_name=FLAGS.cli_split_name,
        )
    else:
        _result = main(
            dataset="cora",
            seed=DEFAULT_SEED,
            test_ratio=DEFAULT_TEST_RATIO,
            val_ratio=DEFAULT_VAL_RATIO,
            split_name="random",
        )
    if _result:
        _safe = {k: v for k, v in _result.items() if v is not None}
        print("__BENCHMARK_RESULT__", _json.dumps(_safe))
