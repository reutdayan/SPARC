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

"""Main script for training GCN models."""


import time
import numpy as np
import models

import partition_utils
import tensorflow.compat.v1 as tf
import scipy.sparse as sp
import utils
import networkx as nx
import os

# SPARC_project/data/graphsage/<dataset>/ (same layout as SPARC_project/data/load_data.py).
_GRAPHSAGE_PREFIX = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 '..', '..', '..', 'data', 'graphsage'))
_SPARC_RESULTS_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 '..', '..', 'sparc_results'))

tf.logging.set_verbosity(tf.logging.INFO)
# Set random seed
seed = 42
np.random.seed(seed)

# Settings
flags = tf.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('save_name', './mymodel.ckpt', 'Path for saving model')
flags.DEFINE_string('dataset', 'ppi', 'Dataset string.')
flags.DEFINE_string(
    'data_prefix', _GRAPHSAGE_PREFIX,
    'GraphSAGE root directory (<prefix>/<dataset>/<dataset>-G.json). '
    'Ignored when --use_sparc_only.')
flags.DEFINE_string(
    'sparc_results_root', _SPARC_RESULTS_ROOT,
    'SPARC output root (<root>/<dataset>/<run>/embeddings.npy). '
    'If that run folder also contains train_mask.npy, val_mask.npy, '
    'test_mask.npy (written by SPARC main.py), splits are loaded from there '
    'instead of GraphSAGE -G.json val/test flags.')
flags.DEFINE_string(
    'sparc_run', '',
    'Run folder under sparc_results/<dataset>/ (e.g. random_test0.03_seed42). '
    'Empty: resolve via --split_name / --test_ratio / --seed (SPARC main.py layout), '
    'then auto-pick if exactly one run exists.')
flags.DEFINE_string(
    'split_name', 'random',
    'Split strategy for default run dir when --sparc_run is empty (matches main.py).')
flags.DEFINE_float(
    'test_ratio', 0.10,
    'Test fraction for default run dir when --sparc_run is empty (matches main.py).')
flags.DEFINE_integer(
    'seed', 42,
    'Random seed for default run dir when --sparc_run is empty (matches main.py).')
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 75, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 2048, 'Number of units in hidden layer 1.')
flags.DEFINE_float('dropout', 0.2, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 0,
                   'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping', 1000,
                     'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('num_clusters', 50, 'Number of clusters.')
flags.DEFINE_integer('bsize', 1, 'Number of clusters for each batch.')
flags.DEFINE_integer('num_clusters_val', 5,
                     'Number of clusters for validation.')
flags.DEFINE_integer('num_clusters_test', 1, 'Number of clusters for test.')
flags.DEFINE_integer('num_layers', 5, 'Number of GCN layers.')
flags.DEFINE_float(
    'diag_lambda', 1,
    'A positive number for diagonal enhancement, -1 indicates normalization without diagonal enhancement'
)
flags.DEFINE_bool(
    'multilabel', False,
    'If True: sigmoid + multilabel accuracy (use for PPI). '
    'If False: softmax + argmax accuracy for one-hot multiclass (citation graphs). '
    'Running multiclass with multilabel=True inflates accuracy.')
flags.DEFINE_bool('layernorm', True, 'Whether to use layer normalization.')
flags.DEFINE_bool(
    'precalc', False,
    'Whether to pre-calculate the first layer (AX preprocessing).')
flags.DEFINE_bool('validation', True,
                  'Print validation accuracy after each epoch.')
flags.DEFINE_bool(
    'use_sparc_only', False,
    'Load features, labels, masks, adjacency from sparc_results run dir only '
    '(SPARC main.py artifacts). No GraphSAGE JSON/feats.')
flags.DEFINE_bool(
    'random_batches', False,
    'Minibatch train/val/test by random node shards (local subgraph per batch) '
    'instead of METIS graph partitioning.')
flags.DEFINE_integer(
    'random_batch_size', 256,
    'Nodes per train minibatch when --random_batches is set; also used to shard '
    'val/test into split-only minibatches (inductive eval, no train nodes in conv).')
flags.DEFINE_bool(
    'adj_identity', False,
    'Replace train/full adjacency with the identity matrix after loading data '
    '(no neighborhood aggregation; useful with --random_batches to remove graph use).')


def load_data(data_prefix, dataset_str, precalc,
              sparc_results_root=None, sparc_run=None, use_sparc_only=False,
              split_name='random', test_ratio=0.10, seed=42):
    """Return the required data formats for GCN models.

    Rows outside ``train_mask`` are zeroed in **training** feature and spectral tensors so
    the loss never sees val/test node coordinates; val/test phases still use full
    ``test_feats`` / ``test_eigen_vecs``. Values on train rows are unchanged (they may
    still encode global structure from how SPARC built ``embeddings.npy``).
    """
    if use_sparc_only:
        (num_data, train_adj, full_adj, feats, train_feats, test_feats, labels,
         train_data, val_data,
         test_data, train_eigen_vecs, test_eigen_vecs) = utils.load_sparc_run_bundle(
             sparc_results_root, dataset_str, sparc_run=sparc_run,
             split_name=split_name, test_ratio=test_ratio, seed=seed)
    else:
        (num_data, train_adj, full_adj, feats, train_feats, test_feats, labels,
         train_data, val_data,
         test_data, train_eigen_vecs, test_eigen_vecs) = utils.load_graphsage_data(
             data_prefix, dataset_str,
             sparc_results_root=sparc_results_root,
             sparc_run=sparc_run,
             split_name=split_name, test_ratio=test_ratio, seed=seed)
    visible_data = train_data

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_data, :] = labels[train_data, :]
    y_val[val_data, :] = labels[val_data, :]
    y_test[test_data, :] = labels[test_data, :]

    train_mask = utils.sample_mask(train_data, labels.shape[0])
    val_mask = utils.sample_mask(val_data, labels.shape[0])
    test_mask = utils.sample_mask(test_data, labels.shape[0])

    # Cold-start safe inputs for the training objective: zero non-train rows so
    # optimizer/SGD never sees SPARC embeddings or raw features at val/test nodes
    # (message passing still uses train_adj; val/test use full tensors below).
    feats_full = np.asarray(feats, dtype=np.float32)
    feats_train_only = feats_full.copy()
    feats_train_only[~train_mask] = 0.0
    spectral_full = np.asarray(train_eigen_vecs, dtype=np.float32).copy()
    spectral_train_only = spectral_full.copy()
    spectral_train_only[~train_mask] = 0.0
    train_eigen_vecs = spectral_train_only
    test_eigen_vecs = spectral_full

    if precalc:
        train_feats = train_adj.dot(feats_train_only)
        train_feats = np.hstack((train_feats, feats_train_only))
        test_feats = full_adj.dot(feats_full)
        test_feats = np.hstack((test_feats, feats_full))
    else:
        train_feats = feats_train_only
        test_feats = feats_full

    tf.logging.info(
        'Train-only masking: spectral and feature rows outside train_mask set to zero '
        'for training tensors; val/test evaluation uses full features/embeddings.')

    return (train_adj, full_adj, train_feats, test_feats, y_train, y_val, y_test,
            train_mask, val_mask, test_mask, train_data, val_data, test_data, train_eigen_vecs, test_eigen_vecs,
            num_data, visible_data)


# Define model evaluation function
def evaluate(sess, model, val_features_batches, val_eigen_vec_batches, val_support_batches,
             y_val_batches, val_mask_batches, val_data, placeholders):
    """evaluate GCN model."""
    total_pred = []
    total_lab = []
    total_loss = 0.0
    total_acc = 0.0
    total_count = 0

    num_batches = len(val_features_batches)
    for i in range(num_batches):
        features_b = val_features_batches[i]
        eigen_vec_b = val_eigen_vec_batches[i]
        support_b = val_support_batches[i]
        y_val_b = y_val_batches[i]
        val_mask_b = val_mask_batches[i]
        num_data_b = int(np.sum(val_mask_b))
        if num_data_b == 0:
            continue
        feed_dict = utils.construct_feed_dict(features_b, eigen_vec_b, support_b, y_val_b,
                                              val_mask_b, placeholders)
        outs = sess.run([model.loss, model.accuracy, model.outputs],
                        feed_dict=feed_dict)

        total_pred.append(outs[2][val_mask_b])
        total_lab.append(y_val_b[val_mask_b])
        total_loss += outs[0] * num_data_b
        total_acc += outs[1] * num_data_b
        total_count += num_data_b

    if not total_pred:
        tf.logging.warning(
            'evaluate: no nodes matched the mask in any batch (often empty val/test '
            'split in the graph JSON). Returning zero metrics.')
        return 0.0, 0.0, 0.0, 0.0

    total_pred = np.vstack(total_pred)
    total_lab = np.vstack(total_lab)
    denom = total_count if total_count > 0 else max(len(val_data), 1)
    loss = total_loss / denom
    acc = total_acc / denom

    micro, macro = utils.calc_f1(total_pred, total_lab, FLAGS.multilabel)
    return loss, acc, micro, macro


def main(unused_argv):
    """Main function for running experiments."""
    # Load data
    (train_adj, full_adj, train_feats, test_feats, y_train, y_val, y_test,
     train_mask, val_mask, test_mask, _, val_data, test_data, train_eigen_vecs, test_eigen_vecs, num_data,
     visible_data) = load_data(
         FLAGS.data_prefix, FLAGS.dataset, FLAGS.precalc,
         sparc_results_root=FLAGS.sparc_results_root,
         sparc_run=(FLAGS.sparc_run or None),
         use_sparc_only=FLAGS.use_sparc_only,
         split_name=FLAGS.split_name,
         test_ratio=FLAGS.test_ratio,
         seed=FLAGS.seed)

    if FLAGS.use_sparc_only:
        tf.logging.info('use_sparc_only: graph/feats/labels loaded from SPARC run dir only.')

    if FLAGS.adj_identity:
        train_adj = sp.eye(num_data, dtype=np.float32, format='csr')
        full_adj = train_adj.copy()
        tf.logging.info('adj_identity: train and full adjacency set to identity (no edges).')

    run_validation = FLAGS.validation and len(val_data) > 0
    if FLAGS.validation and len(val_data) == 0:
        tf.logging.warning(
            'Graph has no validation nodes (val split empty). Skipping val metrics '
            'and validation-based early stopping.')

    if FLAGS.random_batches and FLAGS.bsize > 1:
        tf.logging.warning(
            'random_batches: ignoring --bsize > 1 (training uses random minibatches).')

    features_batches = eigen_vecs_batches = support_batches = None
    y_train_batches = train_mask_batches = None
    idx_parts = []

    # Val/test must never share a minibatch with train nodes: full-graph METIS
    # partitions mixed splits and lets conv propagate from train → val/test (leak).
    rng_val = np.random.RandomState(seed)
    (_, val_features_batches, val_eigen_vecs_batches, val_support_batches, y_val_batches,
     val_mask_batches) = utils.preprocess_random_batches(
         full_adj, test_feats, test_eigen_vecs, y_val, val_data,
         FLAGS.random_batch_size, FLAGS.diag_lambda, rng=rng_val)
    rng_te = np.random.RandomState(seed + 1)
    (_, test_features_batches, test_eigen_vecs_batches, test_support_batches, y_test_batches,
     test_mask_batches) = utils.preprocess_random_batches(
         full_adj, test_feats, test_eigen_vecs, y_test, test_data,
         FLAGS.random_batch_size, FLAGS.diag_lambda, rng=rng_te)
    tf.logging.info(
        'Inductive val/test batches: nodes from one split only; '
        'support is adj restricted to that minibatch.')

    if FLAGS.random_batches:
        parts = []
    else:
        # Train-only METIS / graph partitioning (train_adj, visible_data).
        if FLAGS.bsize > 1:
            _, parts = partition_utils.partition_graph(train_adj, visible_data,
                                                       FLAGS.num_clusters)
            parts = [np.array(pt) for pt in parts]
        else:
            (parts, features_batches, eigen_vecs_batches, support_batches, y_train_batches,
             train_mask_batches) = utils.preprocess(train_adj, train_feats, train_eigen_vecs, y_train,
                                                    train_mask, visible_data,
                                                    FLAGS.num_clusters,
                                                    FLAGS.diag_lambda)
        idx_parts = list(range(len(parts)))

    # print("connected components: ", nx.number_connected_components(nx.Graph(train_adj)))
    # # print size of each componnenet
    # print([len(c) for c in sorted(nx.connected_components(nx.Graph(train_adj)), key=len, reverse=True)])

    # Some preprocessing
    model_func = models.GCN

    # Define placeholders
    placeholders = {
        'support':
            tf.sparse_placeholder(tf.float32),
        'features':
            tf.placeholder(tf.float32),
        'eigen_vecs':
            tf.placeholder(tf.float32),
        'labels':
            tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
        'labels_mask':
            tf.placeholder(tf.int32),
        'dropout':
            tf.placeholder_with_default(0., shape=()),
        'num_features_nonzero':
            tf.placeholder(tf.int32)  # helper variable for sparse dropout
    }

    tf.logging.info(
        'Model multilabel=%s (%s).',
        FLAGS.multilabel,
        'sigmoid' if FLAGS.multilabel else 'softmax multiclass')

    # Create model
    model = model_func(
        placeholders,
        input_dim=test_feats.shape[1],
        u_dim=test_eigen_vecs.shape[1],
        logging=True,
        multilabel=FLAGS.multilabel,
        norm=FLAGS.layernorm,
        precalc=FLAGS.precalc,
        num_layers=FLAGS.num_layers)

    # Adjust 'model' according to your model naming convention
    pretrained_model_name = '{}{}'.format(FLAGS.dataset, 'model')
    pretrained_model_path = './{}.meta'.format(pretrained_model_name)

    # Initialize session
    sess = tf.Session()
    tf.set_random_seed(seed)
    saver = tf.train.Saver()

    if os.path.exists(pretrained_model_path):
        saver = tf.train.import_meta_graph(pretrained_model_path)
        saver.restore(sess, pretrained_model_name)
        tf.logging.info(
            'Pre-trained model loaded from {}'.format(pretrained_model_path))
    else:
        tf.logging.warning(
            'Pre-trained model not found at {}. Training from scratch...'.format(pretrained_model_path))
        # Init variables
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()

    cost_val = []
    total_training_time = 0.0
    # Train model
    train_acc_g = []
    for epoch in range(FLAGS.epochs):
        if FLAGS.random_batches:
            rng_e = np.random.RandomState(seed + epoch)
            (_, features_batches, eigen_vecs_batches, support_batches, y_train_batches,
             train_mask_batches) = utils.preprocess_random_batches(
                 train_adj, train_feats, train_eigen_vecs, y_train, visible_data,
                 FLAGS.random_batch_size, FLAGS.diag_lambda, rng=rng_e)
            idx_parts = list(range(len(features_batches)))

        t = time.time()
        np.random.shuffle(idx_parts)
        if FLAGS.bsize > 1 and not FLAGS.random_batches:
            (features_batches, eigen_vecs_batches, support_batches, y_train_batches,
             train_mask_batches) = utils.preprocess_multicluster(
                 train_adj, parts, train_feats, train_eigen_vecs, y_train, train_mask,
                 FLAGS.num_clusters, FLAGS.bsize, FLAGS.diag_lambda)
            for pid in range(len(features_batches)):
                # Use preprocessed batch data
                features_b = features_batches[pid]
                eigen_vecs_b = eigen_vecs_batches[pid]
                support_b = support_batches[pid]
                y_train_b = y_train_batches[pid]
                train_mask_b = train_mask_batches[pid]
                # print('batch size', len(y_train_b))
                # Construct feed dictionary
                feed_dict = utils.construct_feed_dict(features_b, eigen_vecs_b, support_b, y_train_b,
                                                      train_mask_b, placeholders)
                feed_dict.update({placeholders['dropout']: FLAGS.dropout})
                # Training step
                outs = sess.run([model.opt_op, model.loss, model.accuracy],
                                feed_dict=feed_dict)

        else:
            if not FLAGS.random_batches:
                np.random.shuffle(idx_parts)
            for pid in idx_parts:
                # Use preprocessed batch data
                features_b = features_batches[pid]
                eigen_vecs_b = eigen_vecs_batches[pid]
                support_b = support_batches[pid]
                y_train_b = y_train_batches[pid]
                train_mask_b = train_mask_batches[pid]
                # Construct feed dictionary
                feed_dict = utils.construct_feed_dict(features_b, eigen_vecs_b, support_b, y_train_b,
                                                      train_mask_b, placeholders)
                feed_dict.update({placeholders['dropout']: FLAGS.dropout})
                # Training step
                outs = sess.run([model.opt_op, model.loss, model.accuracy],
                                feed_dict=feed_dict)
        total_training_time += time.time() - t
        print_str = 'Epoch: %04d ' % (epoch + 1) + 'training time: {:.5f} '.format(
            total_training_time) + 'train_acc= {:.5f}'.format(outs[2])
        train_acc_g.append(outs[2])

        cost, acc, micro, macro = evaluate(sess, model, features_batches, eigen_vecs_batches,
                                           support_batches, y_train_batches,
                                           train_mask_batches, visible_data,
                                           placeholders)

        print_str += 'acc= {:.5f} '.format(
            acc) + 'mi F1= {:.5f} ma F1= {:.5f} '.format(micro, macro)
        # Validation
        if run_validation:
            cost, acc, micro, macro = evaluate(sess, model, val_features_batches, val_eigen_vecs_batches,
                                               val_support_batches, y_val_batches,
                                               val_mask_batches, val_data,
                                               placeholders)
            cost_val.append(cost)
            print_str += 'val_acc= {:.5f} '.format(
                acc) + 'mi F1= {:.5f} ma F1= {:.5f} '.format(micro, macro)

        tf.logging.info(print_str)

        if (run_validation and len(cost_val) > FLAGS.early_stopping and epoch > FLAGS.early_stopping
                and cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping + 1):-1])):
            tf.logging.info('Early stopping...')
            break

    tf.logging.info('Optimization Finished!')

    # Save model
    # saver.save(sess, FLAGS.save_name)

    # # Save train acc graph
    # with open('/content/drive/MyDrive/ms/cluster_gcn/spectral.txt', 'w') as file:
    #   for item in train_acc_g:
    #       file.write('%s\n' % item)

    test_cost, test_acc, micro, macro = evaluate(
        sess, model, test_features_batches, test_eigen_vecs_batches, test_support_batches,
        y_test_batches, test_mask_batches, test_data, placeholders)
    print_str = 'Test set results: ' + 'cost= {:.5f} '.format(
        test_cost) + 'accuracy= {:.5f} '.format(
        test_acc) + 'mi F1= {:.5f} ma F1= {:.5f}'.format(micro, macro)
    tf.logging.info(print_str)

    # # Load model (using CPU for inference)
    # with tf.device('/cpu:0'):
    #   sess_cpu = tf.Session(config=tf.ConfigProto(device_count={'GPU': 0}))
    #   sess_cpu.run(tf.global_variables_initializer())
    #   saver = tf.train.Saver()
    #   saver.restore(sess_cpu, FLAGS.save_name)
    #   # Testing
    #   test_cost, test_acc, micro, macro = evaluate(
    #       sess_cpu, model, test_features_batches, test_support_batches,
    #       y_test_batches, test_mask_batches, test_data, placeholders)
    #   print_str = 'Test set results: ' + 'cost= {:.5f} '.format(
    #       test_cost) + 'accuracy= {:.5f} '.format(
    #           test_acc) + 'mi F1= {:.5f} ma F1= {:.5f}'.format(micro, macro)
    #   tf.logging.info(print_str)


if __name__ == '__main__':
    tf.compat.v1.disable_eager_execution()
    tf.app.run(main)
