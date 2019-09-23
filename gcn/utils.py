import networkx as nx
import numpy as np
import scipy.sparse as sp
import tensorflow as tf
import _pickle as pickle


def load_data():
    g = nx.read_edgelist('data/yeast.edgelist')
    adj = nx.adjacency_matrix(g)

    num_nodes = adj.shape[0]
    num_edges = adj.sum()
    # Featureless
    features = sparse_to_tuple(sp.identity(num_nodes))
    num_features = features[2][1]
    features_nonzero = features[1].shape[0]

    with open('data/adj_train.npz', 'rb') as f:
        local_adj_train = sp.load_npz(f)

    with open('data/train_edges.npy', 'rb') as f:
        local_train_edges = np.load(f)
        print("train_edges.shape: {}\n".format(local_train_edges.shape))

    with open('data/val_edges.npy', 'rb') as f:
        local_val_edges = np.load(f)
        print("train_edges.shape: {}\n".format(local_val_edges.shape))

    with open('data/val_edges_false.p', 'rb') as f:
        local_val_edges_false = pickle.load(f)
        print("val_edges_false len: {}\n".format(len(local_val_edges_false)))

    with open('data/test_edges.npy', 'rb') as f:
        local_test_edges = np.load(f)
        print("test_edges.shape: {}\n".format(local_test_edges.shape))

    with open('data/val_edges_false.p', 'rb') as f:
        local_val_edges_false = pickle.load(f)
        print("val_edges_false len: {}\n".format(len(local_val_edges_false)))

    return adj, num_nodes, num_edges, features, num_features, features_nonzero, local_adj_train, local_train_edges, \
        local_val_edges, local_val_edges_false, local_test_edges, local_val_edges_false


def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape
