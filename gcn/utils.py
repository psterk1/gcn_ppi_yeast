import networkx as nx
import numpy as np
import scipy.sparse as sp
import tensorflow as tf
import _pickle as pickle

#
# As an optimization, load precomputed masked edges
#
def load_data():
    g = nx.read_edgelist('data/yeast.edgelist')
    adj = nx.adjacency_matrix(g)

    with open('data/adj_train.npz', 'rb') as f:
        adj_train = sp.load_npz(f)

    with open('data/adj_val.npz', 'rb') as f:
        adj_val = sp.load_npz(f)

    with open('data/adj_test.npz', 'rb') as f:
        adj_test = sp.load_npz(f)

    with open('data/val_edges.npy', 'rb') as f:
        val_edges = np.load(f)
        print("val_edges.shape: {}\n".format(val_edges.shape))

    with open('data/val_edges_false.p', 'rb') as f:
        val_edges_false = pickle.load(f)
        print("val_edges_false len: {}\n".format(len(val_edges_false)))

    with open('data/test_edges.npy', 'rb') as f:
        test_edges = np.load(f)
        print("test_edges.shape: {}\n".format(test_edges.shape))

    with open('data/test_edges_false.p', 'rb') as f:
        test_edges_false = pickle.load(f)
        print("test_edges_false len: {}\n".format(len(test_edges_false)))

    return adj, adj_train, val_edges, val_edges_false, test_edges, test_edges_false


def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape


def mask_test_edges(adj):
    # Function to build test set with 2% positive links
    # Remove diagonal elements
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()

    adj_triu = sp.triu(adj)
    adj_tuple = sparse_to_tuple(adj_triu)
    edges = adj_tuple[0]
    edges_all = sparse_to_tuple(adj)[0]
    # 2% -> 2/100 -> 1/50
    num_test = int(np.floor(edges.shape[0] / 50.))
    num_val = int(np.floor(edges.shape[0] / 50.))

    all_edge_idx = list(range(edges.shape[0]))
    np.random.shuffle(all_edge_idx)
    val_edge_idx = all_edge_idx[:num_val]
    test_edge_idx = all_edge_idx[num_val:(num_val + num_test)]
    test_edges = edges[test_edge_idx]
    val_edges = edges[val_edge_idx]
    train_edges = np.delete(edges, np.hstack([test_edge_idx, val_edge_idx]), axis=0)

    def ismember(a, b):
        rows_close = np.all((a - b[:, None]) == 0, axis=-1)
        return np.any(rows_close)

    test_edges_false = []
    while len(test_edges_false) < len(test_edges):
        n_rnd = len(test_edges) - len(test_edges_false)
        rnd = np.random.randint(0, adj.shape[0], size=2 * n_rnd)
        idxs_i = rnd[:n_rnd]
        idxs_j = rnd[n_rnd:]
        for i in range(n_rnd):
            idx_i = idxs_i[i]
            idx_j = idxs_j[i]
            if idx_i == idx_j:
                continue
            if ismember([idx_i, idx_j], edges_all):
                continue
            if test_edges_false:
                if ismember([idx_j, idx_i], np.array(test_edges_false)):
                    continue
                if ismember([idx_i, idx_j], np.array(test_edges_false)):
                    continue
            test_edges_false.append([idx_i, idx_j])

    val_edges_false = []
    while len(val_edges_false) < len(val_edges):
        n_rnd = len(val_edges) - len(val_edges_false)
        rnd = np.random.randint(0, adj.shape[0], size=2 * n_rnd)
        idxs_i = rnd[:n_rnd]
        idxs_j = rnd[n_rnd:]
        for i in range(n_rnd):
            idx_i = idxs_i[i]
            idx_j = idxs_j[i]
            if idx_i == idx_j:
                continue
            if ismember([idx_i, idx_j], train_edges):
                continue
            if ismember([idx_j, idx_i], train_edges):
                continue
            if ismember([idx_i, idx_j], val_edges):
                continue
            if ismember([idx_j, idx_i], val_edges):
                continue
            if val_edges_false:
                if ismember([idx_j, idx_i], np.array(val_edges_false)):
                    continue
                if ismember([idx_i, idx_j], np.array(val_edges_false)):
                    continue
            val_edges_false.append([idx_i, idx_j])

    # Re-build adj matrix
    data = np.ones(train_edges.shape[0])
    adj_train = sp.csr_matrix((data, (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
    adj_train = adj_train + adj_train.T
    data = np.ones(val_edges.shape[0])
    adj_val = sp.csr_matrix((data, (val_edges[:, 0], val_edges[:, 1])), shape=adj.shape)
    adj_val = adj_val + adj_val.T
    data = np.ones(test_edges.shape[0])
    adj_test = sp.csr_matrix((data, (test_edges[:, 0], test_edges[:, 1])), shape=adj.shape)
    adj_test = adj_test + adj_test.T

    with open('data/adj_train.npz', 'wb') as f:
        sp.save_npz(f, adj_train)

    with open('data/adj_val.npz', 'wb') as f:
        sp.save_npz(f, adj_val)

    with open('data/adj_test.npz', 'wb') as f:
        sp.save_npz(f, adj_test)

    with open('data/val_edges.npy', 'wb') as f:
        np.save(f, val_edges)
        print("val_edges.shape: {}\n".format(val_edges.shape))

    with open('data/val_edges_false.p', 'wb') as f:
        pickle.dump(val_edges_false, f)
        print("val_edges_false len: {}\n".format(len(val_edges_false)))

    with open('data/test_edges.npy', 'wb') as f:
        np.save(f, test_edges)
        print("test_edges.shape: {}\n".format(test_edges.shape))

    with open('data/test_edges_false.p', 'wb') as f:
        pickle.dump(test_edges_false, f)
        print("test_edges_false len: {}\n".format(len(val_edges_false)))

    return adj_train, adj_val, adj_test, val_edges, val_edges_false, test_edges, test_edges_false
