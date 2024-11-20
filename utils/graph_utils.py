import numpy as np
import scipy.sparse as sp
import torch


def normalize_adjacency_matrix(n_users, n_items, n_cri, interaction_matrix, co_matrix):

    num_nodes = n_users + n_items + n_cri
    A = sp.dok_matrix((num_nodes, num_nodes), dtype=np.float32)

    for u_idx, i_idx, rating in zip(interaction_matrix.row, interaction_matrix.col, interaction_matrix.data):
        w_ui = rating
        A[u_idx, n_users + i_idx] = w_ui
        A[n_users + i_idx, u_idx] = w_ui

    for i in range(n_items):
        for c in range(n_cri):
            A[n_users + i, n_users + n_items + c] = 1.0
            A[n_users + n_items + c, n_users + i] = 1.0

    for c_i in range(n_cri):
        for c_j in range(n_cri):
            if c_i != c_j:
                M_cicj = co_matrix[c_i][c_j]
                A[n_users + n_items + c_i, n_users + n_items + c_j] = M_cicj

    sum_arr = A.sum(axis=1).A1
    diag = np.power(sum_arr + 1e-7, -0.5)
    D = sp.diags(diag)
    L = D @ A @ D

    L = sp.coo_matrix(L)
    row = L.row
    col = L.col
    i = torch.LongTensor([row, col])
    data = torch.FloatTensor(L.data)
    SparseL = torch.sparse.FloatTensor(i, data, torch.Size(L.shape))

    return SparseL


def construct_co_occurrence_matrix(data, criteria, threshold=0.6):

    n_criteria = len(criteria)
    co_matrix = np.zeros((n_criteria, n_criteria))

    for i, cri_i in enumerate(criteria):
        for j, cri_j in enumerate(criteria):
            if i != j:
                co_occurrences = len(data[(data[cri_i] >= threshold) & (data[cri_j] >= threshold)])
                total_occurrences = len(data[data[cri_i] >= threshold])
                co_matrix[i][j] = co_occurrences / total_occurrences if total_occurrences > 0 else 0
    return co_matrix


def save_inter_file(data, output_file):

    inter_data = data.copy()
    inter_data.rename(columns={
        'user_id': 'user_id:token',
        'item_id': 'item_id:token',
        'rating': 'rating:float',
    }, inplace=True)
    inter_data.to_csv(output_file, index=False, sep='\t')
