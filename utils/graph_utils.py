import numpy as np
import scipy.sparse as sp
import torch


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
