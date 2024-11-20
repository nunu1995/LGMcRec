import pandas as pd
import numpy as np
from utils.graph_utils import construct_co_occurrence_matrix, save_inter_file


class DataPreprocessor:

    def __init__(self, input_file, output_dir, criteria):
        self.input_file = input_file
        self.output_dir = output_dir
        self.criteria = criteria
        self.df_raw = None
        self.df_processed = None
        self.user_map = None
        self.item_map = None
        self.n_users = 0
        self.n_items = 0

    def load_data(self):

        self.df_raw = pd.read_csv(self.input_file)
        self.df_raw.rename(columns={'UserID': 'user_id', 'ItemID': 'item_id'}, inplace=True)

    def normalize_criteria(self):

        self.df_processed = self.df_raw.copy()
        for criterion in self.criteria:
            self.df_processed[criterion] = self.df_processed[criterion] / 5.0
        self.df_processed['rating'] = self.df_processed[self.criteria].sum(axis=1)

    def filter_and_remap_ids(self, min_interactions=5):

        user_counts = self.df_processed['user_id'].value_counts()
        valid_users = user_counts[user_counts >= min_interactions].index
        self.df_processed = self.df_processed[self.df_processed['user_id'].isin(valid_users)].reset_index(drop=True)

        self.user_map = {old_id: new_id for new_id, old_id in enumerate(sorted(self.df_processed['user_id'].unique()))}
        self.item_map = {old_id: new_id for new_id, old_id in enumerate(sorted(self.df_processed['item_id'].unique()))}

        self.df_processed['user_id'] = self.df_processed['user_id'].map(self.user_map)
        self.df_processed['item_id'] = self.df_processed['item_id'].map(self.item_map)

        self.n_users = len(self.user_map)
        self.n_items = len(self.item_map)

    def split_data(self, train_ratio=0.7, val_ratio=0.1):

        user_groups = self.df_processed.groupby('user_id')
        train_list, val_list, test_list = [], [], []

        for _, group in user_groups:
            num_ratings = len(group)
            train_end = int(num_ratings * train_ratio)
            val_end = train_end + int(num_ratings * val_ratio)

            train_list.append(group.iloc[:train_end])
            val_list.append(group.iloc[train_end:val_end])
            test_list.append(group.iloc[val_end:])

        return pd.concat(train_list), pd.concat(val_list), pd.concat(test_list)

    def preprocess(self):

        self.load_data()
        self.normalize_criteria()
        self.filter_and_remap_ids()
        train_data, val_data, test_data = self.split_data()

        co_matrix = construct_co_occurrence_matrix(train_data, self.criteria)
        save_inter_file(train_data, f"{self.output_dir}/train.inter")
        save_inter_file(val_data, f"{self.output_dir}/val.inter")
        save_inter_file(test_data, f"{self.output_dir}/test.inter")

        return co_matrix


def preprocess_data(input_file, output_dir, criteria):

    preprocessor = DataPreprocessor(input_file, output_dir, criteria)
    return preprocessor.preprocess()
