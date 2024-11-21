from recbole.model.abstract_recommender import GeneralRecommender
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import pickle
from recbole.model.loss import BPRLoss
from recbole.utils import InputType
from utils.graph_utils import normalize_adjacency_matrix
from attention import AttentionAlign


class MCLightGCN(GeneralRecommender):
     """
    MCLightGCN: A light graph convolutional network for multi-criteria recommendations.
    """
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset, n_cri, co_matrix, user_item_criterion_ratings):
        """
        Initialize the MCLightGCN model.

        Args:
            config (Config): Configuration parameters.
            dataset (Dataset): Dataset object.
            n_cri (int): Number of criteria.
            co_matrix (array): Co-occurrence matrix for criteria.
            user_item_criterion_ratings (dict): User-item-criterion ratings.
        """
        super(MCLightGCN, self).__init__(config, dataset)

        self._init_model_params(config, dataset, n_cri, co_matrix, user_item_criterion_ratings)

        self._build_model()

        self.norm_adj_matrix = self._construct_graph().to(self.device)

        self._init_weights()

    def _init_model_params(self, config, dataset, n_cri, co_matrix, user_item_criterion_ratings):
         """
        Initialize model parameters and load pre-trained embeddings.

        Args:
            config (Config): Configuration parameters.
            dataset (Dataset): Dataset object.
            n_cri (int): Number of criteria.
            co_matrix (array): Co-occurrence matrix for criteria.
            user_item_criterion_ratings (dict): User-item-criterion ratings.

        Returns:
            None
        """

        self.n_cri = n_cri
        self.co_matrix = co_matrix
        self.latent_dim = config['embedding_size']
        self.n_layers = config['n_layers']
        self.reg_weight = config['reg_weight']
        self.kd_weight = config['kd_weight']
        self.temperature = config['temperature']
        self.user_item_criterion_ratings = user_item_criterion_ratings
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)

        self.pretrain_user_embeddings, self.pretrain_item_embeddings = self._load_pretrained_embeddings()

    def _load_pretrained_embeddings(self):
          """
        Load pre-trained embeddings for users and items.

        Returns:
            tuple: Pre-trained user and item embeddings as tensors.
        """

        user_path = 'user_profiles_np_TA.pkl'
        item_path = 'item_profiles_np_TA.pkl'

        with open(user_path, 'rb') as f:
            pretrain_user_embeddings = torch.FloatTensor(np.array(pickle.load(f))).to(self.device)

        with open(item_path, 'rb') as f:
            pretrain_item_embeddings = torch.FloatTensor(np.array(pickle.load(f))).to(self.device)

        return pretrain_user_embeddings, pretrain_item_embeddings

    def _build_model(self):
          """
        Build the MCLightGCN model with embedding layers and attention mechanisms.

        Returns:
            None
        """

        self.user_embedding = nn.Embedding(self.n_users, self.latent_dim)
        self.item_embedding = nn.Embedding(self.n_items, self.latent_dim)
        self.criterion_embedding = nn.Embedding(self.n_cri, self.latent_dim)

        pretrain_dim_user = self.pretrain_user_embeddings.shape[1]
        pretrain_dim_item = self.pretrain_item_embeddings.shape[1]

        self.user_attention = AttentionAlign(pretrain_dim_user, self.latent_dim)
        self.item_attention = AttentionAlign(pretrain_dim_item, self.latent_dim)

        self.mf_loss = BPRLoss()

    def _construct_graph(self):
           """
        Construct and normalize the adjacency matrix for the graph.

        Returns:
            SparseTensor: Normalized adjacency matrix.
        """

        return normalize_adjacency_matrix(self.n_users, self.n_items, self.n_cri, self.interaction_matrix, self.co_matrix)

    def _init_weights(self):

        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        nn.init.xavier_uniform_(self.criterion_embedding.weight)

        self.user_attention.to(self.device)
        self.item_attention.to(self.device)

    def forward(self):

        user_embeddings = self.user_embedding.weight
        item_embeddings = self.item_embedding.weight
        criterion_embeddings = self.criterion_embedding.weight

        all_embeddings = torch.cat([user_embeddings, item_embeddings, criterion_embeddings], dim=0)
        embeddings_list = [all_embeddings]

        for _ in range(self.n_layers):
            all_embeddings = torch.sparse.mm(self.norm_adj_matrix, all_embeddings)
            embeddings_list.append(all_embeddings)

        final_embeddings = torch.stack(embeddings_list, dim=1).mean(dim=1)
        user_all_embeddings, item_all_embeddings, criterion_all_embeddings = torch.split(
            final_embeddings, [self.n_users, self.n_items, self.n_cri]
        )
        return user_all_embeddings, item_all_embeddings, criterion_all_embeddings

    def calculate_loss(self, interaction):

        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]

        user_all_embeddings, item_all_embeddings, _ = self.forward()
        u_embeddings = user_all_embeddings[user]
        pos_embeddings = item_all_embeddings[pos_item]
        neg_embeddings = item_all_embeddings[neg_item]

        pos_scores = torch.sum(u_embeddings * pos_embeddings, dim=-1)
        neg_scores = torch.sum(u_embeddings * neg_embeddings, dim=-1)
        mf_loss = self.mf_loss(pos_scores, neg_scores)

        reg_loss = (torch.sum(u_embeddings**2) + torch.sum(pos_embeddings**2) + torch.sum(neg_embeddings**2)) * self.reg_weight

        contrastive_loss = self._calculate_contrastive_loss(user, pos_item, neg_item, u_embeddings, pos_embeddings, neg_embeddings)

        return mf_loss + reg_loss + self.kd_weight * contrastive_loss

    def _calculate_contrastive_loss(self, user, pos_item, neg_item, u_embeddings, pos_embeddings, neg_embeddings):

        mapped_user_embeddings = self.user_attention(self.pretrain_user_embeddings)
        mapped_item_embeddings = self.item_attention(self.pretrain_item_embeddings)

        user_contrastive_loss = self._info_nce_loss(u_embeddings, mapped_user_embeddings[user])
        pos_contrastive_loss = self._info_nce_loss(pos_embeddings, mapped_item_embeddings[pos_item])
        neg_contrastive_loss = self._info_nce_loss(neg_embeddings, mapped_item_embeddings[neg_item])

        return user_contrastive_loss + pos_contrastive_loss + neg_contrastive_loss

    def _info_nce_loss(self, embeddings1, embeddings2):

        embeddings1 = F.normalize(embeddings1, dim=-1)
        embeddings2 = F.normalize(embeddings2, dim=-1)
        logits = torch.mm(embeddings1, embeddings2.t()) / self.temperature
        labels = torch.arange(embeddings1.size(0)).long().to(self.device)
        return F.cross_entropy(logits, labels)

    def predict(self, interaction):

        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]

        user_all_embeddings, item_all_embeddings, _ = self.forward()
        u_embeddings = user_all_embeddings[user]
        i_embeddings = item_all_embeddings[item]

        return torch.sum(u_embeddings * i_embeddings, dim=-1)

    def full_sort_predict(self, interaction):

        user = interaction[self.USER_ID]
        user_all_embeddings, item_all_embeddings, _ = self.forward()
        u_embeddings = user_all_embeddings[user]

        return torch.matmul(u_embeddings, item_all_embeddings.t())
