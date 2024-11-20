import torch


class Config:
    def __init__(self):
        self.seed = 42
        self.reproducibility = True
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.embedding_size = 64
        self.n_layers = 2
        self.reg_weight = 1e-1
        self.kd_weight = 1e-1
        self.temperature = 0.2
        self.epochs = 100
        self.metrics = ['Precision', 'Recall', 'NDCG']
        self.topk = [5, 10]
