import random
import numpy as np
import torch


def init_seed(seed, reproducibility):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if reproducibility:
        torch.cuda.manual_seed_all(seed)
