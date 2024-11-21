from recbole.trainer import Trainer
from recbole.data import create_dataset, data_preparation
from model.mclightgcn import MCLightGCN


def initialize_trainer(config, co_matrix, user_item_criterion_ratings):
      """
    Initialize the trainer for the MCLightGCN model.

    Args:
        config (Config): Configuration parameters.
        co_matrix (array): Co-occurrence matrix for criteria.
        user_item_criterion_ratings (dict): User-item-criterion ratings.

    Returns:
        tuple: Trainer object, training data, and test data.
    """

    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)

    model = MCLightGCN(
        config,
        train_data.dataset,
        n_cri=config.n_criteria,
        co_matrix=co_matrix,
        user_item_criterion_ratings=user_item_criterion_ratings
    ).to(config.device)

    trainer = Trainer(config, model)

    return trainer, train_data, test_data


def train_and_evaluate(trainer, train_data, test_data):
     """
    Train the model and evaluate it on test data.

    Args:
        trainer (Trainer): RecBole trainer object.
        train_data (Dataset): Training dataset.
        test_data (Dataset): Testing dataset.

    Returns:
        dict: Evaluation results.
    """

    best_valid_score, best_valid_result = trainer.fit(train_data)
    test_result = trainer.evaluate(test_data)

    return test_result
