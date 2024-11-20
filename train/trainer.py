from recbole.trainer import Trainer
from recbole.data import create_dataset, data_preparation
from model.mclightgcn import MCLightGCN


def initialize_trainer(config, co_matrix, user_item_criterion_ratings):

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

    best_valid_score, best_valid_result = trainer.fit(train_data)
    test_result = trainer.evaluate(test_data)

    return test_result
