from config import Config
from data.preprocess import preprocess_data
from embedding.generate_profile import generate_profiles
from embedding.generate_emb import generate_embeddings
from train.trainer import initialize_trainer, train_and_evaluate
from utils.seed_utils import init_seed
from utils.logging_utils import init_logger


def main():

    input_file = 'unprocess_dataset/multi_TA.csv'
    output_folder = 'User_TA'

    generate_profiles(
        input_json=f'{output_folder}/user_prompts_TA.json',
        system_prompt_file='user_introduction_prompts_TA.txt',
        output_json=f'{output_folder}/user_profiles_TA.json',
    )

    generate_embeddings(
        input_json=f'{output_folder}/user_profiles_TA.json',
        output_pkl=f'{output_folder}/user_profiles_TA.pkl'
    )

    co_matrix, user_item_criterion_ratings = preprocess_data(
        input_file=input_file,
        output_dir=output_folder,
        criteria=['Value', 'Service', 'Rooms', 'Location', 'Cleanliness', 'Checkin', 'Business', 'overall']
    )

    config = Config()
    init_seed(config.seed, config.reproducibility)
    logger = init_logger(config)

    trainer, train_data, test_data = initialize_trainer(config, co_matrix, user_item_criterion_ratings)

    test_result = train_and_evaluate(trainer, train_data, test_data)

    logger.info(test_result)


if __name__ == "__main__":
    main()
