import pandas as pd
import os
import json


def process_data(input_file, output_folder, output_json):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    data = pd.read_csv(input_file)
    user_groups = data.groupby('UserID')

    for user_id, group in user_groups:
        group.to_csv(f'{output_folder}/user_{user_id}_data.txt', index=False, sep=',')

    user_files = sorted(
        [f for f in os.listdir(output_folder) if f.startswith('user_') and f.endswith('_data.txt')],
        key=lambda x: int(x.split('_')[1])
    )

    json_list = []
    for user_file in user_files:
        file_path = os.path.join(output_folder, user_file)
        user_data = pd.read_csv(file_path)
        prompt = {"prompt": [row.to_dict() for _, row in user_data.iterrows()]}
        json_list.append(prompt)

    with open(output_json, 'w') as f:
        json.dump(json_list, f, separators=(',', ':'))
