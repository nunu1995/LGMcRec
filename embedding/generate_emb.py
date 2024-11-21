import openai
import json
import pickle
import numpy as np


def generate_embeddings(input_json, output_pkl):
    """
       Generate embeddings for user/item profiles using OpenAI API and save them as a pickle file.

       Args:
           input_json (str): Path to the input JSON file containing user/item profiles.
           output_pkl (str): Path to the output pickle file to save the embeddings.

       Returns:
           None
       """

    with open(input_json, 'r') as f:
        responses = json.load(f)

    embeddings = []
    for user in responses:
        embedding = openai.Embedding.create(
            input=user['summarization'],
            model="text-embedding-ada-002"
        )["data"][0]["embedding"]
        embeddings.append(np.array(embedding))

    with open(output_pkl, 'wb') as f:
        pickle.dump(embeddings, f)
