import openai
import json
import pickle
import numpy as np


def generate_embeddings(input_json, output_pkl):

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
