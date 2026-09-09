from demo.base_encoder import BaseEncoder
from bertopic import BERTopic
import os
import numpy as np

class TopicEncoder(BaseEncoder):
    def __init__(self):
        model_path = os.getenv(
            "TOPIC_MODEL_PATH",
            "demo/model/bertopic"
        )

        self.model = BERTopic.load(model_path)
        
    def encode(
        self,
        texts: list[str],
        semantic_vectors: dict[str, list[float]],
    ) -> dict[str, list[float]]:
        semantic_matrix = np.asarray(
            [semantic_vectors[title] for title in texts]
        )

        _, topic_distributions = self.model.transform(
            texts,
            embeddings=semantic_matrix
        )

        return {
            texts[index - 1]: vector.tolist()
            for index, vector in enumerate(topic_distributions, start=1)
        }