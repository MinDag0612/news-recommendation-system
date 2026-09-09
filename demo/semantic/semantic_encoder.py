from demo.base_encoder import BaseEncoder
from src.core.Models import Models

class SemanticEncoder(BaseEncoder):
    def __init__(self):
        self.model = Models().sentence_model
        
    def encode(self, texts: list[str]) -> dict[int, list[float]]:
        semantic_vector = self.model.encode(
            texts, show_progress_bar=True
        )
        return {
            texts[index - 1]: vector.tolist()
            for index, vector in enumerate(semantic_vector, start=1)
        }