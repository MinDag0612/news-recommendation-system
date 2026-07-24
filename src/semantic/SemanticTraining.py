import pandas as pd
from src.core.Vector import Vector

class SentenceVector(Vector):
    def __init__(self, model):
        self.model = model
        self.semantic_vector = None

    def get_vector(self, title_list):
        self.title_list = title_list

        titles = [news["title"] for news in title_list]

        vectors = self.model.encode(
            titles, show_progress_bar=True, convert_to_numpy=True
        )

        self.semantic_vector = {
            news["news_id"]: vector for news, vector in zip(title_list, vectors)
        }
        
        return self.semantic_vector

    def overview(self):
        print("=" * 60)
        print("Sentence Embedding Overview")
        print(f"[Sentence is: {list(self.semantic_vector.keys())[0]}]")
        print("=" * 60)

        print(f"Documents : {len(self.semantic_vector)}")
        print(
            f"Dimension : {self.semantic_vector[list(self.semantic_vector.keys())[0]].shape[0]}"
        )
        print(
            f"Shape     : {self.semantic_vector[list(self.semantic_vector.keys())[0]].shape}"
        )

        print("\nFirst vector (first 10 values):")
        print(self.semantic_vector[list(self.semantic_vector.keys())[0]][:10], "...")

    def summary(self, sample_index=0):

        sample = self.title_list[sample_index]

        news_id = sample["news_id"]
        title = sample["title"]

        vector = self.semantic_vector[news_id]

        metrics = pd.DataFrame(
            {
                "Metric": [
                    "Embedding Model",
                    "Number of Documents",
                    "Embedding Dimension",
                    "Output Shape",
                ],
                "Value": [
                    self.model.__class__.__name__,
                    len(self.semantic_vector),
                    vector.shape[0],
                    str(vector.shape),
                ],
            }
        )

        vector_preview = ", ".join(f"{x:.4f}" for x in vector[:10]) + ", ..."

        example = pd.DataFrame(
            {
                "News ID": [news_id],
                "Sample Title": [title],
                "Embedding (first 10 dims)": [f"[{vector_preview}]"],
            }
        )

        return metrics, example