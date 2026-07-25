from sklearn.feature_extraction.text import CountVectorizer

from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from umap import UMAP

class Models:
    def __init__(self, load_sentence_model=True):
        self.sentence_model = (
            SentenceTransformer("all-MiniLM-L6-v2")
            if load_sentence_model
            else None
        )

        self.svectorizer = CountVectorizer(
            stop_words="english",
        )

        self.umap_model = UMAP(
            n_neighbors=15,
            n_components=5,
            min_dist=0.0,
            metric="cosine",
            low_memory=True,
            n_jobs=1,
            random_state=42,
        )

        self.topic_model = BERTopic(
            calculate_probabilities=True,  # Important to set this to True for probability calculations
            verbose=True,
            vectorizer_model=self.svectorizer,
            umap_model=self.umap_model,
        )
