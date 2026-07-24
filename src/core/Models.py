from sklearn.feature_extraction.text import CountVectorizer

from sentence_transformers import SentenceTransformer
from bertopic import BERTopic

class Models:
    def __init__(self):
        self.sentence_model = SentenceTransformer("all-MiniLM-L6-v2")

        self.svectorizer = CountVectorizer(
            stop_words="english",
        )

        self.topic_model = BERTopic(
            calculate_probabilities=True,  # Important to set this to True for probability calculations
            verbose=True,
            vectorizer_model=self.svectorizer,
        )