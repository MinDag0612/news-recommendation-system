from sklearn.feature_extraction.text import CountVectorizer

from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
import os
from hdbscan import HDBSCAN
from umap import UMAP


class Models:
    def __init__(self):
        cpu_workers = max(1, (os.cpu_count() or 2) - 1)
        
        self.sentence_model = SentenceTransformer(
                    "all-MiniLM-L6-v2",
                    device="cpu",
                )
        self.svectorizer = CountVectorizer(
            stop_words="english",
        )

        self.topic_model = BERTopic(
            calculate_probabilities=True,
            verbose=True,
            vectorizer_model=self.svectorizer,
            umap_model=UMAP(
                n_neighbors=15,
                n_components=5,
                min_dist=0.0,
                metric="cosine",
                n_jobs=cpu_workers,
                low_memory=True,
            ),
            hdbscan_model=HDBSCAN(
                min_cluster_size=20,
                metric="euclidean",
                cluster_selection_method="eom",
                prediction_data=True,
                core_dist_n_jobs=cpu_workers,
            ),
        )
        
        
