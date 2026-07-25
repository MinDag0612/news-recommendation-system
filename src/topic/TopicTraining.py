import numpy as np
import pandas as pd
from src.core.Vector import Vector

class BERTopicVector(Vector):
    def __init__(self, model):
        self.model = model

        self.notice = (
            "BERTopic depends on the sentence transformer model."
            " Please ensure that the sentence transformer model is trained before using BERTopic."
        )

        self.probabilities = None
        self.topics_vector = None

    def get_vector(self, title_list, semantic_vector=None):
        self.title_list = title_list

        if semantic_vector is None:
            print(self.notice)
            return None

        titles = [news["title"] for news in title_list]

        embeddings = np.array([semantic_vector[news["news_id"]] for news in title_list])

        topics, probabilities = self.model.fit_transform(
            titles,
            embeddings,
        )

        self.topic_vector = {
            news["news_id"]: {"topic": topic, "probability": probability}
            for news, topic, probability in zip(title_list, topics, probabilities)
        }

        return self.topic_vector

    def overview(self):

        print("=" * 60)
        print("BERTopic Overview")
        print("=" * 60)

        print(f"Number of documents : {len(self.title_list)}")
        print(
            f"Number of topics    : {len(set(v['topic'] for v in self.topic_vector.values()) - {-1})}"
        )

        print("\nTopic distribution:")
        print(self.model.get_topic_info()[["Topic", "Count"]])

        print("\nFirst 5 documents:")

        for sample in self.title_list[:5]:

            news_id = sample["news_id"]
            title = sample["title"]

            topic = self.topic_vector[news_id]["topic"]

            print(f"{news_id}")
            print(f"Title : {title}")
            print(f"Topic : {topic}")
            print("-" * 40)

        first_probability = next(iter(self.topic_vector.values()))["probability"]

        print("\nProbability shape:")
        print(first_probability.shape)

        print("\nFirst 5 probability vectors:")

        for sample in self.title_list[:5]:

            news_id = sample["news_id"]

            probability = self.topic_vector[news_id]["probability"]

            preview = ", ".join(f"{p:.4f}" for p in probability[:10])

            print(f"{news_id} -> [{preview}, ...]")

    def summary(self, sample_index=0):

        # =========================
        # Metrics
        # =========================

        topics = [value["topic"] for value in self.topic_vector.values()]

        first_probability = next(iter(self.topic_vector.values()))["probability"]

        metrics_df = pd.DataFrame(
            {
                "Metric": [
                    "Topic Model",
                    "Number of Documents",
                    "Number of Topics",
                    "Number of Outliers",
                    "Probability Shape",
                ],
                "Value": [
                    self.model.__class__.__name__,
                    len(self.title_list),
                    len(set(topics) - {-1}),
                    np.sum(np.array(topics) == -1),
                    str(first_probability.shape),
                ],
            }
        )

        # =========================
        # Topic Information
        # =========================

        topic_df = self.model.get_topic_info()[["Topic", "Count"]].copy()

        keywords = []

        for topic in topic_df["Topic"]:

            if topic == -1:
                keywords.append("Outlier")
            else:
                words = [word for word, _ in self.model.get_topic(topic)[:5]]
                keywords.append(", ".join(words))

        topic_df["Top Keywords"] = keywords

        # =========================
        # Sample
        # =========================

        sample = self.title_list[sample_index]

        news_id = sample["news_id"]
        title = sample["title"]

        topic_info = self.topic_vector[news_id]

        probs = ", ".join(f"{p:.4f}" for p in topic_info["probability"])

        sample_df = pd.DataFrame(
            {
                "News ID": [news_id],
                "Sample Title": [title],
                "Assigned Topic": [topic_info["topic"]],
                "Probability Distribution": [f"[{probs}]"],
            }
        )

        return metrics_df, topic_df, sample_df
