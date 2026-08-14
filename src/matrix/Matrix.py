from sklearn.metrics import roc_auc_score
import numpy as np


class Metrix:
    def __init__(self, impression):
        self.impression = impression

    def _prepare(self):
        ranked = sorted(
            self.impression,
            key=lambda x: x["score"],
            reverse=True
        )

        labels = [item["label"] for item in ranked]
        scores = [item["score"] for item in ranked]

        return labels, scores

    def auc(self):
        labels, scores = self._prepare()

        # roc_auc_score cần có cả positive và negative
        if len(set(labels)) < 2:
            return 0.0

        return roc_auc_score(labels, scores)

    def mrr(self):
        labels, _ = self._prepare()

        for rank, label in enumerate(labels, start=1):
            if label == 1:
                return 1 / rank

        return 0.0

    def ndcg(self, k):
        labels, _ = self._prepare()
        ranked_labels = labels[:k]

        dcg = 0
        for i, rel in enumerate(ranked_labels):
            dcg += (2**rel - 1) / np.log2(i + 2)

        ideal = sorted(labels, reverse=True)[:k]

        idcg = 0
        for i, rel in enumerate(ideal):
            idcg += (2**rel - 1) / np.log2(i + 2)

        if idcg == 0:
            return 0.0

        return dcg / idcg

    def hit_rate(self, k):
        if not isinstance(k, int) or isinstance(k, bool) or k <= 0:
            raise ValueError("k must be a positive integer")
        labels, _ = self._prepare()
        return float(any(label > 0 for label in labels[:k]))

    def evaluate(self):
        return {
            "AUC": self.auc(),
            "MRR": self.mrr(),
            "nDCG@5": self.ndcg(5),
            "nDCG@10": self.ndcg(10),
            "HitRate@5": self.hit_rate(5),
            "HitRate@10": self.hit_rate(10),
        }
