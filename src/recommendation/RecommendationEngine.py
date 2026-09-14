from sklearn.metrics.pairwise import cosine_similarity
import pickle
import numpy as np

class RecommendationEngine:
    @staticmethod
    def cosine(v1, v2):
        return np.dot(v1, v2) / (
            np.linalg.norm(v1) * np.linalg.norm(v2)
        )

        
    def __init__(self, alpha=0.5):
        self.alpha = alpha
        self.score = lambda semantic, topic: self.alpha * semantic + (1 - self.alpha) * topic

    def calculate_similarity(self, user_vector, news_vector):
        semantic = RecommendationEngine.cosine(
            user_vector["semantic"],
            news_vector["semantic"]
        )

        topic = RecommendationEngine.cosine(
            user_vector["topic_distribution"],
            news_vector["topic_distribution"]
        )

        return self.score(semantic, topic), semantic, topic

    def recommended_with_label(self, user_vector, candidate_news):
        # news_vector = [self.represented_vector[i] for i in candidate_news]
        scores = []
        
        for news in candidate_news:
            score, semantic_score, topic_score = self.calculate_similarity(
                user_vector,
                news["vector"]
            )

            # scores.append((news_id, score, semantic_score, topic_score))
            scores.append({
                "news_id": news["news_id"],
                "label": news["label"],
                "score": score,
                "semantic_score": semantic_score,
                "topic_score": topic_score
            })
            
        scores.sort(key=lambda x: x["score"], reverse=True)
        self.score_list = scores
        
        return self.score_list

    def recommended_from_vectors(self, urv, news_vectors):
        """Rank already calculated news vectors against an already calculated URV."""
        scores = []

        for news_id, news_vector in enumerate(news_vectors, start=1):
            score, semantic_score, topic_score = self.calculate_similarity(
                urv,
                news_vector,
            )
            scores.append({
                "news_id": news_vector.get("title", news_id),
                "score": float(score),
                "semantic_score": float(semantic_score),
                "topic_score": float(topic_score),
            })

        scores.sort(key=lambda item: item["score"], reverse=True)
        self.score_list = scores

        return self.score_list
    
    def recommended_no_label(self, user_vector, candidate_news):
            # news_vector = [self.represented_vector[i] for i in candidate_news]
            scores = []
            
            for news in candidate_news:
                score, _, _ = self.calculate_similarity(
                    user_vector,
                    news["vector"]
                )
    
                # scores.append((news_id, score, semantic_score, topic_score))
                scores.append({
                    "news_id": news["news_id"],
                    # "label": news["label"],
                    "score": float(score),
                    # "semantic_score": semantic_score,
                    # "topic_score": topic_score
                })
                
            scores.sort(key=lambda x: x["score"], reverse=True)
            self.score_list = scores
            
            return self.score_list
            