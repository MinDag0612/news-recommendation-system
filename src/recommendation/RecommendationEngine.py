from sklearn.metrics.pairwise import cosine_similarity
from src.core.Context import context
import pickle

class RecommendationEngine:
    @staticmethod
    def cosine(v1, v2):
        return cosine_similarity(
            v1.reshape(1, -1),
            v2.reshape(1, -1)
        )[0][0]
        
    def __init__(self, represented_vector, alpha=0.5):
        self.represented_vector = represented_vector
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

    def score_candidates(self, user_vector, candidate_news):
        scores = []
        
        for news_id in candidate_news:
            score, semantic_score, topic_score = self.calculate_similarity(
                user_vector,
                self.represented_vector[news_id]
            )

            # scores.append((news_id, score, semantic_score, topic_score))
            scores.append({
                "news_id": news_id,
                "score": score,
                "semantic_score": semantic_score,
                "topic_score": topic_score
            })

        return scores

    @staticmethod
    def rank(scores):
        return sorted(
            scores,
            key=lambda x: x["score"],
            reverse=True,
        )

    @staticmethod
    def select_top_k(ranked_scores, k):
        if k <= 0:
            raise ValueError("k must be greater than zero")
        return ranked_scores[:k]

    def recommend(self, user_vector, candidate_news, k=10):
        scores = self.score_candidates(user_vector, candidate_news)
        self.score_list = self.rank(scores)
        return self.select_top_k(self.score_list, k)
    
    def calculate_by_impress(self, user_vector, impressed_list):
        impress_score = []
        for impress_set in impressed_list:
            impress_score_set = []
            for news in impress_set:
                news_id = news["news_id"]
                label = news["label"]
            
                score, _, _ = self.calculate_similarity(
                    user_vector,
                    self.represented_vector[news_id]
                )
            
                news_item = {
                    "news_id": news_id,
                    "label": label,
                    "score": score
                }
                
                impress_score_set.append(news_item)
            impress_score.append(impress_score_set)
            
        return impress_score
