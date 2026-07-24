from sklearn.metrics.pairwise import cosine_similarity

class RecommendationEngine:
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
        semantic = self.cosine(
            user_vector["semantic"],
            news_vector["semantic"]
        )

        topic = self.cosine(
            user_vector["topic_distribution"],
            news_vector["topic_distribution"]
        )

        return self.score(semantic, topic), semantic, topic

    def recommend(self, user_vector, candidate_news):
        # news_vector = [self.represented_vector[i] for i in candidate_news]
        scores = []
        
        for news_id in candidate_news:
            score, semantic_score, topic_score = self.calculate_similarity(
                user_vector,
                self.represented_vector[news_id]
            )

            scores.append((news_id, score, semantic_score, topic_score))
                
        scores.sort(key=lambda x: x[1], reverse=True)