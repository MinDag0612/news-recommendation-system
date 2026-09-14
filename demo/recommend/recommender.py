from src.urv.URV import URV

class Recommender:
    def __init__(self, semantic_client, topic_client, recommender_engine):
        self.semantic_client = semantic_client
        self.topic_client = topic_client
        self.recommender_engine = recommender_engine
        self.urv = URV()
        

    def recommend(self, news: list[str], history: list[str]):
        news_semantic_vectors = self.semantic_client.encode(news)

        news_topic_vectors = self.topic_client.encode(
            news,
            news_semantic_vectors
        )
        
        news_vectors = [
            {
                "title": key,
                "semantic": news_semantic_vectors[key],
                "topic_distribution": news_topic_vectors[key]
            }
            for key in news_semantic_vectors
        ]
    
        urv = self.handleHistory(history)
        
        return self.recommender_engine.recommended_from_vectors(
            urv,
            news_vectors
        )
        
    def handleHistory(self, history: list[str]):
        history_semantic_vectors = self.semantic_client.encode(history)

        history_topic_vectors = self.topic_client.encode(
            history,
            history_semantic_vectors
        )

        user_history_vector = [
            {
                "semantic": history_semantic_vectors[key],
                "topic_distribution": history_topic_vectors[key]
            }
            for key in history_semantic_vectors
        ]

        return self.urv.getURVFromVector(
            user_history_vector = user_history_vector
        )
    
