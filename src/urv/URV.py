import numpy as np

class URV:
    def __init__(self, represented_vector):
        self.represented_vector = represented_vector
        self.user_representation_vector = None
        
    def getURV(self, history, user_id=None, impression_id=None):
        user_history_id = history.split() if isinstance(history, str) else list(history)
        if not user_history_id:
            raise ValueError("History must contain at least one news ID")
        
        user_history_vector = [
            self.represented_vector[news_id]
            for news_id in user_history_id
        ]

        user_representation_vector = {
                "user_id": user_id,
                "impression_id": impression_id,
                "semantic": np.mean(
                    [v["semantic"] for v in user_history_vector],
                    axis=0
                ),
                "topic_distribution": np.mean(
                    [v["topic_distribution"] for v in user_history_vector],
                    axis=0
                )
            }
        
        self.user_representation_vector = user_representation_vector
        
        assert np.allclose(
            np.mean([v["semantic"] for v in user_history_vector], axis=0),
            sum(v["semantic"] for v in user_history_vector) / len(user_history_vector)
        )
        
        return self.user_representation_vector
