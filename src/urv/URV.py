import numpy as np

class URV:
    def __init__(self, represented_vector):
        self.represented_vector = represented_vector
        # self.user_behaviours = user_behaviours
        # self.user_representation_vector = None
        
    def getURV(self, user_history: dict):
        user_history_id = user_history.split()
        # behavior_id = user_behaviours["behavior_id"]
        
        user_history_vector = [
            self.represented_vector[news_id]
            for news_id in user_history_id
        ]

        user_representation_vector = {
                # "behavior_id": behavior_id,
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