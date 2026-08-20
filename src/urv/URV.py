import numpy as np

class URV:
    def __init__(self):
        pass

        
    def getURV(self, user_history: dict, represented_vector):
        user_history_id = user_history.split()
        # behavior_id = user_behaviours["behavior_id"]
        
        user_history_vector = [
            represented_vector[news_id]
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
        
        
        return user_representation_vector
    
    def getURVFromVector(self, user_history_vector):
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
        
        
        return user_representation_vector