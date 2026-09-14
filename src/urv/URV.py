import numpy as np

class URV:
    def __init__(self):
        pass

    @staticmethod
    def _mean_vectors(vectors):
        vectors = iter(vectors)
        first = np.asarray(next(vectors), dtype=np.float32)
        total = first.copy()
        count = 1

        for vector in vectors:
            total += np.asarray(vector, dtype=np.float32)
            count += 1

        return total / count

        
    def getURV(self, user_history: dict, represented_vector):
        user_history_id = user_history.split()
        # behavior_id = user_behaviours["behavior_id"]
        
        user_history_vector = [
            represented_vector[news_id]
            for news_id in user_history_id
        ]

        user_representation_vector = {
                # "behavior_id": behavior_id,
                "semantic": self._mean_vectors(
                    v["semantic"] for v in user_history_vector
                ),
                "topic_distribution": self._mean_vectors(
                    v["topic_distribution"] for v in user_history_vector
                )
            }
        
        
        return user_representation_vector
    
    def getURVFromVector(self, user_history_vector):
        user_representation_vector = {
                # "behavior_id": behavior_id,
                "semantic": self._mean_vectors(
                    v["semantic"] for v in user_history_vector
                ),
                "topic_distribution": self._mean_vectors(
                    v["topic_distribution"] for v in user_history_vector
                )
            }
        
        
        return user_representation_vector