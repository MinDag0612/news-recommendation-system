import pickle
import pandas as pd

from src.core.Models import Models
from src.core.Context import context
from topic.BERTopicVector import BERTopicVector


model = Models()

vector_df = pd.read_pickle("vectors/semantic_vectors.pkl")

semantic_vector = (
    vector_df
    .set_index("News_ID")["semantic_vector"]
    .to_dict()
)

bertopic = BERTopicVector(model.topic_model)

bertopic.get_vector(
    context.title_list,
    semantic_vector
)

with open("vectors/topic_vectors.pkl", "wb") as f:
    pickle.dump(
        bertopic.topic_vector,
        f,
        protocol=pickle.HIGHEST_PROTOCOL,
    )

bertopic.model.save("models/bertopic")