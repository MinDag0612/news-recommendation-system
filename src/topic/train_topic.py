import pickle
from pathlib import Path
import pandas as pd

from src.core.Models import Models
from src.core.Context import context
from src.topic.TopicTraining import BERTopicVector


model = Models()

project_root = Path(__file__).resolve().parents[2]
vectors_dir = project_root / "vectors"
models_dir = project_root / "models"
vectors_dir.mkdir(parents=True, exist_ok=True)
models_dir.mkdir(parents=True, exist_ok=True)

vector_df = pd.read_pickle(vectors_dir / "semantic_vectors.pkl")

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

with open(vectors_dir / "topic_vectors.pkl", "wb") as f:
    pickle.dump(
        bertopic.topic_vector,
        f,
        protocol=pickle.HIGHEST_PROTOCOL,
    )

bertopic.model.save(models_dir / "bertopic")
