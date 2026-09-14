from fastapi import FastAPI
from pydantic import BaseModel
from demo.topic.topic_encoder import TopicEncoder

app = FastAPI()

encoder = TopicEncoder()

@app.get("/health")
def health():
    return {"status": "topic healthy"}

class TopicRequest(BaseModel):
    texts: list[str]
    semantic_vectors: dict[str, list[float]]

@app.post("/topic-encode")
def encode(request: TopicRequest):
    vectors = encoder.encode(
        request.texts,
        request.semantic_vectors
    )

    return {"vectors": vectors}