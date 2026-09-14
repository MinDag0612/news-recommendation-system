from fastapi import FastAPI
from demo.semantic.semantic_encoder import SemanticEncoder

app = FastAPI()

encoder = SemanticEncoder()

@app.get("/semantic-health")
def health():
    return {"status": "semantic healthy"}

@app.post("/semantic-encode")
def encode(texts: list[str]):
    vectors = encoder.encode(texts)
    return {"vectors": vectors}