from fastapi import FastAPI
from demo.recommend.recommender import Recommender
from demo.semantic.semantic_client import SemanticClient
from demo.topic.topic_client import TopicClient
from src.recommendation.RecommendationEngine import RecommendationEngine

app = FastAPI()

recommender = Recommender(
    semantic_client=SemanticClient("http://semantic:8001"),
    topic_client=TopicClient("http://topic:8002"),
    recommender_engine=RecommendationEngine(alpha=0.75)
)
    

@app.get("/recommender-health")
def health():
    return {"status": "recommender healthy"}

@app.post("/recommender-recommend")
def recommend(news: list[str], history: list[str]):
    results = recommender.recommend(news=news, history=history)
    return {"results": results}