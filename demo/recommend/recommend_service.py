import os
import hashlib
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator
import requests
from demo.recommend.recommender import Recommender
from demo.semantic.semantic_client import SemanticClient
from demo.topic.topic_client import TopicClient
from src.recommendation.RecommendationEngine import RecommendationEngine

app = FastAPI()
web_dir = Path(__file__).resolve().parents[1] / "web"
app.mount("/web", StaticFiles(directory=web_dir), name="web")


@app.middleware("http")
async def refresh_demo_assets(request, call_next):
    response = await call_next(request)
    if request.url.path == "/" or request.url.path.startswith("/web/"):
        response.headers["Cache-Control"] = "no-cache, must-revalidate"
    return response


@app.get("/", include_in_schema=False)
def web_demo():
    html = (web_dir / "index.html").read_text(encoding="utf-8")
    for asset in ("app.js", "styles.css"):
        version = hashlib.sha256((web_dir / asset).read_bytes()).hexdigest()[:16]
        html = html.replace(f'"/web/{asset}"', f'"/web/{asset}?v={version}"')
    return HTMLResponse(html)


class RecommendationRequest(BaseModel):
    news: list[str] = Field(min_length=1, max_length=200)
    history: list[str] = Field(min_length=1, max_length=200)

    @field_validator("news", "history")
    @classmethod
    def validate_titles(cls, values):
        titles = [value.strip() for value in values]
        if any(not title or len(title) > 2000 for title in titles):
            raise ValueError("Titles must contain 1 to 2000 characters")
        return list(dict.fromkeys(titles))

recommender = Recommender(
    semantic_client=SemanticClient(os.getenv("SEMANTIC_URL", "http://semantic:8001")),
    topic_client=TopicClient(os.getenv("TOPIC_URL", "http://topic:8002")),
    recommender_engine=RecommendationEngine(alpha=0.75)
)
    

@app.get("/recommender-health")
def health():
    return {"status": "recommender healthy"}

@app.post("/recommender-recommend")
def recommend(payload: RecommendationRequest):
    try:
        results = recommender.recommend(news=payload.news, history=payload.history)
    except requests.RequestException as exc:
        raise HTTPException(status_code=503, detail="Encoder service unavailable") from exc
    return {"results": results}


# Load once under a lock: concurrent initial requests must not duplicate the large cache.
from threading import Lock
from fastapi import Query
from demo.recommend.mind_store import MindStore

_mind_store = None
_mind_lock = Lock()


def mind_store():
    global _mind_store
    with _mind_lock:
        if _mind_store is None:
            try:
                _mind_store = MindStore()
            except (OSError, ValueError, KeyError) as exc:
                raise HTTPException(503, 'MIND data/vector artifacts unavailable or inconsistent') from exc
    return _mind_store


@app.get('/mind/metadata')
def mind_metadata():
    return mind_store().metadata()


@app.get('/mind/impressions')
def mind_impressions(offset: int = Query(0, ge=0), limit: int = Query(20, ge=1, le=100),
                     user: str = Query('', max_length=100)):
    return mind_store().impressions_page(offset, limit, user)


@app.get('/mind/impressions/{impression_id}')
def mind_impression(impression_id: str):
    try:
        return mind_store().impression(impression_id)
    except KeyError as exc:
        raise HTTPException(404, 'MIND impression not found') from exc


@app.post('/mind/impressions/{impression_id}/recommend')
def mind_recommend(impression_id: str):
    try:
        return mind_store().rank(impression_id)
    except KeyError as exc:
        raise HTTPException(404, 'MIND impression not found') from exc


@app.get('/mind/news')
def mind_news(offset: int = Query(0, ge=0), limit: int = Query(20, ge=1, le=100),
              q: str = Query('', max_length=200), category: str = Query('', max_length=100)):
    store = mind_store()
    rows = [n for n in store.news.values() if (not category or n['category'] == category)
            and (not q or q.lower() in n['title'].lower() or q.lower() == n['news_id'].lower())]
    return dict(total=len(rows), offset=offset, items=rows[offset:offset + limit])
