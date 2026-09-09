import requests

class TopicClient:
    def __init__(self, base_url: str):
        self.base_url = base_url

    def encode(
        self,
        texts: list[str],
        semantic_vectors: dict[str, list[float]],
    ) -> dict[str, list[float]]:
        response = requests.post(
            f"{self.base_url}/topic-encode",
            json={
                "texts": texts,
                "semantic_vectors": semantic_vectors
            },
            timeout=300
        )
        response.raise_for_status()

        return response.json()["vectors"]