import requests

class SemanticClient:
    def __init__(self, base_url: str):
        self.base_url = base_url

    def encode(self, texts: list[str]) -> dict[str, list[float]]:
        response = requests.post(
            f"{self.base_url}/semantic-encode",
            json=texts,
            timeout=300
        )
        response.raise_for_status()

        return response.json()["vectors"]