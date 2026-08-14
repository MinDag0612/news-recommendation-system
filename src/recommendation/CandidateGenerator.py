class CandidateGenerator:

    def __init__(self, available_news):
        if isinstance(available_news, dict):
            available_news = available_news.keys()
        self.available_news = tuple(dict.fromkeys(available_news))
        self._available_news = set(self.available_news)

    @staticmethod
    def _news_ids(news):
        if news is None:
            return set()
        if isinstance(news, str):
            return set(news.split())
        return set(news)

    def generate(self, user_history=None, candidate_news=None):
        # Return unique, known, unread news IDs while preserving input order
        read_news = self._news_ids(user_history)
        source = self.available_news if candidate_news is None else candidate_news

        candidates = []
        seen = set()
        for news_id in source:
            if (
                news_id in self._available_news
                and news_id not in read_news
                and news_id not in seen
            ):
                candidates.append(news_id)
                seen.add(news_id)
        return candidates

    def from_impressions(self, impressions):
        # Return valid labelled candidates from a MIND impression list
        candidates = []
        seen = set()
        for item in impressions:
            news_id = item["news_id"]
            if news_id in self._available_news and news_id not in seen:
                candidates.append({"news_id": news_id, "label": int(item["label"])})
                seen.add(news_id)
        return candidates
