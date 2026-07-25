import random


class CandidateGenerator:
    def __init__(self, represented_vector):
        self.represented_vector = represented_vector

    def from_impression(self, impression):
        """Return candidate IDs from a MIND impression."""
        if isinstance(impression, str):
            items = impression.split()
        else:
            items = impression

        candidate_ids = []
        for item in items:
            if isinstance(item, dict):
                news_id = item["news_id"]
            else:
                news_id = item.rsplit("-", 1)[0]

            if news_id in self.represented_vector:
                candidate_ids.append(news_id)

        return list(dict.fromkeys(candidate_ids))

    def from_catalog(self, history, limit=None, random_state=42):
        """Generate candidates from the catalog, excluding read articles."""
        history_ids = set(history.split() if isinstance(history, str) else history)
        candidate_ids = [
            news_id
            for news_id in self.represented_vector
            if news_id not in history_ids
        ]

        if limit is not None:
            if limit <= 0:
                raise ValueError("Candidate limit must be greater than zero")
            if len(candidate_ids) > limit:
                candidate_ids = random.Random(random_state).sample(
                    candidate_ids,
                    limit,
                )

        return candidate_ids
