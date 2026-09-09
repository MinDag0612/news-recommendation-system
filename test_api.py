import requests


texts = [
    "NFL Week 9 Power Rankings: More ammo for Belichick as greatest coach ever",
    "Former NBA first-round pick Jim Farmer arrested in sex sting operation",
    "Clippers set bad precedent resting Kawhi Leonard against Bucks",
    "The Latest: Kansas City to remove King's name from street",
    "7 biggest takeaways from the opening College Football Playoff rankings",
    "Pregnant U.S. women's soccer star Alex Morgan still plans to play in 2020 Summer Olympics"
]


# 1. Call Semantic Service
semantic_response = requests.post(
    "http://localhost:8001/semantic-encode",
    json=texts
)

semantic_response.raise_for_status()

semantic_vectors = semantic_response.json()["vectors"]


# 2. Call Topic Service
topic_response = requests.post(
    "http://localhost:8002/topic-encode",
    json={
        "texts": texts,
        "semantic_vectors": semantic_vectors
    }
)

topic_response.raise_for_status()

topic_result = topic_response.json()

print(topic_result)