import pandas as pd

from src.core.Vector import Vector

class RepresentedVector(Vector):
    def __init__(self, title_list):
        self.title_list = title_list
        self.represented_vector = {}

    def get_vector(self, sentence_dict, bertopic_dict):
        self.represented_vector = {
            news["news_id"]: {
                "title": news["title"],
                "semantic": sentence_dict[news["news_id"]],
                # "topic": self.bertopic_dict[news["news_id"]]["topic"],
                "topic_distribution": bertopic_dict[news["news_id"]][
                    "probability"
                ],
            }
            for news in self.title_list
        }

        return self.represented_vector

    def overview(self):
        print("=" * 80)
        print("Represented Vector Overview")
        print("=" * 80)

        print(f"Documents           : {len(self.represented_vector)}")

        first_news_id = next(iter(self.represented_vector))

        sample = self.represented_vector[first_news_id]

        print(f"Semantic Dimension  : {len(sample['semantic'])}")
        print(f"Topic Distribution  : {len(sample['topic_distribution'])}")
        print(f"Stored Fields       : {list(sample.keys())}")

        print("=" * 80)

    # def preview_vector(vector, preview_dims=4):
    #     vector = [round(float(x), 4) for x in vector]

    #     if len(vector) <= preview_dims * 2:
    #         return vector

    #     return vector[:preview_dims] + ["..."] + vector[-preview_dims:]

    def summary(self, sample_index=0, preview_dims=4):

        news = self.title_list[sample_index]
        news_id = news["news_id"]

        represented = self.represented_vector[news_id]

        semantic = represented["semantic"]
        probability = represented["topic_distribution"]

        def preview(vector):
            vector = [round(float(x), 4) for x in vector]

            if len(vector) <= preview_dims * 2:
                return vector

            return vector[:preview_dims] + ["..."] + vector[-preview_dims:]

        semantic_preview = preview(semantic)
        probability_preview = preview(probability)

        summary_df = pd.DataFrame(
            {
                "Field": [
                    "News ID",
                    "Title",
                    # "Assigned Topic",
                    "Semantic Dimension",
                    "Topic Distribution Dimension",
                ],
                "Value": [
                    news_id,
                    represented["title"],
                    # represented["topic"],
                    len(semantic),
                    len(probability),
                ],
            }
        )

        represented_preview = {
            news_id: {
                "title": represented["title"],
                "semantic": semantic_preview,
                # "topic": represented["topic"],
                "topic_distribution": probability_preview,
            }
        }

        return summary_df, represented_preview