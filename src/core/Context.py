import os
import pandas as pd


class VectorContext:
    def __init__(self):
        self.title_list = None
        self.destination = r"D:\CDNC\MIND-research\data\raw"

    def createTitleList(self):

        news_path = os.path.join(self.destination, "news.tsv")

        columns = [
            "News_ID",
            "Category",
            "SubCategory",
            "Title",
            "Abstract",
            "URL",
            "Title_Entities",
            "Abstract_Entities",
        ]

        self.news = pd.read_csv(
            news_path,
            sep="\t",
            names=columns,
        )

        self.news = self.news[["News_ID", "Category", "Title"]]

        title_list = self.news.rename(columns={"News_ID": "news_id", "Title": "title"})[
            ["news_id", "title"]
        ].to_dict("records")

        self.title_list = title_list
        return title_list

    def createImpressed(self):
        impressions_path = os.path.join(self.destination, "behaviors.tsv")

        columns_impressions = [
            "behavior_id",
            "user_id",
            "time",
            "history",
            "impressions",
        ]

        impressions_df = pd.read_csv(
            impressions_path,
            sep="\t",
            names=columns_impressions,
        )

        impressions_df = impressions_df[
            ["behavior_id", "user_id", "history", "impressions"]
        ]
        
        impressions_df = impressions_df.dropna(subset=["history"])
        impressions_df = impressions_df.dropna(subset=["impressions"])
        impressions_df = impressions_df.dropna(subset=["impressions"])

        impressions_df = impressions_df.to_dict(orient="records")

        self.impressions = {}

        for impress in impressions_df:
            behavior_id = impress["behavior_id"]
            user_id = impress["user_id"]
            history = impress["history"]
            impression_list = impress["impressions"].split(" ")
            news_list = []

            for j in impression_list:
                news_id, label = j.split("-")
                impression_item = {
                    "news_id": news_id,
                    "label": int(label),
                }

                news_list.append(impression_item)
            
            behavior_row = {
                "user_id": user_id,
                "history": history,
                "impressions": news_list
            }

            self.impressions[behavior_id] = behavior_row
            # print(self.impressions[user_id])

        return self.impressions


context = VectorContext()
context.createTitleList()
context.createImpressed()
