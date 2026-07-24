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

        title_list = (
            self.news.rename(columns={
                "News_ID": "news_id",
                "Title": "title"
            })[["news_id", "title"]]
            .to_dict("records")
        )

        self.title_list = title_list
        return title_list
    
    def createHistory(self):
        behaviours_path = os.path.join(self.destination, "behaviors.tsv")

        columns_behaviours = ["user_id", "time", "history", "impressions"]

        behaviours = pd.read_csv(
            behaviours_path,
            sep="\t",
            names=columns_behaviours,
        )

        behaviours = behaviours[["user_id", "history"]]
        behaviours = behaviours.dropna(subset=["history"])

        # sample_behaviours = behaviours.sample(n=10, random_state=42).reset_index(drop=True)

        output_dir = r"D:\CDNC\MIND-research\data\sample"

        behaviours.to_csv(os.path.join(output_dir, "behaviours.csv"), index=False)

        return behaviours
    
context = VectorContext()
context.createTitleList()
context.createHistory()