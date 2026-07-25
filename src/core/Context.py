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

        return behaviours
    
    def createImpressed(self):
        impressions_path = os.path.join(self.destination, "behaviors.tsv")

        columns_impressions = ["user_id", "time", "history", "impressions"]

        impressions_df = pd.read_csv(
            impressions_path,
            sep="\t",
            names=columns_impressions,
        )

        impressions_df = impressions_df[["user_id", "impressions"]]
        impressions_df = impressions_df.dropna(subset=["impressions"])
        
        impressions_df = impressions_df.to_dict(orient="records")
        
        self.impressions = {}
        
        for i in impressions_df:
            user_id = i["user_id"]
            temp = i["impressions"].split(" ")
            news_list = []
            
            for j in temp:
                news_id, label = j.split("-")
                
                news_list.append(
                    {
                        "news_id": news_id,
                        "label": int(label)
                    }
                )
                
            
            self.impressions.setdefault(i["user_id"], []).append(news_list)
            # print(self.impressions[user_id])
            
        return self.impressions
        
    
context = VectorContext()
context.createTitleList()
context.createHistory()
context.createImpressed()