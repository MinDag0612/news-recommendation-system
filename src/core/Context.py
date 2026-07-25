from pathlib import Path
import pandas as pd

class VectorContext:
    def __init__(self, destination=None):
        self.title_list = None
        project_root = Path(__file__).resolve().parents[2]
        self.destination = Path(destination) if destination else project_root / "data" / "raw"
        
    def createTitleList(self):
        
        news_path = self.destination / "news.tsv"

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
        behaviours_path = self.destination / "behaviors.tsv"

        columns_behaviours = [
            "impression_id",
            "user_id",
            "time",
            "history",
            "impressions",
        ]

        behaviours = pd.read_csv(
            behaviours_path,
            sep="\t",
            names=columns_behaviours,
        )

        behaviours = behaviours.dropna(subset=["history"])

        self.behaviours = behaviours
        return self.behaviours
    
    def createImpressed(self):
        if not hasattr(self, "behaviours"):
            self.createHistory()

        self.impressions = self.behaviours.dropna(subset=["impressions"]).copy()
        self.impressions["candidates"] = self.impressions["impressions"].map(
            self.parseImpressions
        )
        return self.impressions

    @staticmethod
    def parseImpressions(impressions):
        candidates = []
        for item in impressions.split():
            news_id, label = item.rsplit("-", 1)
            candidates.append({
                "news_id": news_id,
                "label": int(label),
            })
        return candidates
        
    
context = VectorContext()
context.createTitleList()
context.createHistory()
context.createImpressed()
