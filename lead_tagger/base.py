from abc import ABC, abstractmethod
import pandas as pd

class BaseTagger(ABC):
    def __init__(self, df: pd.DataFrame, tag_mapping: dict[str, list[str]]):
        self.df = df.copy()
        self.tag_mapping = tag_mapping

    def apply_tags(self) -> pd.DataFrame:
        self.df["tags"] = self.df.apply(self.generate_tags, axis=1)
        return self.df

    @abstractmethod
    def generate_tags(self, row: pd.Series) -> list[str]:
        """Return list of tags for a given row."""
        pass
    
    @staticmethod
    def get_description(tagger_type: str) -> str:
        """Static method to return description based on tagger type"""
        descriptions = {
            "Standard Tagger": "Tags leads based on the intent columns provided, adding multiple tags if multiple intents are detected.",
        }
        return descriptions.get(tagger_type, "No description available.")