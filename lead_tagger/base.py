from abc import ABC, abstractmethod
import pandas as pd

class BaseTagger(ABC):
    def __init__(self, df: pd.DataFrame, tag_mapping: dict[str, list[str]], priority_list: list[str] = None):
        self.df = df.copy()
        self.tag_mapping = tag_mapping
        self.priority_list = priority_list if priority_list else []

    def apply_tags(self) -> pd.DataFrame:
        self.df["tags"] = self.df.apply(self.generate_tags, axis=1)
        return self.df

    @abstractmethod
    def generate_tags(self, row: pd.Series) ->str:
        """Return a string of comma separated tags based on the row's intent columns"""
        pass
    
    @staticmethod
    def get_description(tagger_type: str) -> str:
        """Static method to return description based on tagger type"""
        descriptions = {
            "Standard Tagger": "Tags leads based on the intent columns provided, adding multiple (unique) tags if multiple intents are detected.",
            "Custom Tagger #1": "Standard Tagger with custom logic that ensures only 1 tag per lead, and custom priority in the case of multiple intents.",
        }
        return descriptions.get(tagger_type, "No description available.")