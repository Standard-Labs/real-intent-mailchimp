""" Custom Tagger for Jason, with custom logic that ensures only 1 tag per lead, and custom priority if multiple intents are detected. """
from .base import BaseTagger
import pandas as pd

class Custom_1_Tagger(BaseTagger):
    def generate_tags(self, row: pd.Series) -> str:
        tags: set = set()

        for col, tags_list in self.tag_mapping.items():
            if pd.notna(row.get(col)):
                tags.update(tags_list)

        sorted_tags = sorted(tags, key=lambda x: self.priority_list.index(x))

        return sorted_tags[0] if sorted_tags else ''
