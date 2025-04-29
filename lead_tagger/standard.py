from .base import BaseTagger
import pandas as pd

class StandardTagger(BaseTagger):
    def generate_tags(self, row: pd.Series) -> str:
        tags: set = set()

        for col, tags_list in self.tag_mapping.items():
            if pd.notna(row.get(col)):
                tags.update(tags_list)

        return ', '.join(sorted(tags))
