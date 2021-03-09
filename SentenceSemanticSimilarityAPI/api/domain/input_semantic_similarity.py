from typing import List
from pydantic import BaseModel


class InputCosineSimilarity(BaseModel):
    sentence1: str
    sentence2: str


class InputSemanticSearch(BaseModel):
    sentence: str
    topn: int
    batch: List[str]