from typing import List
from pydantic import BaseModel
from .semantic_similarity import SemanticSearch


class OutputCosineSimilarity(BaseModel):
    cosine_similarity_score: float


class OutputSemanticSearch(BaseModel):
    iterations: List[SemanticSearch]
