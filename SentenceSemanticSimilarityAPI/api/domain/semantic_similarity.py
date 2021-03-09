from typing import Dict
from pydantic import BaseModel


class SemanticSearch(BaseModel):
    semantic_search: Dict[str, str]
