from fastapi import FastAPI

from api.configuration_manager import ConfigurationManager
from api.domain import InputCosineSimilarity, InputSemanticSearch, OutputCosineSimilarity, OutputSemanticSearch

app = FastAPI()
semantic_similarity_handler = ConfigurationManager.semantic_similarity_command_handler()


@app.post("/cosine_similarity/", response_model=OutputCosineSimilarity, status_code=200, tags=["Semantic Similarity Score"])
def semantic_similarity(payload: InputCosineSimilarity):
    return semantic_similarity_handler.handle_cosine_similarity(payload)


@app.post("/semantic_search/", response_model=OutputSemanticSearch, status_code=200, tags=["Semantic Search"])
def semantic_search(payload: InputSemanticSearch):
    return semantic_similarity_handler.handle_semantic_search(payload)
