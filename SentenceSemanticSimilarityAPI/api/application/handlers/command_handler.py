from typing import Dict
from sentence_transformers.util import pytorch_cos_sim, semantic_search
from api.domain import InputCosineSimilarity, InputSemanticSearch, \
    SemanticSearch, OutputCosineSimilarity, OutputSemanticSearch
from api.application.handlers.text_preprocessing import TextPreprocessing
from sentence_transformers import SentenceTransformer


class SemanticSimilarityCommandHandler:
    def __init__(self, models: Dict[str, SentenceTransformer]):
        self._models = models['1']

    def handle_cosine_similarity(self, input_sentences: InputCosineSimilarity) -> OutputCosineSimilarity:
        # cleaning
        clean_sentence1 = TextPreprocessing(input_sentences.sentence1).clear_text()
        clean_sentence2 = TextPreprocessing(input_sentences.sentence2).clear_text()

        # to tensor
        source_embedding = self._models.encode(clean_sentence1, convert_to_tensor=True)
        target_embedding = self._models.encode(clean_sentence2, convert_to_tensor=True)

        # cosine similarity score
        score = pytorch_cos_sim(source_embedding, target_embedding)
        result = score.cpu().numpy()[0][0]

        return OutputCosineSimilarity(cosine_similarity_score=result)

    def handle_semantic_search(self, input_sentences: InputSemanticSearch) -> OutputSemanticSearch:
        result = []

        # cleaning
        clean_sentence = TextPreprocessing(input_sentences.sentence).clear_text()
        clean_batch = [TextPreprocessing(sentence).clear_text() for sentence in input_sentences.batch]

        # to tensor
        query = self._models.encode(clean_sentence, convert_to_tensor=True)
        batch = self._models.encode(clean_batch, convert_to_tensor=True)

        # semantic search
        search = semantic_search(query_embeddings=query, corpus_embeddings=batch, top_k=input_sentences.topn)
        for s in search[0]:
            result.append(SemanticSearch(
                semantic_search={"sentence": clean_batch[s['corpus_id']], "score": str(s['score'])}))

        return OutputSemanticSearch(iterations=result)



