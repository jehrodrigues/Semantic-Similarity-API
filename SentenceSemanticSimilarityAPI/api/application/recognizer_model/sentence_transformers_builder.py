from typing import Dict
from sentence_transformers import SentenceTransformer


class SentenceTransformerBuilder:
    @classmethod
    def from_config(cls, config: Dict, has_gpu: bool = False) -> SentenceTransformer:
        # uses the first sentence_transformer_model in the queue
        if config.queue == "1":
            if config.framework == "sentence_transformers":
                return cls._build_sentence_transformers(config, has_gpu)

        raise Exception("Framework option does not exist.")

    @staticmethod
    def _build_sentence_transformers(config: Dict, has_gpu: bool=False) -> SentenceTransformer:
        return SentenceTransformer(config.location)

        raise Exception("Sentence Transformer recognizer_model type does not exist")
