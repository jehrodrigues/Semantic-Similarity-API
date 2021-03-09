from typing import Dict
from .recognizer import Recognizer
from sentence_transformers import SentenceTransformer


class RecognizerBuilder:
    @classmethod
    def from_config(cls, config: Dict, has_gpu: bool = False) -> Recognizer:
        # uses the first model in the queue
        if config.queue == "1":
            if config.framework == "sentence_transformers":
                return cls._build_sentence_transformers_recognizer(config, has_gpu)

        raise Exception("Framework option does not exist.")

    @staticmethod
    def _build_sentence_transformers_recognizer(config: Dict, has_gpu: bool=False) -> Recognizer:
        return SentenceTransformer(config.location)

        raise Exception("Sentence Transformer model type does not exist")
