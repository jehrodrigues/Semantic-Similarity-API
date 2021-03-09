import json

from typing import Dict

import dacite

from api.application.handlers.command_handler import \
    SemanticSimilarityCommandHandler
from api.application.recognizer_model.sentence_transformers_builder import SentenceTransformerBuilder
from api.application.recognizer_model.sentence_transformers_container_location import \
    SentenceTransformerContainerLocation


class ConfigurationManager:

    @classmethod
    def configuration(cls) -> Dict:
        if not hasattr(cls, "_config"):
            with open("config.json", "r") as configuration_file:
                cls._config = json.load(configuration_file)

        return cls._config

    @classmethod
    def has_gpu_and_cuda_available(cls) -> bool:
        configuration = cls.configuration()
        if not hasattr(cls, '_has_gpu_and_cuda'):
            gpu_available = configuration["gpu_available"]
            cls._has_gpu_and_cuda = gpu_available #and cuda_available()
        return cls._has_gpu_and_cuda


    @classmethod
    def model_locations(cls):
        if not hasattr(cls, "_model_locations"):
            cls._model_locations = [
                dacite.from_dict(SentenceTransformerContainerLocation, data=model_config)
                for model_config in cls.configuration()["models"]
            ]

        return cls._model_locations


    @classmethod
    def sentence_transformer_models(cls):
        locations = cls.model_locations()
        models = {}
        for model_location in locations:
            models[model_location.queue] = SentenceTransformerBuilder.from_config(model_location, cls.has_gpu_and_cuda_available())

        return models

    @classmethod
    def semantic_similarity_command_handler(cls) -> SemanticSimilarityCommandHandler:
        if not hasattr(cls, "_semantic_similarity_handler"):
            cls._semantic_similarity_handler = SemanticSimilarityCommandHandler(
                models=cls.sentence_transformer_models(),
            )

        return cls._semantic_similarity_handler
