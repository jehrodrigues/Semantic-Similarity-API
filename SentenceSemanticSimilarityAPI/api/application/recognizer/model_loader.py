from typing import Dict

from .recognizer_builder import RecognizerBuilder


class RecognizerLoader:

    def __init__(self, model_locations: Dict[str, Dict]):
        self._model_locations = model_locations

    def get_model(self, model_name: str):
        if model_name in self._model_locations:
            return RecognizerBuilder.from_config(self._model_locations[model_name])

        return
