from abc import ABC, abstractmethod
from typing import Dict, List

class Recognizer(ABC):

    @abstractmethod
    def predict(self, audio_directory: str, utterances: List[str], *args, **kwargs) -> Dict[str, str]:
        pass