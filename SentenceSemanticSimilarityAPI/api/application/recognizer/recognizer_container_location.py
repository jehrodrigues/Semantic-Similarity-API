from dataclasses import dataclass

@dataclass
class RecognizerContainerLocation:
    queue: str
    name: str
    location: str
    framework: str