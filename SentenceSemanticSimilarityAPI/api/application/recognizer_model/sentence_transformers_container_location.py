from dataclasses import dataclass

@dataclass
class SentenceTransformerContainerLocation:
    queue: str
    name: str
    location: str
    framework: str