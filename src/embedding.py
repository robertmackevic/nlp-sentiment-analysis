from enum import Enum
from typing import List

from sentence_transformers import SentenceTransformer


class GTE(Enum):
    SMALL = "thenlper/gte-small"
    BASE = "thenlper/gte-base"
    LARGE = "thenlper/gte-large"


class EmbeddingModel:
    def __init__(self, model_version: GTE) -> None:
        self.model = SentenceTransformer(model_name_or_path=model_version.value)
        self.embedding_dim = self.model[1].word_embedding_dimension

    def get_embedding(self, text: str) -> List[float]:
        if len(text) == 0:
            raise ValueError("Input text is empty.")

        embedding = self.model.encode(text)
        return embedding.tolist()
