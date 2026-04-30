from __future__ import annotations

import hashlib
import math
import re


class HashingEmbeddingModel:
    """Deterministic local embedding for demos and tests.

    Production deployments can replace this class with OpenAI, Qwen, DeepSeek,
    or a sentence-transformer embedding model without changing retriever code.
    """

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        for token in self._tokens(text):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            bucket = int.from_bytes(digest[:4], "big") % self.dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[bucket] += sign

        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector
        return [value / norm for value in vector]

    @staticmethod
    def _tokens(text: str) -> list[str]:
        return re.findall(r"[a-zA-Z0-9_]+|[\u4e00-\u9fff]", text.lower())
