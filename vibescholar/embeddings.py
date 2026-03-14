from __future__ import annotations

import numpy as np
from fastembed import TextEmbedding

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"


class Embedder:
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        cuda: bool = False,
        device_ids: list[int] | None = None,
    ):
        self.model_name = model_name
        kwargs: dict = {}
        if cuda:
            kwargs["cuda"] = True
            if device_ids is not None:
                kwargs["device_ids"] = device_ids
        self._model = TextEmbedding(model_name=model_name, **kwargs)
        self._dim: int | None = None

    @property
    def dimension(self) -> int:
        if self._dim is None:
            self._dim = int(self.embed_texts(["dimension probe"]).shape[1])
        return self._dim

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        if not texts:
            if self._dim is None:
                return np.empty((0, 0), dtype=np.float32)
            return np.empty((0, self._dim), dtype=np.float32)

        vectors = np.asarray(list(self._model.embed(texts)), dtype=np.float32)
        if self._dim is None and vectors.size:
            self._dim = int(vectors.shape[1])
        return vectors

