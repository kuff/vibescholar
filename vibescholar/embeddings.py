from __future__ import annotations

import numpy as np

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"


class Embedder:
    """Text embedder supporting CPU (via fastembed) and GPU (via ONNX Runtime direct).

    When ``cuda=True``, uses ONNX Runtime with CUDAExecutionProvider directly,
    bypassing fastembed's wrapper which has tokenizer bugs on the GPU path.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        cuda: bool = False,
        device_ids: list[int] | None = None,
    ):
        self.model_name = model_name
        self._dim: int | None = None

        if cuda:
            self._backend = _OnnxGpuBackend(model_name, device_ids)
        else:
            self._backend = _FastEmbedBackend(model_name)

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

        vectors = self._backend.embed(texts)
        if self._dim is None and vectors.size:
            self._dim = int(vectors.shape[1])
        return vectors


class _FastEmbedBackend:
    """CPU embedding via fastembed (default)."""

    def __init__(self, model_name: str):
        from fastembed import TextEmbedding
        self._model = TextEmbedding(model_name=model_name)

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.asarray(list(self._model.embed(texts)), dtype=np.float32)


class _OnnxGpuBackend:
    """GPU embedding via ONNX Runtime + HuggingFace tokenizers directly."""

    def __init__(self, model_name: str, device_ids: list[int] | None = None):
        import onnxruntime as ort
        from tokenizers import Tokenizer
        from huggingface_hub import hf_hub_download

        device_id = device_ids[0] if device_ids else 0

        # Download model files from HuggingFace
        onnx_path = hf_hub_download(model_name, "onnx/model.onnx")
        tokenizer_path = hf_hub_download(model_name, "tokenizer.json")

        self._tokenizer = Tokenizer.from_file(tokenizer_path)
        self._tokenizer.enable_padding()
        self._tokenizer.enable_truncation(max_length=512)

        providers = [
            ("CUDAExecutionProvider", {"device_id": device_id}),
            "CPUExecutionProvider",
        ]
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._session = ort.InferenceSession(onnx_path, sess_options, providers=providers)

        actual = self._session.get_providers()
        print(f"  ONNX providers: {actual}")

    def embed(self, texts: list[str], batch_size: int = 256) -> np.ndarray:
        all_embeddings: list[np.ndarray] = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            encoded = self._tokenizer.encode_batch(batch)

            input_ids = np.array([e.ids for e in encoded], dtype=np.int64)
            attention_mask = np.array([e.attention_mask for e in encoded], dtype=np.int64)
            token_type_ids = np.zeros_like(input_ids)

            outputs = self._session.run(
                None,
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "token_type_ids": token_type_ids,
                },
            )

            # Mean pooling over token embeddings with attention mask
            token_embeddings = outputs[0]  # (batch, seq_len, dim)
            mask_expanded = attention_mask[:, :, np.newaxis].astype(np.float32)
            summed = (token_embeddings * mask_expanded).sum(axis=1)
            counts = mask_expanded.sum(axis=1).clip(min=1e-9)
            embeddings = summed / counts

            # L2 normalize
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True).clip(min=1e-9)
            embeddings = embeddings / norms

            all_embeddings.append(embeddings.astype(np.float32))

        return np.concatenate(all_embeddings, axis=0)

