"""
src/rag_database/embedder.py

RAGEmbedder: Dense vector embedding generator for the Zar3a RAG pipeline.
Uses multilingual-e5-large (or any configured HuggingFace sentence-transformer model)
to produce 1024-dimensional embeddings with E5-style query/passage prefixing.
"""

import logging
from typing import List

import torch
from transformers import AutoTokenizer, AutoModel

from src.rag_database.config import rag_config

logger = logging.getLogger("Zar3a.RAGEmbedder")


class RAGEmbedder:
    """
    Generates dense vector embeddings using a HuggingFace transformer model.

    Supports E5-style instruction prefixing:
      - Passages (documents): prefixed with "passage: "
      - Queries (search): prefixed with "query: "
    """

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or rag_config.EMBEDDING_MODEL_NAME
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(
            "Loading embedding model '%s' on device '%s'...",
            self.model_name,
            self.device,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        # low_cpu_mem_usage=False prevents PyTorch from placing weights on a
        # "meta" device before copying to the target, which raises:
        #   NotImplementedError: Cannot copy out of meta tensor; no data!
        self.model = AutoModel.from_pretrained(
            self.model_name, low_cpu_mem_usage=False
        ).to(self.device)
        self.model.eval()
        logger.info("Embedding model loaded successfully.")

    @staticmethod
    def _average_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Mean-pool token embeddings, ignoring padding tokens."""
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    def _add_prefix(self, texts: List[str], is_query: bool) -> List[str]:
        """Prepend E5 instruction prefix to each text."""
        prefix = "query: " if is_query else "passage: "
        return [f"{prefix}{t}" for t in texts]

    @torch.no_grad()
    def embed_texts(self, texts: List[str], is_query: bool = False) -> List[List[float]]:
        """
        Generates normalized dense embeddings for a list of texts.

        Args:
            texts: List of raw text strings to embed.
            is_query: If True, applies 'query: ' prefix (for retrieval);
                      if False, applies 'passage: ' prefix (for indexing).

        Returns:
            List of float lists — one embedding vector per input text.
        """
        if not texts:
            return []

        prefixed = self._add_prefix(texts, is_query=is_query)

        batch_dict = self.tokenizer(
            prefixed,
            max_length=512,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        batch_dict = {k: v.to(self.device) for k, v in batch_dict.items()}

        outputs = self.model(**batch_dict)
        embeddings = self._average_pool(outputs.last_hidden_state, batch_dict["attention_mask"])

        # L2 normalise for cosine similarity
        import torch.nn.functional as F
        embeddings = F.normalize(embeddings, p=2, dim=1)

        return embeddings.cpu().tolist()

    def embed_query(self, query: str) -> List[float]:
        """Convenience method for embedding a single search query."""
        results = self.embed_texts([query], is_query=True)
        return results[0] if results else []