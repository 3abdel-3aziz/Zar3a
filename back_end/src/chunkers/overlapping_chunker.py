from typing import List, Dict, Any

# pyrefly: ignore [missing-import]
from .base_chunker import BaseChunker


class OverlappingChunker(BaseChunker):
    """
    Fixed-size overlapping window chunker (migrated from the original pipeline).

    Splits the full text every ``chunk_size`` characters, backing up by
    ``overlap`` characters so that no sentence is cut dead at a boundary.
    Best for general-purpose RAG where dense retrieval matters more than
    precise article alignment.
    """

    def __init__(
        self,
        chunk_size: int = 600,
        overlap: int = 100,
        min_chunk_length: int = 50,
    ) -> None:
        """
        Args:
            chunk_size (int):       Maximum character length per chunk.
            overlap (int):          How many characters of the previous chunk
                                    to repeat at the start of the next.
            min_chunk_length (int): Chunks shorter than this are merged into
                                    the previous one instead of being kept as
                                    standalone entries.
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.min_chunk_length = min_chunk_length

    # ------------------------------------------------------------------
    # BaseChunker contract
    # ------------------------------------------------------------------

    def create_chunks(self, full_text: str, doc_id: str) -> List[Dict[str, Any]]:
        """
        Split *full_text* into overlapping windows.

        Args:
            full_text (str): The complete Markdown string from the extractor.
            doc_id (str):    Parent document identifier.

        Returns:
            List[Dict[str, Any]]: Chunk dicts with keys
                ``doc_id``, ``chunk_index``, ``content``, ``metadata``.
        """
        text = full_text.strip()
        print(
            f"[*] OverlappingChunker: Splitting {len(text):,} chars "
            f"(window={self.chunk_size}, overlap={self.overlap})…"
        )

        # Edge case: entire document fits in one chunk
        if len(text) <= self.chunk_size:
            return [
                {
                    "doc_id": doc_id,
                    "chunk_index": 0,
                    "content": text,
                    "metadata": {"type": "full_document_block"},
                }
            ]

        blocks: List[Dict[str, Any]] = []
        step = self.chunk_size - self.overlap
        idx = 0

        for start in range(0, len(text), step):
            chunk_content = text[start : start + self.chunk_size]

            # Merge tiny tail-end fragments into the previous block
            if len(chunk_content) < self.min_chunk_length and blocks:
                blocks[-1]["content"] += " " + chunk_content
            else:
                blocks.append(
                    {
                        "doc_id": doc_id,
                        "chunk_index": idx,
                        "content": chunk_content,
                        "metadata": {"type": "overlapping_block"},
                    }
                )
                idx += 1

        print(f"[+] OverlappingChunker: {len(blocks)} chunks created.")
        return blocks