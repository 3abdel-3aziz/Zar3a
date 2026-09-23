from abc import ABC, abstractmethod
from typing import List, Dict, Any


class BaseChunker(ABC):
    """
    Abstract base class (interface) for all chunking strategies.

    Every chunker receives the full Markdown text produced by an extractor and
    returns a list of chunk dicts. The pipeline orchestrator depends only on this
    interface, never on concrete implementations.
    """

    @abstractmethod
    def create_chunks(self, full_text: str, doc_id: str) -> List[Dict[str, Any]]:
        """
        Split a full Markdown document into a list of structured chunk dicts.

        Args:
            full_text (str): The complete Markdown string from the extractor.
            doc_id (str):    The database ID of the parent document, embedded in
                             every returned chunk for traceability.

        Returns:
            List[Dict[str, Any]]: Each dict must contain at minimum:
                - ``doc_id``      (str)  – parent document identifier
                - ``chunk_index`` (int)  – zero-based position within this version
                - ``content``     (str)  – the chunk's textual content
                - ``metadata``    (dict) – arbitrary strategy-specific metadata
        """
        ...