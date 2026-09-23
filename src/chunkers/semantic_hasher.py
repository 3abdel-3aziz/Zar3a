import hashlib
import logging
import re
from typing import List, Dict, Any
# pyrefly: ignore [missing-import]
from .chunk_id_generator import AgriculturalLawParser

logger = logging.getLogger(__name__)


class SemanticHasher:
    """
    Layer 2: Handles Arabic-specific text normalization and semantic hashing.
    Ensures that orthographic variations (like Alef shapes) don't trigger false diffs.
    """

    def __init__(self) -> None:
        # Initialize the specialized agricultural law article parser
        self.article_parser: AgriculturalLawParser = AgriculturalLawParser()

    def process_layer_two(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Refines a batch of raw chunks by normalizing text, extracting article 
        numbers to build deterministic IDs, and calculating robust SHA-256 hashes.

        Args:
            chunks (List[Dict[str, Any]]): The list of raw chunk dictionaries from Layer 1.

        Returns:
            List[Dict[str, Any]]: The mutation-safe, enriched chunks ready for next steps.
        """
        logger.info("Layer 2: Normalizing and assigning IDs to %d chunks...", len(chunks))
        
        for idx, chunk in enumerate(chunks):
            # Identity Check (Table vs Text)
            is_table: bool = chunk.get('type') == 'table'
            
            # Extract and normalize text to strip orthographic noise
            raw_content: str = chunk.get('content', '')
            normalized_content: str = self._normalize_text(raw_content, is_table=is_table)
            chunk['content'] = normalized_content
            
            doc_id: str = chunk.get('doc_id', 'unknown_doc')
            
            # Uniformly extract the article number for all chunks
            article_num: str = self.article_parser.extract_article_id(normalized_content)
            
            # Use the article number if valid, otherwise fallback to sequence index
            if article_num and article_num != "0":
                chunk['chunk_id'] = f"{doc_id}_art_{article_num}"
            else:
                chunk['chunk_id'] = f"{doc_id}_chunk_{idx}"
            
            # Generate accurate hash for each chunk based on the normalized text
            chunk['chunk_hash'] = self._generate_hash(normalized_content)

        logger.info("Layer 2 complete: Hashes and deterministic legal IDs assigned.")
        return chunks         

    def _normalize_text(self, text: str, is_table: bool = False) -> str:
        if not text: 
            return ""
            
        # Standardize whitespace based on content type
        if is_table:
            text = re.sub(r'[ \t]+', ' ', text)
        else:
            text = re.sub(r'\s+', ' ', text)
        
        # Arabic Normalization (Standardizing Alef, Ta-Marbuta, and Ya)
        text = re.sub(r'[إأآ]', 'ا', text)
        text = re.sub(r'ة\b', 'ه', text)
        text = re.sub(r'[يى]\b', 'ي', text)
        
        return text.strip()

    def _generate_hash(self, text: str) -> str:
        return hashlib.sha256(text.encode('utf-8')).hexdigest()