import logging
import os
from pathlib import Path
from typing import List, Dict, Any, Union, Optional

# Project component imports
# pyrefly: ignore [missing-import]
from .text_formatter import TextFormatter
# pyrefly: ignore [missing-import]
from .semantic_chunker import SemanticChunker
# pyrefly: ignore [missing-import]
from .semantic_hasher import SemanticHasher

logger = logging.getLogger(__name__)

class ChunkingPipeline:
    """
    Unified Chunking & Hashing Pipeline.
    Orchestrates text formatting, semantic chunking, orthographic normalization,
    SHA-256 hashing, and deterministic ID assignment for documents.
    """

    def __init__(self) -> None:
        self.formatter = TextFormatter()
        self.semantic_chunker = SemanticChunker()
        self.hasher = SemanticHasher()

    def run(
        self, 
        full_text: str, 
        doc_id: Union[int, str], 
        cleaning_mode: str = "full"
    ) -> List[Dict[str, Any]]:
        """
        Executes the end-to-end chunking, formatting, and hashing process for a single document.

        Args:
            full_text (str): Raw Markdown text coming from the extraction/OCR stage.
            doc_id (int | str): Unique identifier of the document.
            cleaning_mode (str): 'light' for clean markdown or 'full' for heavy OCR cleaning.

        Returns:
            List[Dict[str, Any]]: Fully structured, hashed, and ID-assigned chunks ready for storage.
        """
        if not isinstance(full_text, str) or not full_text.strip():
            logger.warning("ChunkingPipeline: Received empty or invalid text for doc_id=%s", doc_id)
            return []

        logger.info("ChunkingPipeline: Starting processing for doc_id=%s", doc_id)

        # 1. Text Formatting & OCR Noise Cleaning
        if cleaning_mode.lower() == "light":
            formatted_text = self.formatter.light_format(full_text)
        else:
            formatted_text = self.formatter.full_clean_format(full_text)

        # 2. Semantic Chunking (Splitting into logical articles/sections)
        raw_chunks = self.semantic_chunker.create_chunks(formatted_text, doc_id)
        
        if not raw_chunks:
            logger.warning("ChunkingPipeline: SemanticChunker produced no chunks for doc_id=%s", doc_id)
            return []

        # 3. Layer 2: Arabic Normalization & SHA-256 Hashing
        hashed_chunks = self.hasher.process_layer_two(raw_chunks)

        # 4. Attach source metadata (chunk_id already set by SemanticHasher)
        for chunk in hashed_chunks:
            chunk['source'] = doc_id

        logger.info(
            "ChunkingPipeline: Successfully processed and finalized %d chunks for doc_id=%s",
            len(hashed_chunks),
            doc_id
        )
        return hashed_chunks

    def process_folder(
        self, 
        folder_path: Union[str, Path], 
        cleaning_mode: str = "full"
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Loops through all Markdown (.md) files in a given folder,
        processes each file through the pipeline, and returns a dictionary 
        mapping each document ID to its corresponding processed chunks list.

        Args:
            folder_path (str | Path): Path to the folder containing markdown files.
            cleaning_mode (str): 'light' or 'full' cleaning level.

        Returns:
            Dict[str, List[Dict[str, Any]]]: Dictionary where keys are doc_ids and values are lists of chunk dicts.
        """
        folder = Path(folder_path)
        if not folder.exists() or not folder.is_dir():
            logger.error("Folder path not found or is not a directory: %s", folder_path)
            return {}

        results: Dict[str, List[Dict[str, Any]]] = {}
        markdown_files = list(folder.glob("*.md"))
        
        logger.info("Found %d Markdown files in '%s'. Starting batch folder processing...", len(markdown_files), folder_path)

        for file_path in markdown_files:
            # Use file name (without extension) as the document ID
            doc_id = file_path.stem 
            
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    full_text = f.read()

                # Run the pipeline for the current file
                chunks = self.run(full_text=full_text, doc_id=doc_id, cleaning_mode=cleaning_mode)
                
                results[doc_id] = chunks
                logger.info("Successfully processed file: %s (%d chunks generated)", file_path.name, len(chunks))

            except Exception as e:
                logger.error("Error processing file %s: %s", file_path.name, e)

        logger.info("Folder processing completed successfully. Total processed documents: %d", len(results))
        return results