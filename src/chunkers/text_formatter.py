# src/document_processor/chunkers/text_formatter.py

import re

class TextFormatter:
    """
    Standardizes and cleans text extracted from OCR engines.
    Provides different levels of cleaning based on the source quality.
    """

    def __init__(self) -> None:
        # Pattern to identify legal articles 
        self.article_pattern = re.compile(
            r'^(مادة|المادة)\s*(\d+|[٠-٩]+|[أ-ي]+)', 
            re.MULTILINE
        )

    def light_format(self, text: str) -> str:
        """
        Applies minimal formatting to high-quality Markdown (e.g., from Mistral).
        Only ensures that legal articles have proper Markdown headers.
        """
        lines = text.split('\n')
        formatted_lines = []
        
        for line in lines:
            stripped = line.strip()
            # If it's an article but missing the header tag, add it
            if self.article_pattern.match(stripped) and not stripped.startswith('#'):
                formatted_lines.append(f"### {stripped}")
            else:
                formatted_lines.append(line)
                
        return "\n".join(formatted_lines)

    def full_clean_format(self, text: str) -> str:
        """
        Performs heavy cleaning and regex-based formatting for raw OCR (e.g., EasyOCR).
        Fixes whitespaces and enforces structural headers.
        """
        # Basic noise reduction: remove multiple newlines and leading/trailing spaces
        text = re.sub(r'\n\s*\n', '\n\n', text).strip()
        
        lines = text.split('\n')
        formatted_lines = []
        
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
                
            if self.article_pattern.match(stripped):
                formatted_lines.append(f"### {stripped}")
            else:
                formatted_lines.append(stripped)
                
        return "\n".join(formatted_lines)