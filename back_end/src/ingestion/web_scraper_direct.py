import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import asyncio

from crawl4ai import AsyncWebCrawler, LLMConfig
from crawl4ai.extraction_strategy import LLMExtractionStrategy
from pydantic import BaseModel, Field

# Importing the central configuration path from your config file
# pyrefly: ignore [missing-import]
from config import OUTPUT_JSON_DIR

# Configure logger to use INFO level without debug noise
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


class LawDocumentSchema(BaseModel):
    title: str = Field(..., description="The main document title such as Law No. 119 of 2008.")
    content: str = Field(..., description="The complete and detailed main text, articles, and provisions located directly under the main heading, excluding sidebars and navigation menus.")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Any additional metadata or dates related to the document.")


class DirectWebScraper:
    """
    A modular and scalable web scraper utilizing Crawl4AI and Mistral LLM 
    to extract targeted legal content and output unified JSON schemas.
    Follows SOLID principles with single-responsibility methods.
    """
    
    def __init__(self, output_dir: Path = OUTPUT_JSON_DIR, model_name: str = "mistral-small-latest"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name
        self._validate_api_key()

    def _validate_api_key(self) -> None:
        """Ensures that the Mistral API key environment variable is set."""
        if not os.getenv("MISTRAL_API_KEY"):
            logger.error("MISTRAL_API_KEY environment variable is missing.")
            raise ValueError("[ERROR] MISTRAL_API_KEY environment variable is missing. Please set it to use LLM extraction.")

    def _create_extraction_strategy(self) -> LLMExtractionStrategy:
        """Responsibility: Create and configure the LLM extraction strategy using Mistral."""
        instruction = (
            "Extract only the main body text and articles from the page, "
            "specifically focusing on the content and articles located right under the heading 'قانون رقم 119 لسنة 2008' (Law No. 119 of 2008) "
            "and subsequent regulatory articles. Completely ignore sidebars, table of contents, navigation menus, and advertisements."
        )
        
        llm_config = LLMConfig(
            provider=f"mistral/{self.model_name}",
            api_token=os.getenv("MISTRAL_API_KEY")
        )
        
        return LLMExtractionStrategy(
            llm_config=llm_config,
            schema=LawDocumentSchema.model_json_schema(),
            extraction_type="schema",
            instruction=instruction,
            verbose=False  # Disabled internal verbose debug to keep logs clean
        )

    async def _fetch_page_content(self, url: str) -> Optional[str]:
        """Responsibility: Fetch raw content and perform LLM extraction using Crawl4AI."""
        extraction_strategy = self._create_extraction_strategy()
        
        async with AsyncWebCrawler(verbose=False) as crawler:
            result = await crawler.arun(
                url=url,
                extraction_strategy=extraction_strategy,
                bypass_cache=True
            )
            
            if result.success:
                return result.extracted_content
            else:
                logger.error(f"Failed to scrape URL {url}. Error: {result.error_message}")
                return None

    def store_final_output(self, data: Dict[str, Any], filename: str) -> Path:
        """Responsibility: Store the final processed output strictly into the central JSON folder from config."""
        output_path = self.output_dir / f"{filename}.json"
        
        unified_payload = {
            "source_type": "direct_web_scraping_llm",
            "data": data
        }
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(unified_payload, f, ensure_ascii=False, indent=4)
            
        logger.info(f"Final JSON output successfully stored at: {output_path}")
        return output_path

    async def process_url(self, url: str, output_filename: str) -> Optional[Path]:
        """Main Orchestrator method: Manages the full scraping, extraction, and storing pipeline."""
        logger.info(f"Starting scraping and LLM extraction for: {url}")
        raw_extracted_json_string = await self._fetch_page_content(url)
        
        if not raw_extracted_json_string:
            return None
            
        try:
            parsed_data = json.loads(raw_extracted_json_string)
            if isinstance(parsed_data, list) and len(parsed_data) > 0:
                parsed_data = parsed_data[0]
                
            saved_file_path = self.store_final_output(parsed_data, output_filename)
            return saved_file_path
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM extracted output as JSON: {e}")
            return None