from pathlib import Path
# pyrefly: ignore [missing-import]
from typing import List, Dict , Any 

import logging 


logging.basicConfig(
    level=logging.INFO,  
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("Zar3a.Ingestion.LocalPDFLoader")


class LocalPDFLoader:

    """
    A class responsible for discovering, validating, and loading local PDF files 
    from a specified directory, preparing them as a batch queue for downstream OCR processing.  
    """    

    def __init__(self, data_dir: str = "Data") -> None:

        self.data_dir: Path = Path(data_dir)
        self._validate_directory()

    def _validate_directory(self)-> None :

        """Ensures that the target data directory exists; creates it if missing."""

        if not self.data_dir.exists() :
            logger.warning(f"Directory '{self.data_dir}' not found. Creating it...")
            self.data_dir.mkdir(parents=True, exist_ok=True)

    def get_pdf_paths(self) -> List[Path]:

        """
        Recursively scans the directory for PDF files while filtering out 
        any potential duplicate filenames to ensure data integrity.
        """                

        seen_filenames: set =set()
        unique_pdf_files: List[Path] = []

        for pdf_path in self.data_dir.rglob("*.pdf"):
            if pdf_path.name not in seen_filenames:
                seen_filenames.add(pdf_path.name) 
                unique_pdf_files.append(pdf_path)
        return unique_pdf_files 

    def load_folder_batch(self)-> List[Dict[str, Any]]:
        """
        Loops through the discovered unique PDFs and builds a structured batch metadata queue 
        ready to be consumed by the OCR processor.
        """

        pdf_files: List[Path] = self.get_pdf_paths()
        batch_metadata: List[Dict[str,Any]] = []
        
        if not pdf_files:
            logger.info(f"No PDF files found in '{self.data_dir}' directory.")
            return batch_metadata
   
        logger.info(f"Found {len(pdf_files)} unique PDF file(s) ready for ingestion.")    
        
        for pdf_path in pdf_files:
            file_info = {
                'file_path': str(pdf_path.resolve()),
                'file_name': pdf_path.name,
                'file_size_bytes': pdf_path.stat().st_size,
                'file_stem': pdf_path.stem,
                'last_modified_timestamp': pdf_path.stat().st_mtime,
                'source_directory': str(self.data_dir),
                'status': 'pending_ocr', 
                'pages_count': None,    
                'ocr_result': None      
            }
            batch_metadata.append(file_info)

            logger.info(f"Queued file: {pdf_path.name} (from folder: {pdf_path.parent.name})")
        
        return batch_metadata
        