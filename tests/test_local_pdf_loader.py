import sys
import pytest
from pathlib import Path

# Ensure project root directory is in sys.path when running via python directly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# pyrefly: ignore [missing-import]
from src.ingestion.local_pdf_loader import LocalPDFLoader

def test_loader_creates_directory(tmp_path: Path) -> None:
    """Test if the loader automatically creates the target directory if it doesn't exist."""
    target_dir = tmp_path / "non_existent_data"
    assert not target_dir.exists()
    
    # Initialize loader with the non-existent path
    loader = LocalPDFLoader(data_dir=str(target_dir))
    
    # Assert that the directory was successfully created
    assert target_dir.exists()

def test_load_folder_batch_empty(tmp_path: Path) -> None:
    """Test if the loader returns an empty list when no PDFs are present."""
    loader = LocalPDFLoader(data_dir=str(tmp_path))
    batch = loader.load_folder_batch()
    
    assert isinstance(batch, list)
    assert len(batch) == 0

def test_load_folder_batch_with_pdfs(tmp_path: Path) -> None:
    """Test if the loader correctly discovers PDFs, filters them, and builds proper metadata."""
    # Create dummy PDF files for testing
    pdf1 = tmp_path / "document_one.pdf"
    pdf2 = tmp_path / "document_two.pdf"
    
    pdf1.write_text("Dummy content for doc 1")
    pdf2.write_text("Dummy content for doc 2")
    
    # Create a dummy non-pdf file to ensure it gets ignored
    txt_file = tmp_path / "ignore_me.txt"
    txt_file.write_text("Should not be loaded")

    loader = LocalPDFLoader(data_dir=str(tmp_path))
    batch = loader.load_folder_batch()

    # Assertions
    assert len(batch) == 2  # Should only pick the 2 PDFs
    
    filenames = {item["file_name"] for item in batch}
    assert filenames == {"document_one.pdf", "document_two.pdf"}
    
    # Check structure of the metadata dictionary
    first_item = batch[0]
    assert "file_path" in first_item
    assert "file_size_bytes" in first_item
    assert "file_stem" in first_item
    assert "last_modified_timestamp" in first_item
    assert first_item["status"] == "pending_ocr"


