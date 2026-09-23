import sys
import asyncio
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

# Ensure project root directory is in sys.path when running via python directly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# pyrefly: ignore [missing-import]
from src.ingestion.ingestion_pipeline import (
    run_direct_text_ingestion,
    run_pdf_download,
    run_ocr_processing,
    run_full_pipeline,
)
# pyrefly: ignore [missing-import]
from config import DIRECT_TEXT_URL, PAGINATION_PDF_URL


# ── Stage 1 Tests ─────────────────────────────────────────────────────────────


@pytest.mark.anyio
async def test_stage1_direct_text_ingestion_success(tmp_path, monkeypatch):
    """Stage 1: DirectWebScraper saves a JSON file and returns its path."""
    monkeypatch.setenv("MISTRAL_API_KEY", "fake_key")
    expected_json = tmp_path / "law_119_2008_wikisource.json"

    with patch("src.ingestion.ingestion_pipeline.DirectWebScraper") as MockScraper:
        instance = MockScraper.return_value
        instance.process_url = AsyncMock(return_value=expected_json)

        result = await run_direct_text_ingestion(
            url=DIRECT_TEXT_URL,
            output_dir=tmp_path,
        )

    assert result == expected_json
    instance.process_url.assert_awaited_once_with(DIRECT_TEXT_URL, "law_119_2008_wikisource")


@pytest.mark.anyio
async def test_stage1_direct_text_ingestion_scraper_returns_none(tmp_path, monkeypatch):
    """Stage 1: Returns None gracefully when scraper finds nothing."""
    monkeypatch.setenv("MISTRAL_API_KEY", "fake_key")

    with patch("src.ingestion.ingestion_pipeline.DirectWebScraper") as MockScraper:
        instance = MockScraper.return_value
        instance.process_url = AsyncMock(return_value=None)

        result = await run_direct_text_ingestion(url=DIRECT_TEXT_URL, output_dir=tmp_path)

    assert result is None


@pytest.mark.anyio
async def test_stage1_direct_text_ingestion_exception(tmp_path, monkeypatch):
    """Stage 1: Handles unexpected exceptions without crashing the pipeline."""
    monkeypatch.setenv("MISTRAL_API_KEY", "fake_key")

    with patch(
        "src.ingestion.ingestion_pipeline.DirectWebScraper",
        side_effect=ValueError("API key missing"),
    ):
        result = await run_direct_text_ingestion(url=DIRECT_TEXT_URL, output_dir=tmp_path)

    assert result is None


# ── Stage 2 Tests ─────────────────────────────────────────────────────────────


@pytest.mark.anyio
async def test_stage2_pdf_download_returns_paths(tmp_path, monkeypatch):
    """Stage 2: PDFDownloaderScraper is called and its returned paths are forwarded."""
    monkeypatch.setenv("MISTRAL_API_KEY", "fake_key")
    fake_pdfs = [tmp_path / "doc1.pdf", tmp_path / "doc2.pdf"]

    with patch("src.ingestion.ingestion_pipeline.PDFDownloaderScraper") as MockScraper:
        instance = MockScraper.return_value
        instance.scrape_and_download_workflow = AsyncMock(return_value=fake_pdfs)

        result = await run_pdf_download(
            url=PAGINATION_PDF_URL,
            pdf_output_dir=tmp_path,
            max_pages=2,
        )

    assert result == fake_pdfs
    instance.scrape_and_download_workflow.assert_awaited_once_with(PAGINATION_PDF_URL, max_pages=2)


@pytest.mark.anyio
async def test_stage2_pdf_download_empty_result(tmp_path, monkeypatch):
    """Stage 2: Returns empty list when no PDFs are found."""
    monkeypatch.setenv("MISTRAL_API_KEY", "fake_key")

    with patch("src.ingestion.ingestion_pipeline.PDFDownloaderScraper") as MockScraper:
        instance = MockScraper.return_value
        instance.scrape_and_download_workflow = AsyncMock(return_value=[])

        result = await run_pdf_download(url=PAGINATION_PDF_URL, pdf_output_dir=tmp_path)

    assert result == []


@pytest.mark.anyio
async def test_stage2_pdf_download_exception(tmp_path, monkeypatch):
    """Stage 2: Handles unexpected exceptions without crashing the pipeline."""
    monkeypatch.setenv("MISTRAL_API_KEY", "fake_key")

    with patch(
        "src.ingestion.ingestion_pipeline.PDFDownloaderScraper",
        side_effect=RuntimeError("Network error"),
    ):
        result = await run_pdf_download(url=PAGINATION_PDF_URL, pdf_output_dir=tmp_path)

    assert result == []


# ── Stage 3 Tests ─────────────────────────────────────────────────────────────


def test_stage3_ocr_processing_no_pdfs(tmp_path, monkeypatch):
    """Stage 3: Returns empty list gracefully when PDF directory contains no PDFs."""
    monkeypatch.setenv("MISTRAL_API_KEY", "fake_key")
    result = run_ocr_processing(pdf_dir=tmp_path, json_output_dir=tmp_path)
    assert result == []


def test_stage3_ocr_processing_with_pdfs(tmp_path, monkeypatch):
    """Stage 3: Iterates PDFs and returns produced JSON paths from OCR processor."""
    monkeypatch.setenv("MISTRAL_API_KEY", "fake_key")

    pdf1 = tmp_path / "doc1.pdf"
    pdf2 = tmp_path / "doc2.pdf"
    pdf1.write_bytes(b"%PDF-1.4 dummy")
    pdf2.write_bytes(b"%PDF-1.4 dummy")

    json1 = tmp_path / "doc1.json"
    json2 = tmp_path / "doc2.json"

    with patch("src.ingestion.ingestion_pipeline.MistralPDFProcessor") as MockOCR:
        instance = MockOCR.return_value
        instance.process_pdf = MagicMock(return_value={"pages": []})
        instance._get_expected_json_path = MagicMock(side_effect=[json1, json2])

        result = run_ocr_processing(pdf_dir=tmp_path, json_output_dir=tmp_path)

    assert len(result) == 2
    assert json1 in result
    assert json2 in result


def test_stage3_ocr_processing_individual_failure(tmp_path, monkeypatch):
    """Stage 3: Skips a failing PDF and continues processing the rest."""
    monkeypatch.setenv("MISTRAL_API_KEY", "fake_key")

    pdf1 = tmp_path / "good.pdf"
    pdf2 = tmp_path / "bad.pdf"
    pdf1.write_bytes(b"%PDF-1.4 good")
    pdf2.write_bytes(b"%PDF-1.4 bad")

    json_good = tmp_path / "good.json"

    def mock_get_json(pdf_path):
        return tmp_path / f"{Path(pdf_path).stem}.json"

    call_count = {"n": 0}

    def mock_process(pdf_path):
        call_count["n"] += 1
        if "bad" in str(pdf_path):
            raise RuntimeError("OCR API error")
        return {"pages": []}

    with patch("src.ingestion.ingestion_pipeline.MistralPDFProcessor") as MockOCR:
        instance = MockOCR.return_value
        instance.process_pdf = MagicMock(side_effect=mock_process)
        instance._get_expected_json_path = MagicMock(side_effect=mock_get_json)

        result = run_ocr_processing(pdf_dir=tmp_path, json_output_dir=tmp_path)

    # Only the good PDF should appear in results
    assert len(result) == 1
    assert tmp_path / "good.json" in result


# ── Full Pipeline Integration Tests ───────────────────────────────────────────


@pytest.mark.anyio
async def test_full_pipeline_integration(tmp_path, monkeypatch):
    """Full pipeline: all three stages are invoked and summary dict is correct."""
    monkeypatch.setenv("MISTRAL_API_KEY", "fake_key")

    json_dir = tmp_path / "json"
    pdf_dir = tmp_path / "pdfs"
    json_dir.mkdir()
    pdf_dir.mkdir()

    fake_direct_json = json_dir / "law_119_2008_wikisource.json"
    fake_pdfs = [pdf_dir / "doc1.pdf", pdf_dir / "doc2.pdf"]
    for p in fake_pdfs:
        p.write_bytes(b"%PDF-1.4 dummy")
    fake_ocr_jsons = [json_dir / "doc1.json", json_dir / "doc2.json"]

    with (
        patch("src.ingestion.ingestion_pipeline.DirectWebScraper") as MockDirect,
        patch("src.ingestion.ingestion_pipeline.PDFDownloaderScraper") as MockPDF,
        patch("src.ingestion.ingestion_pipeline.MistralPDFProcessor") as MockOCR,
        patch("src.ingestion.ingestion_pipeline.LocalPDFLoader") as MockLoader,
    ):
        MockDirect.return_value.process_url = AsyncMock(return_value=fake_direct_json)
        MockPDF.return_value.scrape_and_download_workflow = AsyncMock(return_value=fake_pdfs)

        def loader_batch():
            return [
                {"file_path": str(p), "file_name": p.name, "status": "pending_ocr"}
                for p in fake_pdfs
            ]

        MockLoader.return_value.load_folder_batch = MagicMock(side_effect=loader_batch)
        MockOCR.return_value.process_pdf = MagicMock(return_value={"pages": []})
        MockOCR.return_value._get_expected_json_path = MagicMock(
            side_effect=lambda p: json_dir / f"{Path(p).stem}.json"
        )

        summary = await run_full_pipeline(
            pdf_max_pages=2,
            pdf_output_dir=pdf_dir,
            json_output_dir=json_dir,
        )

    assert summary["direct_text_json"] == fake_direct_json
    assert summary["downloaded_pdfs"] == fake_pdfs
    assert len(summary["ocr_jsons"]) == 2


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
