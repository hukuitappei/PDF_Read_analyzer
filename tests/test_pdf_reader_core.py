import pytest

from pdf_reader_core import (
    DEFAULT_COLLECTION_NAME,
    DEFAULT_EMBEDDING_MODEL_NAME,
    DEFAULT_QDRANT_PATH,
    AppSettings,
    StructuredPage,
    assess_extraction_quality,
    check_ocr_availability,
    ensure_text_was_extracted,
    load_settings,
    merge_page_texts,
    ocr_pdf_bytes,
)


def test_load_settings_requires_openai_api_key():
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        load_settings({})


def test_load_settings_uses_defaults():
    settings = load_settings({"OPENAI_API_KEY": "test-key"})

    assert settings.openai_api_key == "test-key"
    assert settings.llm_provider == "openai"
    assert settings.embedding_provider == "openai"
    assert settings.embedding_model_name == DEFAULT_EMBEDDING_MODEL_NAME
    assert settings.qdrant_path == DEFAULT_QDRANT_PATH
    assert settings.collection_name == DEFAULT_COLLECTION_NAME
    assert settings.vector_size == 1536


def test_load_settings_allows_ollama_without_openai_api_key():
    settings = load_settings(
        {
            "LLM_PROVIDER": "ollama",
            "EMBEDDING_PROVIDER": "ollama",
        }
    )

    assert settings.openai_api_key is None
    assert settings.llm_provider == "ollama"
    assert settings.embedding_provider == "ollama"
    assert settings.ollama_base_url == "http://localhost:11434"
    assert settings.ollama_llm_model == "llama3.1"
    assert settings.ollama_embedding_model == "nomic-embed-text"
    assert settings.vector_size == 768


def test_load_settings_reads_ocr_options():
    settings = load_settings(
        {
            "OPENAI_API_KEY": "test-key",
            "OCR_ENABLED": "true",
            "OCR_COMMAND": "custom-ocr",
            "OCR_LANGUAGE": "jpn",
        }
    )

    assert settings.ocr_enabled is True
    assert settings.ocr_command == "custom-ocr"
    assert settings.ocr_language == "jpn"


def test_load_settings_rejects_unknown_provider():
    with pytest.raises(ValueError, match="LLM_PROVIDER"):
        load_settings({"LLM_PROVIDER": "local"})


def test_merge_page_texts_skips_empty_pages():
    text = merge_page_texts([" first page ", None, "", "second page"])

    assert text == "first page\n\nsecond page"


def test_ensure_text_was_extracted_rejects_empty_text():
    with pytest.raises(ValueError, match="No text"):
        ensure_text_was_extracted("   ")


def test_assess_extraction_quality_flags_sparse_text():
    pages = [StructuredPage(page_number=1, raw_text="A"), StructuredPage(page_number=2, raw_text="")]

    report = assess_extraction_quality(pages)

    assert report.needs_ocr is True
    assert report.low_text_pages == [1, 2]
    assert report.warnings


def test_assess_extraction_quality_allows_regular_text():
    pages = [StructuredPage(page_number=1, raw_text="This page has enough extracted text for downstream processing.")]

    report = assess_extraction_quality(pages)

    assert report.needs_ocr is False
    assert report.low_text_pages == []


def test_check_ocr_availability_reports_disabled():
    settings = AppSettings(openai_api_key="test-key", ocr_enabled=False)

    availability = check_ocr_availability(settings)

    assert availability.enabled is False
    assert availability.available is False
    assert "disabled" in availability.message


def test_check_ocr_availability_reports_available_command():
    settings = AppSettings(openai_api_key="test-key", ocr_enabled=True, ocr_command="ocrmypdf")

    availability = check_ocr_availability(settings, command_resolver=lambda command: f"/usr/bin/{command}")

    assert availability.enabled is True
    assert availability.available is True
    assert availability.resolved_path == "/usr/bin/ocrmypdf"


def test_check_ocr_availability_reports_missing_command():
    settings = AppSettings(openai_api_key="test-key", ocr_enabled=True, ocr_command="ocrmypdf")

    availability = check_ocr_availability(settings, command_resolver=lambda _command: None)

    assert availability.enabled is True
    assert availability.available is False
    assert "not found" in availability.message


def test_ocr_pdf_bytes_runs_configured_command():
    settings = AppSettings(openai_api_key="test-key", ocr_enabled=True, ocr_command="ocrmypdf", ocr_language="jpn+eng")
    captured = {}

    def fake_runner(command, check, capture_output, text):
        captured["command"] = command
        output_path = command[-1]
        with open(output_path, "wb") as output_file:
            output_file.write(b"ocr-pdf")

    output = ocr_pdf_bytes(b"input-pdf", settings, runner=fake_runner)

    assert output == b"ocr-pdf"
    assert captured["command"][:4] == ["ocrmypdf", "--force-ocr", "-l", "jpn+eng"]


def test_ocr_pdf_bytes_rejects_disabled_ocr():
    settings = AppSettings(openai_api_key="test-key", ocr_enabled=False)

    with pytest.raises(ValueError, match="OCR is disabled"):
        ocr_pdf_bytes(b"input-pdf", settings, runner=lambda *args, **kwargs: None)
