from types import SimpleNamespace

import pdf_uploader
from pdf_reader_core import AppSettings, StructuredPage
from pdf_uploader import (
    delete_qdrant_collection,
    extract_structured_pdf_with_optional_ocr,
    format_sources,
    get_ocr_status,
    get_qdrant_collection_status,
)


def test_format_sources_returns_unique_source_metadata():
    answer = {
        "source_documents": [
            SimpleNamespace(
                metadata={
                    "document_id": "doc-1",
                    "page": 1,
                    "section_type": "section",
                    "heading": "Summary",
                    "keywords": "Python",
                    "document_type": "resume",
                    "needs_ocr": False,
                }
            ),
            SimpleNamespace(
                metadata={
                    "document_id": "doc-1",
                    "page": 1,
                    "section_type": "section",
                    "heading": "Summary",
                    "keywords": "Python",
                    "document_type": "resume",
                    "needs_ocr": False,
                }
            ),
            SimpleNamespace(
                metadata={
                    "document_id": "doc-1",
                    "page": 2,
                    "section_type": "table",
                    "heading": "Table 1",
                    "keywords": "Excel",
                    "document_type": "resume",
                    "needs_ocr": False,
                }
            ),
        ]
    }

    sources = format_sources(answer)

    assert sources == [
        {
            "page": 1,
            "section_type": "section",
            "heading": "Summary",
            "keywords": "Python",
            "document_type": "resume",
            "needs_ocr": False,
            "document_id": "doc-1",
        },
        {
            "page": 2,
            "section_type": "table",
            "heading": "Table 1",
            "keywords": "Excel",
            "document_type": "resume",
            "needs_ocr": False,
            "document_id": "doc-1",
        },
    ]


def test_format_sources_ignores_non_dict_answers():
    assert format_sources("plain answer") == []


def test_extract_structured_pdf_with_optional_ocr_reruns_when_sparse(monkeypatch):
    sparse_pages = [StructuredPage(page_number=1, raw_text="")]
    ocr_pages = [
        StructuredPage(
            page_number=1,
            raw_text="This OCR result has enough extracted text to continue processing.",
        )
    ]
    calls = {"extract": 0, "ocr": 0}

    def fake_extract(_pdf_source):
        calls["extract"] += 1
        return sparse_pages if calls["extract"] == 1 else ocr_pages

    def fake_ocr(pdf_bytes, settings):
        calls["ocr"] += 1
        assert pdf_bytes == b"pdf-bytes"
        assert settings.ocr_enabled is True
        return b"ocr-pdf-bytes"

    monkeypatch.setattr(pdf_uploader, "extract_structured_pdf", fake_extract)
    monkeypatch.setattr(pdf_uploader, "ocr_pdf_bytes", fake_ocr)
    monkeypatch.setattr(
        pdf_uploader,
        "check_ocr_availability",
        lambda settings: SimpleNamespace(to_dict=lambda: {"available": True}, available=True),
    )
    monkeypatch.setattr(pdf_uploader, "load_settings", lambda: AppSettings(openai_api_key=None, ocr_enabled=True))
    monkeypatch.setattr(pdf_uploader.st, "info", lambda _message: None)
    monkeypatch.setattr(pdf_uploader.st, "session_state", {})

    pages = extract_structured_pdf_with_optional_ocr(SimpleNamespace(getvalue=lambda: b"pdf-bytes"))

    assert pages == ocr_pages
    assert calls == {"extract": 2, "ocr": 1}
    assert pdf_uploader.st.session_state["last_ocr_result"]["applied"] is True
    assert pdf_uploader.st.session_state["last_ocr_result"]["output_quality"]["needs_ocr"] is False


def test_extract_structured_pdf_with_optional_ocr_skips_ocr_when_quality_is_good(monkeypatch):
    pages = [
        StructuredPage(
            page_number=1,
            raw_text="This PDF already has enough extracted text to continue processing.",
        )
    ]
    calls = {"extract": 0, "ocr": 0}

    def fake_extract(_pdf_source):
        calls["extract"] += 1
        return pages

    def fake_ocr(_pdf_bytes, _settings):
        calls["ocr"] += 1
        return b"ocr-pdf-bytes"

    monkeypatch.setattr(pdf_uploader, "extract_structured_pdf", fake_extract)
    monkeypatch.setattr(pdf_uploader, "ocr_pdf_bytes", fake_ocr)
    monkeypatch.setattr(pdf_uploader, "load_settings", lambda: AppSettings(openai_api_key=None, ocr_enabled=True))
    monkeypatch.setattr(pdf_uploader.st, "session_state", {})

    result = extract_structured_pdf_with_optional_ocr(SimpleNamespace(getvalue=lambda: b"pdf-bytes"))

    assert result == pages
    assert calls == {"extract": 1, "ocr": 0}
    assert pdf_uploader.st.session_state["last_ocr_result"]["applied"] is False


def test_extract_structured_pdf_with_optional_ocr_falls_back_when_command_missing(monkeypatch):
    sparse_pages = [StructuredPage(page_number=1, raw_text="A")]
    calls = {"extract": 0, "ocr": 0, "warning": 0}

    def fake_extract(_pdf_source):
        calls["extract"] += 1
        return sparse_pages

    def fake_ocr(_pdf_bytes, _settings):
        calls["ocr"] += 1
        return b"ocr-pdf-bytes"

    monkeypatch.setattr(pdf_uploader, "extract_structured_pdf", fake_extract)
    monkeypatch.setattr(pdf_uploader, "ocr_pdf_bytes", fake_ocr)
    monkeypatch.setattr(pdf_uploader, "load_settings", lambda: AppSettings(openai_api_key=None, ocr_enabled=True))
    monkeypatch.setattr(
        pdf_uploader,
        "check_ocr_availability",
        lambda settings: SimpleNamespace(
            to_dict=lambda: {"available": False, "message": "missing"},
            available=False,
            message="missing",
        ),
    )
    monkeypatch.setattr(pdf_uploader.st, "warning", lambda _message: calls.update(warning=calls["warning"] + 1))
    monkeypatch.setattr(pdf_uploader.st, "session_state", {})

    pages = extract_structured_pdf_with_optional_ocr(SimpleNamespace(getvalue=lambda: b"pdf-bytes"))

    assert pages == sparse_pages
    assert calls == {"extract": 1, "ocr": 0, "warning": 1}
    assert pdf_uploader.st.session_state["last_ocr_result"]["error"] == "missing"


def test_get_qdrant_collection_status_reports_count(monkeypatch):
    class FakeClient:
        def __init__(self, path):
            self.path = path

        def get_collections(self):
            return SimpleNamespace(collections=[SimpleNamespace(name="docs")])

        def count(self, collection_name, exact):
            assert collection_name == "docs"
            assert exact is True
            return SimpleNamespace(count=12)

        def close(self):
            pass

    monkeypatch.setattr(pdf_uploader, "QdrantClient", FakeClient)
    monkeypatch.setattr(
        pdf_uploader,
        "load_settings",
        lambda: AppSettings(openai_api_key=None, qdrant_path="./qdrant", collection_name="docs"),
    )

    status = get_qdrant_collection_status()

    assert status == {
        "qdrant_path": "./qdrant",
        "collection_name": "docs",
        "exists": True,
        "record_count": 12,
    }


def test_get_ocr_status_reports_settings_and_availability(monkeypatch):
    monkeypatch.setattr(
        pdf_uploader,
        "load_settings",
        lambda: AppSettings(openai_api_key=None, ocr_enabled=True, ocr_command="ocrmypdf", ocr_language="jpn+eng"),
    )
    monkeypatch.setattr(
        pdf_uploader,
        "check_ocr_availability",
        lambda settings: SimpleNamespace(
            to_dict=lambda: {
                "enabled": True,
                "command": settings.ocr_command,
                "available": False,
                "message": "missing",
            }
        ),
    )

    status = get_ocr_status()

    assert status == {
        "ocr_enabled": True,
        "ocr_command": "ocrmypdf",
        "ocr_language": "jpn+eng",
        "availability": {
            "enabled": True,
            "command": "ocrmypdf",
            "available": False,
            "message": "missing",
        },
    }


def test_delete_qdrant_collection_deletes_existing_collection(monkeypatch):
    calls = {"deleted": 0}

    class FakeClient:
        def __init__(self, path):
            self.path = path

        def get_collections(self):
            return SimpleNamespace(collections=[SimpleNamespace(name="docs")])

        def delete_collection(self, collection_name):
            assert collection_name == "docs"
            calls["deleted"] += 1

        def close(self):
            pass

    monkeypatch.setattr(pdf_uploader, "QdrantClient", FakeClient)
    monkeypatch.setattr(
        pdf_uploader,
        "load_settings",
        lambda: AppSettings(openai_api_key=None, qdrant_path="./qdrant", collection_name="docs"),
    )

    assert delete_qdrant_collection() is True
    assert calls["deleted"] == 1
