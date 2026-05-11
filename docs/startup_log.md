# Startup Log

Use this file to record local startup attempts.

| Item | Result |
| --- | --- |
| Date | 2026-05-09 |
| Python version | Python 3.13 runtime used for compile/test in this workspace |
| Dependency command | `uv sync` succeeded and created `.venv` with Python 3.10.6 |
| Environment file | `.env` from `env.example`; local verification can use `LLM_PROVIDER=ollama` and `EMBEDDING_PROVIDER=ollama` |
| Startup command | `uv run streamlit run pdf_uploader.py` |
| Result | `uv run python -m py_compile pdf_uploader.py pdf_reader_core.py`, `uv run pytest -q`, `uv run python -c "import pdf_uploader"`, and headless Streamlit HTTP startup passed. Streamlit returned `HTTP 200` on `http://localhost:8501`. |
| Error message | No compile, pytest, import, or HTTP startup error. `.env` contents were not opened or displayed. |
| Fix attempted | Added testable core helpers, README rewrite, env template rewrite, process-flow docs, pytest coverage, and Ollama provider settings. |
| Next action | Verify a real PDF upload and question flow with the configured local Ollama models. Current `ollama list` output did not show the README default models `llama3.1` or `nomic-embed-text`, so use installed model names in `.env` or pull the defaults before end-to-end testing. |

## Local PDF Smoke Test

| Item | Result |
| --- | --- |
| Date | 2026-05-09 |
| PDF | Local resume PDF from Downloads; contents were not printed or recorded. |
| `.env` handling | `.env` contents were not opened or displayed. |
| PDF extraction | Passed. 2 pages, 1682 extracted characters. |
| Chunking | Passed. 5 chunks. |
| Embedding model | `nomic-embed-text` was missing at first, then installed with `ollama pull nomic-embed-text`. |
| Vector store write | Passed after installing `nomic-embed-text`. |
| QA test | Passed with configured local models after `llama3.1` became available. Answer content was not printed; only answer length was checked. |
| Issue found | Initial run failed because `nomic-embed-text` and `llama3.1` were missing. Both are now installed locally. |
| Workaround | Earlier temporary workaround used installed `gemma3:4B`; no longer required after `llama3.1` install completed. |
| Warning | Resolved. The app now uses `langchain-ollama` and `langchain-qdrant`, calls `invoke()`, and explicitly closes local Qdrant clients after vector writes and QA calls. |

## Improvement Log

| Date | Change | Verification |
| --- | --- | --- |
| 2026-05-10 | Migrated from deprecated `langchain_community` Ollama/Qdrant classes to `langchain-ollama` and `langchain-qdrant`. | `uv run pytest -q` passed with 6 tests. |
| 2026-05-10 | Added explicit Qdrant client close after vector writes and QA calls. | Local PDF smoke test passed without deprecation or Qdrant cleanup warnings. |
| 2026-05-10 | Added pdfplumber-based page structure extraction, table normalization, section inference, and metadata-backed vector records. | Local PDF smoke test produced 2 structured pages, 7 sections, 3 tables, 12 vector records, and a successful QA response. |
| 2026-05-10 | Added structured resume summary mode using the full page structure instead of top-k retrieval. | Local PDF smoke test passed with 2 pages and a generated summary response. |
| 2026-05-10 | Added extraction diagnostics and table pruning for empty rows/columns. | Diagnostics reported 2 pages, 7 sections, 3 tables, and reduced empty table cells from 71 to 23 after pruning. |
| 2026-05-10 | Replaced resume-specific career extraction with generic document blocks using block type, score, and signals. The resume prompt now prioritizes career-like document blocks without making extraction resume-only. | Diagnostics found 10 document blocks and structured summary smoke test passed. |
| 2026-05-10 | Added stable Qdrant record IDs and retrieval source display. Removed automatic `.env` loading so settings must come from the process environment. | `uv run pytest -q` passed with 17 tests. Local PDF smoke test produced 2 pages, 12 records, 12 unique stable IDs, and Qdrant still had 12 records after two writes. |
| 2026-05-10 | Added lightweight document type inference and document-type aware structured summary prompts for resume, report, manual, invoice, meeting notes, and general documents. | `uv run pytest -q` passed with 20 tests. Local PDF smoke test inferred `resume` from 2 pages without printing PDF contents. |
| 2026-05-10 | Added extraction quality reporting, OCR-needed warnings for sparse text, and a minimum document-type signal threshold that falls back to `general` on weak matches. | `uv run pytest -q` passed with 23 tests. Local PDF smoke test reported OCR not needed, 1551 extracted characters, and `resume` with score 2. |
| 2026-05-10 | Added document type and OCR quality flags to vector metadata and retrieval source display. Removed the deprecated PyPDF2 dependency and aligned old text extraction with pdfplumber. | `uv run pytest -q` passed with 25 tests. Local PDF smoke test produced 12 records with `document_type=resume` and `needs_ocr=False` metadata. |
| 2026-05-10 | Added optional OCR execution through the `ocrmypdf` CLI when sparse extraction is detected and `OCR_ENABLED=true`. OCR command wiring is unit-tested without invoking a real OCR binary. | `uv run pytest -q` passed with 30 tests. |
| 2026-05-10 | Added heading scoring, table semantic labels, and expanded structured summary templates for contract, specification, and plan documents. | `uv run pytest -q` passed with 34 tests. |
| 2026-05-10 | Added a Vector DB Admin page for local collection status, guarded collection deletion, and session upload history. | `uv run pytest -q` passed with 36 tests. |
| 2026-05-10 | Added GitHub Actions CI for locked dependency install and pytest. | Local verification passed with `uv run pytest -q` and import smoke test. |
| 2026-05-10 | Completed OCR execution handling with command availability checks, OCR result diagnostics, post-OCR quality evaluation, and fallback behavior when OCR is unavailable or fails. | `uv run pytest -q` passed with 40 tests. Local machine did not expose `ocrmypdf` or `tesseract` on PATH, so real OCR binary execution was not run here. |
| 2026-05-10 | Added OCR status diagnostics to the admin page so OCR readiness can be checked before uploading scanned PDFs. | `uv run pytest -q` passed with 41 tests. |
