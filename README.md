# PDF Reader

Streamlit and LangChain based PDF reader application. It uploads a PDF, extracts page-level text, sections, and tables, creates embeddings with OpenAI or Ollama, and stores structured records in a local Qdrant database.

## Portfolio Role

This repository is the Python PDF/RAG learning and comparison project. The main generated-AI app path is `ROADTONEWSU`; this project exists to show how a similar PDF/RAG workflow is structured in Python with Streamlit, LangChain, Qdrant, and pytest.

## What You Can Learn

- How a small RAG-style PDF app is structured.
- How PDF text, sections, and tables become embeddings and vector database records.
- How Streamlit connects a browser UI to Python processing.
- Why environment variables and error handling matter for AI apps.
- Why structured PDFs should be parsed before vectorization.

## Features

- Upload a PDF file from the Streamlit UI.
- Extract text from PDF pages.
- Extract page-level text, sections, tables, and keywords with pdfplumber.
- Show extraction diagnostics without displaying raw PDF text.
- Warn when extracted text is sparse and optionally run OCR with `ocrmypdf`.
- Report OCR availability, execution result, and post-OCR extraction quality.
- Score section headings and classify tables with simple semantic labels.
- Extract generic document blocks from sections and tables, with type labels and scores.
- Infer a lightweight document type for summaries, falling back to general when confidence is low.
- Store structured page records and metadata in Qdrant with stable record IDs.
- Store document type and extraction quality flags with each vector record.
- Ask questions against the stored PDF text.
- Show source page, section/table type, and heading for retrieval answers.
- Use a structured summary mode with document-type specific output sections.
- Inspect local Qdrant collection status, OCR status, and session upload history from the admin page.

## Tech Stack

- Streamlit
- LangChain
- langchain-openai
- langchain-ollama
- langchain-qdrant
- pdfplumber
- Qdrant local mode
- pytest

## Processing Flow

```text
PDF upload
-> page structure extraction
-> empty-text validation
-> extraction quality check
-> document type inference
-> section and table flattening
-> stable record ID generation
-> metadata enrichment
-> embeddings
-> Qdrant vector storage with metadata
-> retrieval QA
-> answer and source display
-> optional vector DB admin
```

See [docs/process_flow.md](docs/process_flow.md) for a more detailed explanation.

## Setup

### 1. Install dependencies

```bash
uv sync
```

### 2. Configure environment variables

Set environment variables in your shell or process manager. `env.example` is a template only; the app does not automatically read `.env`.

For local verification without an OpenAI API key, use Ollama:

```env
LLM_PROVIDER=ollama
EMBEDDING_PROVIDER=ollama
OPENAI_API_KEY=your_openai_api_key_here
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_LLM_MODEL=llama3.1
OLLAMA_EMBEDDING_MODEL=nomic-embed-text
VECTOR_SIZE=768
QDRANT_PATH=./local_qdrant
COLLECTION_NAME=my_collection
OCR_ENABLED=false
OCR_COMMAND=ocrmypdf
OCR_LANGUAGE=jpn+eng
```

Before starting the app, make sure Ollama is running and the models are available:

```bash
ollama pull llama3.1
ollama pull nomic-embed-text
```

For scanned PDFs, install `ocrmypdf` and its system dependencies, then set:

```env
OCR_ENABLED=true
OCR_COMMAND=ocrmypdf
OCR_LANGUAGE=jpn+eng
```

When OCR is enabled, the app checks whether `OCR_COMMAND` is available. If sparse extraction is detected and the command exists, it creates an OCR-processed PDF, re-extracts it, and shows both pre-OCR and post-OCR quality diagnostics.

For OpenAI verification, switch the providers and set a real API key:

```env
LLM_PROVIDER=openai
EMBEDDING_PROVIDER=openai
OPENAI_API_KEY=your_actual_openai_api_key_here
EMBEDDING_MODEL_NAME=text-embedding-ada-002
QDRANT_PATH=./local_qdrant
COLLECTION_NAME=my_collection
VECTOR_SIZE=1536
```

### 3. Run the app

```bash
uv run streamlit run pdf_uploader.py
```

On Windows, you can also use:

```bash
run.bat
```

## Usage

1. Start the Streamlit app.
2. Select `PDF Upload` from the sidebar.
3. Upload a text-based PDF.
4. Confirm the structured page summary shown in the app.
5. Check extraction diagnostics for table row/column counts, empty cells, and detected headings.
6. Wait for the page structure to be embedded and saved to Qdrant. Re-uploading the same PDF uses stable record IDs so matching records are updated instead of inserted as unrelated duplicates.
   Each stored record includes page, section/table type, document type, and extraction quality flags.
7. Select `Ask My PDF(s)` and enter a question.
8. Choose `Retrieval QA` for fact lookup, or `Structured summary` for document-type aware summaries.

## Tests

```bash
uv run pytest
```

The tests cover settings validation, PDF text helpers, optional OCR command wiring, OCR availability diagnostics, page section inference, heading scoring, table semantics, metadata enrichment, stable record ID generation, vector DB admin helpers, and retrieval source formatting. These tests do not call OpenAI, Ollama, Qdrant, or a real OCR binary.

CI is defined in `.github/workflows/tests.yml` and runs `uv sync --locked` plus `uv run pytest -q`.

## Answer Modes

- `Retrieval QA`: retrieves relevant records from Qdrant and answers the question. Use this for specific lookups.
- `Structured summary`: uses the page-level structure directly and asks the LLM to summarize using sections suited to the inferred document type.
  Resume documents prioritize career, achievement, task, and skill blocks. Reports, manuals, invoices, meeting notes, and general documents use their own summary sections while falling back to the full structured page context.

## Directory Structure

```text
PDF_Read_analyzer/
├── pdf_uploader.py       # Streamlit app
├── pdf_reader_core.py    # Testable settings, text, and metadata helpers
├── pdf_structure.py      # pdfplumber-based page structure extraction
├── tests/                # pytest tests
├── docs/                 # learning and operation notes
├── env.example           # environment variable template
├── pyproject.toml        # project settings and dependencies
└── run.bat               # Windows startup helper
```

## Common Pitfalls

- `OPENAI_API_KEY` is required only when `LLM_PROVIDER` or `EMBEDDING_PROVIDER` is `openai`.
- Ollama verification requires the Ollama service and local models to be available.
- `VECTOR_SIZE` must match the selected embedding model. `nomic-embed-text` uses 768, and `text-embedding-ada-002` uses 1536.
- Scanned image PDFs may not return useful text without OCR.
- OCR runs only when `OCR_ENABLED=true` and `OCR_COMMAND` is available on the machine. If OCR fails, the app keeps the original extraction and shows the OCR error.
- Qdrant local data is created under `QDRANT_PATH`.
- If you change embedding provider or vector size, use a different `COLLECTION_NAME` or Qdrant path to avoid vector-size mismatches.
- Large PDFs may take time and consume OpenAI API usage.

## Future Improvements

- Improve document-type inference with more signals and calibrated thresholds.
- Add richer summary templates for additional business document types.
- Add a sample PDF for local manual testing.
- Add screenshot-based usage documentation.
