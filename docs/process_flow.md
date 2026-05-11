# PDF Reader Process Flow

## Overview

The application turns a PDF into searchable vector data, then uses retrieval QA to answer questions.

```text
PDF upload -> page structure extraction -> extraction quality check -> optional OCR -> document type inference -> enriched metadata records -> stable IDs -> embeddings -> Qdrant -> retrieval QA
```

For summary-style questions, the app can skip retrieval and send the structured page context directly to a document-type aware summary prompt.
The app also extracts generic document blocks from sections and tables. Resume-style prompts prioritize blocks tagged as career, achievement, task, or skill, while other document types can add their own prompt priorities later.

## Steps

1. Streamlit receives the uploaded PDF file.
2. pdfplumber reads each page and extracts text and tables.
3. The app infers simple sections, keywords, and page metadata.
4. The app checks extraction quality and warns when OCR may be required.
5. If `OCR_ENABLED=true` and extracted text is sparse, the app checks OCR command availability, runs `ocrmypdf`, re-extracts the OCR output PDF, and records pre/post OCR quality.
6. The app infers a lightweight document type from headings and body text. Weak matches fall back to general.
7. Empty extraction results are rejected with a clear error.
8. Section headings are scored and tables receive simple semantic labels such as invoice, schedule, career, or comparison.
9. Page text, inferred sections, tables, document type, and extraction quality flags are flattened into vector records.
10. OpenAI Embeddings or Ollama local embeddings convert records into vectors.
11. Qdrant stores the vectors and metadata in a local collection. Stable record IDs make repeated uploads of the same PDF update matching records instead of growing unrelated duplicates.
12. A retriever fetches relevant page, section, or table records for a user question.
13. The selected LLM generates an answer from the retrieved context.
14. The answer screen shows the answer and source metadata such as page, section/table type, and heading.
15. The admin page shows local Qdrant collection status, session upload history, and a guarded collection delete action.

## Answer Modes

- Retrieval QA: best for specific questions about a small part of the document.
- Structured summary: best for summaries where page headings, sections, and tables should stay together.
  The prompt switches summary sections based on inferred type: resume, report, manual, invoice, meeting notes, or general.

## Extraction Diagnostics

The upload screen shows diagnostics without printing raw PDF text:

- Page count, extracted character count, and non-empty line count.
- Extraction quality, OCR-needed flag, and low-text page numbers.
- Inferred document type, confidence score, and matched signals.
- Section count and detected section headings.
- Table count, row count, column count, empty cell count, short cell count, and ragged row count.
- Keyword hints inferred from the page text.

The app tries multiple pdfplumber table extraction strategies and chooses the result with better table quality. Empty rows and empty columns are pruned before diagnostics and vector storage.

## Teaching Points

- A RAG app is easier to understand when the data flow is visible.
- PDF extraction and vector storage are separate concerns.
- Tables and section headings should be preserved before vectorization when the PDF is structured.
- Summaries often work better from the full structured page context than from top-k retrieval alone.
- Document block extraction gives document-specific prompts a higher-signal intermediate representation than raw page text alone.
- Stable vector IDs are important because local testing often re-uploads the same document many times.
- Retrieval source metadata should include document type and OCR status so answer quality issues can be traced back to extraction quality.
- Operational tools should expose local vector DB state, but destructive actions need explicit confirmation.
- Diagnostics should expose extraction quality without leaking raw document text.
- Settings should come from environment variables, not hard-coded constants.
- Local LLM verification still needs local embeddings; otherwise PDF upload would still call OpenAI.
- OCR is optional because it depends on local system tools outside the Python package set.
- OCR failures should not discard the original extraction; the app records the error and continues with the original pages.
- The first useful tests can avoid paid APIs by targeting pure helper functions.
