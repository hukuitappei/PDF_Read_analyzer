import hashlib
import os
import shutil
import subprocess
import tempfile
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path


DEFAULT_EMBEDDING_MODEL_NAME = "text-embedding-ada-002"
DEFAULT_EMBEDDING_PROVIDER = "openai"
DEFAULT_LLM_PROVIDER = "openai"
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_OLLAMA_LLM_MODEL = "llama3.1"
DEFAULT_OLLAMA_EMBEDDING_MODEL = "nomic-embed-text"
DEFAULT_QDRANT_PATH = "./local_qdrant"
DEFAULT_COLLECTION_NAME = "my_collection"
DEFAULT_OCR_COMMAND = "ocrmypdf"
DEFAULT_OCR_LANGUAGE = "jpn+eng"
DEFAULT_VECTOR_SIZE_BY_PROVIDER = {
    "openai": 1536,
    "ollama": 768,
}
HEADING_TERMS_BY_TYPE = {
    "resume": (
        "\u8077\u52d9\u8981\u7d04",
        "\u8077\u52d9\u7d4c\u6b74",
        "\u696d\u52d9\u5185\u5bb9",
        "\u4fdd\u6709\u8cc7\u683c",
        "\u6d3b\u304b\u305b\u308b\u7d4c\u9a13",
        "\u30b9\u30ad\u30eb",
        "\u81ea\u5df1PR",
        "\u5fd7\u671b\u52d5\u6a5f",
    ),
    "report": ("\u5831\u544a\u66f8", "\u76ee\u7684", "\u7d50\u679c", "\u8003\u5bdf", "\u8ab2\u984c", "\u5bfe\u7b56", "\u9032\u6357"),
    "manual": ("\u624b\u9806", "\u64cd\u4f5c", "\u8a2d\u5b9a", "\u6ce8\u610f", "\u30c1\u30a7\u30c3\u30af", "\u30d5\u30ed\u30fc"),
    "invoice": ("\u8acb\u6c42\u66f8", "\u898b\u7a4d\u66f8", "\u7d0d\u54c1\u66f8", "\u5408\u8a08", "\u6d88\u8cbb\u7a0e", "\u632f\u8fbc"),
    "meeting_notes": ("\u8b70\u4e8b\u9332", "\u4f1a\u8b70", "\u6c7a\u5b9a\u4e8b\u9805", "\u30a2\u30af\u30b7\u30e7\u30f3", "\u53c2\u52a0\u8005", "\u5bbf\u984c"),
    "contract": ("\u5951\u7d04\u66f8", "\u6761\u9805", "\u7532", "\u4e59", "\u6709\u52b9\u671f\u9593", "\u89e3\u7d04", "\u640d\u5bb3\u8ce0\u511f"),
    "specification": ("\u4ed5\u69d8\u66f8", "\u8981\u4ef6", "\u6a5f\u80fd", "\u975e\u6a5f\u80fd", "\u5236\u7d04", "\u30a4\u30f3\u30bf\u30fc\u30d5\u30a7\u30fc\u30b9"),
    "plan": ("\u8a08\u753b\u66f8", "\u76ee\u6a19", "\u30b9\u30b1\u30b8\u30e5\u30fc\u30eb", "\u4f53\u5236", "\u30ea\u30b9\u30af", "\u30de\u30a4\u30eb\u30b9\u30c8\u30fc\u30f3"),
    "general": ("\u6982\u8981", "\u76ee\u7684", "\u80cc\u666f", "\u8a73\u7d30", "\u8981\u70b9", "\u78ba\u8a8d\u4e8b\u9805"),
}
TABLE_SEMANTIC_TERMS = {
    "invoice": ("\u91d1\u984d", "\u5358\u4fa1", "\u6570\u91cf", "\u5408\u8a08", "\u6d88\u8cbb\u7a0e", "\u8acb\u6c42"),
    "schedule": ("\u65e5\u4ed8", "\u671f\u65e5", "\u4e88\u5b9a", "\u62c5\u5f53", "\u72b6\u614b", "\u9032\u6357"),
    "career": ("\u671f\u9593", "\u8077\u52d9", "\u696d\u52d9", "\u62c5\u5f53", "\u4f1a\u793e", "\u90e8\u7f72"),
    "comparison": ("\u9805\u76ee", "\u6bd4\u8f03", "\u8a55\u4fa1", "\u30e1\u30ea\u30c3\u30c8", "\u30c7\u30e1\u30ea\u30c3\u30c8"),
}


@dataclass(frozen=True)
class AppSettings:
    openai_api_key: str | None
    llm_provider: str = DEFAULT_LLM_PROVIDER
    embedding_provider: str = DEFAULT_EMBEDDING_PROVIDER
    embedding_model_name: str = DEFAULT_EMBEDDING_MODEL_NAME
    ollama_base_url: str = DEFAULT_OLLAMA_BASE_URL
    ollama_llm_model: str = DEFAULT_OLLAMA_LLM_MODEL
    ollama_embedding_model: str = DEFAULT_OLLAMA_EMBEDDING_MODEL
    qdrant_path: str = DEFAULT_QDRANT_PATH
    collection_name: str = DEFAULT_COLLECTION_NAME
    vector_size: int = DEFAULT_VECTOR_SIZE_BY_PROVIDER[DEFAULT_EMBEDDING_PROVIDER]
    ocr_enabled: bool = False
    ocr_command: str = DEFAULT_OCR_COMMAND
    ocr_language: str = DEFAULT_OCR_LANGUAGE


@dataclass(frozen=True)
class PageSection:
    heading: str
    content: str


@dataclass(frozen=True)
class StructuredPage:
    page_number: int
    raw_text: str
    sections: list[PageSection] = field(default_factory=list)
    tables: list[list[list[str]]] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class DocumentBlock:
    page_number: int
    source_type: str
    title: str
    content: str
    block_type: str = "unknown"
    score: int = 0
    signals: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class DocumentTypeProfile:
    document_type: str
    score: int
    confidence: float = 0.0
    matched_signals: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ExtractionQualityReport:
    total_characters: int
    average_page_characters: float
    low_text_pages: list[int] = field(default_factory=list)
    needs_ocr: bool = False
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class OcrAvailability:
    enabled: bool
    command: str
    available: bool
    resolved_path: str | None = None
    message: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class OcrExecutionResult:
    applied: bool
    input_quality: ExtractionQualityReport
    output_quality: ExtractionQualityReport | None = None
    availability: OcrAvailability | None = None
    error: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def _normalized_provider(value: str, name: str) -> str:
    provider = value.strip().lower()
    if provider not in {"openai", "ollama"}:
        raise ValueError(f"{name} must be either 'openai' or 'ollama'.")
    return provider


def _env_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def load_settings(environ: dict[str, str] | None = None) -> AppSettings:
    env = environ if environ is not None else os.environ
    llm_provider = _normalized_provider(env.get("LLM_PROVIDER", DEFAULT_LLM_PROVIDER), "LLM_PROVIDER")
    embedding_provider = _normalized_provider(
        env.get("EMBEDDING_PROVIDER", DEFAULT_EMBEDDING_PROVIDER),
        "EMBEDDING_PROVIDER",
    )
    api_key = env.get("OPENAI_API_KEY", "").strip()
    if (llm_provider == "openai" or embedding_provider == "openai") and (
        not api_key or api_key == "your_openai_api_key_here"
    ):
        raise ValueError("OPENAI_API_KEY is not set. Copy env.example to .env and set a real key.")

    vector_size = int(env.get("VECTOR_SIZE", DEFAULT_VECTOR_SIZE_BY_PROVIDER[embedding_provider]))

    return AppSettings(
        openai_api_key=api_key or None,
        llm_provider=llm_provider,
        embedding_provider=embedding_provider,
        embedding_model_name=env.get("EMBEDDING_MODEL_NAME", DEFAULT_EMBEDDING_MODEL_NAME).strip()
        or DEFAULT_EMBEDDING_MODEL_NAME,
        ollama_base_url=env.get("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL).strip() or DEFAULT_OLLAMA_BASE_URL,
        ollama_llm_model=env.get("OLLAMA_LLM_MODEL", DEFAULT_OLLAMA_LLM_MODEL).strip() or DEFAULT_OLLAMA_LLM_MODEL,
        ollama_embedding_model=env.get("OLLAMA_EMBEDDING_MODEL", DEFAULT_OLLAMA_EMBEDDING_MODEL).strip()
        or DEFAULT_OLLAMA_EMBEDDING_MODEL,
        qdrant_path=env.get("QDRANT_PATH", DEFAULT_QDRANT_PATH).strip() or DEFAULT_QDRANT_PATH,
        collection_name=env.get("COLLECTION_NAME", DEFAULT_COLLECTION_NAME).strip() or DEFAULT_COLLECTION_NAME,
        vector_size=vector_size,
        ocr_enabled=_env_bool(env.get("OCR_ENABLED"), default=False),
        ocr_command=env.get("OCR_COMMAND", DEFAULT_OCR_COMMAND).strip() or DEFAULT_OCR_COMMAND,
        ocr_language=env.get("OCR_LANGUAGE", DEFAULT_OCR_LANGUAGE).strip() or DEFAULT_OCR_LANGUAGE,
    )


def merge_page_texts(page_texts: list[str | None]) -> str:
    return "\n\n".join(text.strip() for text in page_texts if text and text.strip())


def ensure_text_was_extracted(text: str) -> str:
    if not text.strip():
        raise ValueError("No text could be extracted from the PDF. Try a text-based PDF instead of a scanned image.")
    return text


def assess_extraction_quality(
    pages: list[StructuredPage],
    min_total_characters: int = 20,
    min_average_page_characters: int = 40,
) -> ExtractionQualityReport:
    total_characters = sum(len(page.raw_text.strip()) for page in pages)
    average_page_characters = total_characters / len(pages) if pages else 0.0
    low_text_pages = [
        page.page_number
        for page in pages
        if len(page.raw_text.strip()) < min_average_page_characters and not page.tables
    ]
    has_any_table = any(page.tables for page in pages)
    needs_ocr = bool(pages) and (
        total_characters < min_total_characters
        or (average_page_characters < min_average_page_characters and not has_any_table)
        or (len(low_text_pages) == len(pages) and not has_any_table)
    )
    warnings = []
    if needs_ocr:
        warnings.append(
            "Extracted text is very small for the page count. This PDF may be scanned or image-based; OCR may be required."
        )
    return ExtractionQualityReport(
        total_characters=total_characters,
        average_page_characters=round(average_page_characters, 2),
        low_text_pages=low_text_pages,
        needs_ocr=needs_ocr,
        warnings=warnings,
    )


def check_ocr_availability(settings: AppSettings, command_resolver=None) -> OcrAvailability:
    resolver = command_resolver or shutil.which
    if not settings.ocr_enabled:
        return OcrAvailability(
            enabled=False,
            command=settings.ocr_command,
            available=False,
            message="OCR is disabled. Set OCR_ENABLED=true to enable OCR.",
        )

    resolved_path = resolver(settings.ocr_command)
    if not resolved_path:
        return OcrAvailability(
            enabled=True,
            command=settings.ocr_command,
            available=False,
            message=f"OCR command '{settings.ocr_command}' was not found.",
        )

    return OcrAvailability(
        enabled=True,
        command=settings.ocr_command,
        available=True,
        resolved_path=resolved_path,
        message="OCR command is available.",
    )


def ocr_pdf_bytes(pdf_bytes: bytes, settings: AppSettings, runner=None) -> bytes:
    if not settings.ocr_enabled:
        raise ValueError("OCR is disabled. Set OCR_ENABLED=true to run OCR for sparse PDFs.")

    command_runner = runner or subprocess.run
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        input_path = temp_path / "input.pdf"
        output_path = temp_path / "output.pdf"
        input_path.write_bytes(pdf_bytes)

        command = [
            settings.ocr_command,
            "--force-ocr",
            "-l",
            settings.ocr_language,
            str(input_path),
            str(output_path),
        ]
        try:
            command_runner(command, check=True, capture_output=True, text=True)
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"OCR command '{settings.ocr_command}' was not found. Install ocrmypdf or set OCR_COMMAND."
            ) from exc
        except subprocess.CalledProcessError as exc:
            detail = (exc.stderr or exc.stdout or str(exc)).strip()
            raise RuntimeError(f"OCR command failed: {detail}") from exc

        if not output_path.exists():
            raise RuntimeError("OCR command completed but did not create an output PDF.")
        return output_path.read_bytes()


def normalize_table(table: list[list[object | None]] | None) -> list[list[str]]:
    if not table:
        return []
    normalized: list[list[str]] = []
    for row in table:
        normalized.append(["" if cell is None else str(cell).strip() for cell in row])
    return prune_empty_table_edges(normalized)


def prune_empty_table_edges(table: list[list[str]]) -> list[list[str]]:
    rows = [row for row in table if any(cell.strip() for cell in row)]
    if not rows:
        return []

    max_columns = max(len(row) for row in rows)
    padded = [row + [""] * (max_columns - len(row)) for row in rows]
    non_empty_columns = [
        index
        for index in range(max_columns)
        if any(row[index].strip() for row in padded)
    ]
    return [[row[index] for index in non_empty_columns] for row in padded]


def score_heading(line: str) -> int:
    text = line.strip()
    if not text:
        return 0
    score = 0
    if len(text) <= 30 and text.endswith((":", "\uff1a")):
        score += 2
    if len(text) <= 40:
        matched_terms = [
            term
            for terms in HEADING_TERMS_BY_TYPE.values()
            for term in terms
            if term.lower() in text.lower()
        ]
        score += len(set(matched_terms)) * 2
    if len(text) <= 18 and not any(separator in text for separator in ("。", ".", ",", "\u3001")):
        score += 1
    return score


def looks_like_heading(line: str) -> bool:
    return score_heading(line) >= 2


def infer_sections(raw_text: str) -> list[PageSection]:
    sections: list[PageSection] = []
    current_heading = "Page Body"
    current_lines: list[str] = []

    for line in raw_text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if looks_like_heading(stripped):
            if current_lines:
                sections.append(PageSection(current_heading, "\n".join(current_lines)))
            current_heading = stripped.rstrip(":\uff1a")
            current_lines = []
        else:
            current_lines.append(stripped)

    if current_lines:
        sections.append(PageSection(current_heading, "\n".join(current_lines)))
    return sections


def classify_table_semantic_type(table: list[list[str]]) -> str:
    text = "\n".join(" ".join(row) for row in table)
    scores = {
        semantic_type: sum(1 for term in terms if term.lower() in text.lower())
        for semantic_type, terms in TABLE_SEMANTIC_TERMS.items()
    }
    semantic_type, score = max(scores.items(), key=lambda item: item[1])
    return semantic_type if score > 0 else "unknown"


def infer_keywords(raw_text: str, limit: int = 12) -> list[str]:
    candidates = (
        "Python",
        "AI",
        "FastAPI",
        "Streamlit",
        "SQL",
        "Excel",
        "VBA",
        "\u88fd\u9020",
        "\u54c1\u8cea",
        "\u6539\u5584",
        "\u81ea\u52d5\u5316",
        "\u8a2d\u8a08",
        "\u958b\u767a",
        "\u904b\u7528",
        "\u5206\u6790",
    )
    keywords = [term for term in candidates if term.lower() in raw_text.lower()]
    return keywords[:limit]


def infer_document_type(pages: list[StructuredPage], min_score: int = 2) -> DocumentTypeProfile:
    text = "\n".join(page.raw_text for page in pages)
    headings = "\n".join(section.heading for page in pages for section in page.sections)
    combined = f"{headings}\n{text}"
    type_terms = {
        "resume": (
            "\u8077\u52d9\u7d4c\u6b74",
            "\u8077\u52d9\u8981\u7d04",
            "\u81ea\u5df1PR",
            "\u4fdd\u6709\u8cc7\u683c",
            "\u30b9\u30ad\u30eb",
        ),
        "report": (
            "\u5831\u544a\u66f8",
            "\u76ee\u7684",
            "\u7d50\u679c",
            "\u8003\u5bdf",
            "\u8ab2\u984c",
            "\u5bfe\u7b56",
            "\u9032\u6357",
        ),
        "manual": (
            "\u624b\u9806",
            "\u64cd\u4f5c",
            "\u8a2d\u5b9a",
            "\u6ce8\u610f",
            "\u30c1\u30a7\u30c3\u30af",
            "\u30d5\u30ed\u30fc",
        ),
        "invoice": (
            "\u8acb\u6c42\u66f8",
            "\u898b\u7a4d\u66f8",
            "\u7d0d\u54c1\u66f8",
            "\u5408\u8a08",
            "\u6d88\u8cbb\u7a0e",
            "\u632f\u8fbc",
        ),
        "meeting_notes": (
            "\u8b70\u4e8b\u9332",
            "\u4f1a\u8b70",
            "\u6c7a\u5b9a\u4e8b\u9805",
            "\u30a2\u30af\u30b7\u30e7\u30f3",
            "\u53c2\u52a0\u8005",
            "\u5bbf\u984c",
        ),
        "contract": ("\u5951\u7d04\u66f8", "\u6761\u9805", "\u7532", "\u4e59", "\u6709\u52b9\u671f\u9593", "\u89e3\u7d04", "\u640d\u5bb3\u8ce0\u511f"),
        "specification": ("\u4ed5\u69d8\u66f8", "\u8981\u4ef6", "\u6a5f\u80fd", "\u975e\u6a5f\u80fd", "\u5236\u7d04", "\u30a4\u30f3\u30bf\u30fc\u30d5\u30a7\u30fc\u30b9"),
        "plan": ("\u8a08\u753b\u66f8", "\u76ee\u6a19", "\u30b9\u30b1\u30b8\u30e5\u30fc\u30eb", "\u4f53\u5236", "\u30ea\u30b9\u30af", "\u30de\u30a4\u30eb\u30b9\u30c8\u30fc\u30f3"),
    }
    scores: dict[str, tuple[int, list[str]]] = {}
    for document_type, terms in type_terms.items():
        matched = [term for term in terms if term.lower() in combined.lower()]
        scores[document_type] = (len(matched), matched)

    document_type, (score, matched_signals) = max(scores.items(), key=lambda item: item[1][0])
    confidence = score / len(type_terms[document_type])
    if score < min_score:
        return DocumentTypeProfile(
            document_type="general",
            score=score,
            confidence=round(confidence, 2),
            matched_signals=matched_signals,
        )
    return DocumentTypeProfile(
        document_type=document_type,
        score=score,
        confidence=round(confidence, 2),
        matched_signals=matched_signals,
    )


def build_document_id(pages: list[StructuredPage]) -> str:
    content = "\n\n".join(page.raw_text for page in pages)
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
    return digest[:16]


def build_record_id(document_id: str, text: str, metadata: dict) -> str:
    key = "|".join(
        [
            document_id,
            str(metadata.get("page", "")),
            str(metadata.get("section_type", "")),
            str(metadata.get("heading", "")),
            hashlib.sha256(text.encode("utf-8")).hexdigest(),
        ]
    )
    return str(uuid.uuid5(uuid.NAMESPACE_URL, key))


def flatten_structured_pages_with_ids(pages: list[StructuredPage]) -> tuple[list[str], list[dict], list[str]]:
    texts: list[str] = []
    metadatas: list[dict] = []
    ids: list[str] = []
    document_id = build_document_id(pages)
    document_type_profile = infer_document_type(pages)
    extraction_quality = assess_extraction_quality(pages)
    common_metadata = {
        "document_id": document_id,
        "document_type": document_type_profile.document_type,
        "document_type_score": document_type_profile.score,
        "document_type_confidence": document_type_profile.confidence,
        "needs_ocr": extraction_quality.needs_ocr,
    }

    def append_record(text: str, metadata: dict):
        metadata = {**common_metadata, **metadata}
        texts.append(text)
        metadatas.append(metadata)
        ids.append(build_record_id(document_id, text, metadata))

    for page in pages:
        if page.raw_text.strip():
            append_record(
                page.raw_text,
                {
                    "page": page.page_number,
                    "section_type": "page",
                    "heading": "Full page text",
                    "keywords": ", ".join(page.keywords),
                    "table_count": len(page.tables),
                },
            )

        for section in page.sections:
            if section.content.strip():
                append_record(
                    section.content,
                    {
                        "page": page.page_number,
                        "section_type": "section",
                        "heading": section.heading,
                        "keywords": ", ".join(page.keywords),
                        "table_count": len(page.tables),
                    },
                )

        for index, table in enumerate(page.tables, start=1):
            rows = [" | ".join(cell for cell in row if cell) for row in table]
            table_text = "\n".join(row for row in rows if row.strip())
            if table_text.strip():
                append_record(
                    table_text,
                    {
                        "page": page.page_number,
                        "section_type": "table",
                        "heading": f"Table {index}",
                        "keywords": ", ".join(page.keywords),
                        "table_count": len(page.tables),
                        "table_semantic_type": classify_table_semantic_type(table),
                    },
                )

    return texts, metadatas, ids


def flatten_structured_pages(pages: list[StructuredPage]) -> tuple[list[str], list[dict]]:
    texts, metadatas, _ids = flatten_structured_pages_with_ids(pages)
    return texts, metadatas


def table_to_lines(table: list[list[str]]) -> list[str]:
    return [" | ".join(cell for cell in row if cell.strip()) for row in table if any(cell.strip() for cell in row)]


def contains_any(text: str, terms: tuple[str, ...]) -> bool:
    return any(term.lower() in text.lower() for term in terms)


def infer_block_signals(text: str) -> list[str]:
    signal_terms = {
        "career": (
            "\u8077\u52d9\u7d4c\u6b74",
            "\u696d\u52d9\u5185\u5bb9",
            "\u5b9f\u52d9",
            "\u7d4c\u9a13",
            "\u90e8\u9580",
            "\u30e1\u30f3\u30d0\u30fc",
        ),
        "task": ("\u62c5\u5f53", "\u7ba1\u7406", "\u7dad\u6301", "\u8a2d\u8a08", "\u958b\u767a", "\u5bfe\u5fdc"),
        "achievement": ("\u6539\u5584", "\u52b9\u7387", "\u5c0e\u5165", "\u524a\u6e1b", "\u6210\u679c"),
        "skill": ("Python", "AI", "FastAPI", "Streamlit", "SQL", "Excel", "VBA"),
        "procedure": ("\u624b\u9806", "\u65b9\u6cd5", "\u30d5\u30ed\u30fc", "\u30c1\u30a7\u30c3\u30af", "\u78ba\u8a8d"),
        "decision": ("\u6c7a\u5b9a", "\u65b9\u91dd", "\u5224\u65ad", "\u63a1\u7528"),
    }
    signals = []
    for label, terms in signal_terms.items():
        if contains_any(text, terms):
            signals.append(label)
    return signals


def infer_block_type(signals: list[str]) -> str:
    for candidate in ("career", "achievement", "task", "skill", "procedure", "decision"):
        if candidate in signals:
            return candidate
    return "unknown"


def score_document_block(source_type: str, title: str, content: str, signals: list[str]) -> int:
    score = len(signals)
    if source_type == "table":
        score += 1
    if contains_any(title, ("\u696d\u52d9\u5185\u5bb9", "\u8077\u52d9\u7d4c\u6b74", "\u81ea\u5df1PR")):
        score += 1
    if len(content) > 80:
        score += 1
    if "achievement" in signals:
        score += 1
    return score


def extract_document_blocks(pages: list[StructuredPage]) -> list[DocumentBlock]:
    blocks: list[DocumentBlock] = []
    for page in pages:
        for section in page.sections:
            source = f"{section.heading}\n{section.content}"
            signals = infer_block_signals(source)
            if not signals and len(section.content) < 80:
                continue
            block_type = infer_block_type(signals)
            blocks.append(
                DocumentBlock(
                    page_number=page.page_number,
                    source_type="section",
                    title=section.heading,
                    content=section.content,
                    block_type=block_type,
                    score=score_document_block("section", section.heading, section.content, signals),
                    signals=signals,
                )
            )

        for index, table in enumerate(page.tables, start=1):
            lines = table_to_lines(table)
            table_text = "\n".join(lines)
            signals = infer_block_signals(table_text)
            if not signals and len(table_text) < 80:
                continue
            title = f"Table {index}"
            block_type = infer_block_type(signals)
            blocks.append(
                DocumentBlock(
                    page_number=page.page_number,
                    source_type="table",
                    title=title,
                    content=table_text,
                    block_type=block_type,
                    score=score_document_block("table", title, table_text, signals),
                    signals=signals,
                )
            )

    return sorted(blocks, key=lambda block: block.score, reverse=True)


def extract_career_blocks(pages: list[StructuredPage]) -> list[DocumentBlock]:
    preferred_types = {"career", "achievement", "task", "skill"}
    return [block for block in extract_document_blocks(pages) if block.block_type in preferred_types]


def build_document_blocks_context(blocks: list[DocumentBlock], max_characters: int = 4000) -> str:
    parts: list[str] = []
    for index, block in enumerate(blocks, start=1):
        parts.append(
            "\n".join(
                [
                    f"[Document Block {index}]",
                    f"page: {block.page_number}",
                    f"source_type: {block.source_type}",
                    f"block_type: {block.block_type}",
                    f"score: {block.score}",
                    f"signals: {', '.join(block.signals)}",
                    f"title: {block.title}",
                    f"content: {block.content}",
                ]
            )
        )

    context = "\n\n".join(parts)
    if len(context) <= max_characters:
        return context
    return context[:max_characters] + "\n\n[TRUNCATED]"


def build_career_blocks_context(blocks: list[DocumentBlock], max_characters: int = 4000) -> str:
    return build_document_blocks_context(blocks, max_characters=max_characters)


def get_summary_schema(document_type: str) -> str:
    schemas = {
        "resume": "\n".join(
            [
                "1. \u8077\u7a2e",
                "2. \u7d4c\u9a13\u9818\u57df",
                "3. \u4e3b\u306a\u5f37\u307f",
            ]
        ),
        "report": "\n".join(
            [
                "1. \u76ee\u7684",
                "2. \u4e3b\u306a\u7d50\u679c",
                "3. \u8ab2\u984c\u307e\u305f\u306f\u6b21\u30a2\u30af\u30b7\u30e7\u30f3",
            ]
        ),
        "manual": "\n".join(
            [
                "1. \u5bfe\u8c61\u4f5c\u696d",
                "2. \u4e3b\u306a\u624b\u9806",
                "3. \u6ce8\u610f\u70b9",
            ]
        ),
        "invoice": "\n".join(
            [
                "1. \u6587\u66f8\u306e\u76ee\u7684",
                "2. \u91d1\u984d\u30fb\u671f\u65e5\u306a\u3069\u306e\u8981\u70b9",
                "3. \u78ba\u8a8d\u304c\u5fc5\u8981\u306a\u70b9",
            ]
        ),
        "meeting_notes": "\n".join(
            [
                "1. \u8b70\u984c",
                "2. \u6c7a\u5b9a\u4e8b\u9805",
                "3. \u30a2\u30af\u30b7\u30e7\u30f3\u30a2\u30a4\u30c6\u30e0",
            ]
        ),
        "contract": "\n".join(
            [
                "1. \u5951\u7d04\u306e\u76ee\u7684\u3068\u5f53\u4e8b\u8005",
                "2. \u4e3b\u306a\u7fa9\u52d9\u30fb\u671f\u9593\u30fb\u91d1\u984d",
                "3. \u6ce8\u610f\u3059\u3079\u304d\u6761\u9805",
            ]
        ),
        "specification": "\n".join(
            [
                "1. \u5bfe\u8c61\u7bc4\u56f2",
                "2. \u4e3b\u8981\u8981\u4ef6\u30fb\u6a5f\u80fd",
                "3. \u5236\u7d04\u30fb\u78ba\u8a8d\u4e8b\u9805",
            ]
        ),
        "plan": "\n".join(
            [
                "1. \u76ee\u6a19\u3068\u6210\u679c\u7269",
                "2. \u4f53\u5236\u3068\u30b9\u30b1\u30b8\u30e5\u30fc\u30eb",
                "3. \u30ea\u30b9\u30af\u3068\u5bfe\u5fdc\u65b9\u91dd",
            ]
        ),
        "general": "\n".join(
            [
                "1. \u6587\u66f8\u306e\u76ee\u7684",
                "2. \u4e3b\u306a\u5185\u5bb9",
                "3. \u78ba\u8a8d\u304c\u5fc5\u8981\u306a\u70b9",
            ]
        ),
    }
    return schemas.get(document_type, schemas["general"])


def get_priority_block_types(document_type: str) -> set[str]:
    priorities = {
        "resume": {"career", "achievement", "task", "skill"},
        "report": {"achievement", "decision", "task"},
        "manual": {"procedure", "task", "decision"},
        "invoice": {"decision", "task"},
        "meeting_notes": {"decision", "task"},
        "contract": {"decision", "task"},
        "specification": {"procedure", "decision", "task"},
        "plan": {"decision", "task", "procedure"},
        "general": {"decision", "procedure", "achievement", "task", "skill", "career"},
    }
    return priorities.get(document_type, priorities["general"])


def build_structured_summary_prompt(pages: list[StructuredPage], user_instruction: str | None = None) -> str:
    profile = infer_document_type(pages)
    document_type = profile.document_type
    context = build_structured_context(pages)
    document_blocks = extract_document_blocks(pages)
    priority_blocks = [block for block in document_blocks if block.block_type in get_priority_block_types(document_type)]
    priority_context = build_document_blocks_context(priority_blocks)
    document_context = build_document_blocks_context(document_blocks)
    extra_instruction = f"\n\u8ffd\u52a0\u306e\u8cea\u554f\u307e\u305f\u306f\u6307\u793a: {user_instruction}\n" if user_instruction else ""
    matched_signals = ", ".join(profile.matched_signals) if profile.matched_signals else "none"
    return (
        "\u4ee5\u4e0b\u306fPDF\u304b\u3089\u30da\u30fc\u30b8\u5358\u4f4d\u3067\u62bd\u51fa\u3057\u305f"
        "\u69cb\u9020\u5316\u30c7\u30fc\u30bf\u3067\u3059\u3002\u500b\u4eba\u540d\u3001\u4f4f\u6240\u3001"
        "\u96fb\u8a71\u756a\u53f7\u3001\u30e1\u30fc\u30eb\u30a2\u30c9\u30ec\u30b9\u306a\u3069\u306e"
        "\u500b\u4eba\u60c5\u5831\u306f\u5fc5\u8981\u304c\u306a\u3044\u9650\u308a\u51fa\u529b\u3057\u306a\u3044\u3067\u304f\u3060\u3055\u3044\u3002\n"
        f"\u63a8\u5b9a\u6587\u66f8\u30bf\u30a4\u30d7: {document_type}\n"
        f"\u63a8\u5b9a\u6839\u62e0\u30b7\u30b0\u30ca\u30eb: {matched_signals}\n\n"
        "\u6b21\u306e\u9805\u76ee\u3067\u7c21\u6f54\u306b\u6574\u7406\u3057\u3066\u304f\u3060\u3055\u3044\u3002\n"
        f"{get_summary_schema(document_type)}\n\n"
        "\u5404\u9805\u76ee\u306f1\u301c3\u6587\u306b\u3057\u3066\u3001\u6839\u62e0\u304c\u69cb\u9020\u5316\u30c7\u30fc\u30bf\u306b\u3042\u308b\u5834\u5408\u3060\u3051\u66f8\u3044\u3066\u304f\u3060\u3055\u3044\u3002\n"
        "\u63a8\u5b9a\u6587\u66f8\u30bf\u30a4\u30d7\u306f\u88dc\u52a9\u60c5\u5831\u3067\u3059\u3002\u30e6\u30fc\u30b6\u30fc\u306e\u6307\u793a\u3068PDF\u5185\u5bb9\u3092\u512a\u5148\u3057\u3066\u304f\u3060\u3055\u3044\u3002\n"
        "\u307e\u305aPriority Document Blocks\u3092\u512a\u5148\u3057\u3001\u4e0d\u8db3\u3059\u308b\u5834\u5408\u306fDocument Blocks\u3068Full Structured Context\u3092\u6839\u62e0\u3068\u3057\u3066\u53c2\u7167\u3057\u3066\u304f\u3060\u3055\u3044\u3002\n"
        f"{extra_instruction}\n"
        f"Priority Document Blocks:\n{priority_context}\n\n"
        f"Document Blocks:\n{document_context}\n\n"
        "Full Structured Context:\n"
        f"{context}"
    )


def _legacy_extract_career_blocks_unused(pages: list[StructuredPage]) -> list[DocumentBlock]:
    career_terms = (
        "\u8077\u52d9\u7d4c\u6b74",
        "\u696d\u52d9\u5185\u5bb9",
        "\u5b9f\u52d9",
        "\u7d4c\u9a13",
        "\u90e8\u9580",
        "\u30e1\u30f3\u30d0\u30fc",
    )
    responsibility_terms = ("\u62c5\u5f53", "\u7ba1\u7406", "\u7dad\u6301", "\u8a2d\u8a08", "\u958b\u767a", "\u5bfe\u5fdc")
    achievement_terms = ("\u6539\u5584", "\u52b9\u7387", "\u5c0e\u5165", "\u7dad\u6301", "\u524a\u6e1b", "\u6210\u679c")
    blocks: list[DocumentBlock] = []

    for page in pages:
        for section in page.sections:
            source = f"{section.heading}\n{section.content}"
            if not contains_any(source, career_terms + responsibility_terms + achievement_terms):
                continue
            blocks.append(
                DocumentBlock(
                    page_number=page.page_number,
                    source_type="section",
                    title=section.heading,
                    content=section.content,
                    block_type="career",
                    score=1,
                )
            )

        for index, table in enumerate(page.tables, start=1):
            lines = table_to_lines(table)
            table_text = "\n".join(lines)
            if not contains_any(table_text, career_terms + responsibility_terms + achievement_terms):
                continue
            blocks.append(
                DocumentBlock(
                    page_number=page.page_number,
                    source_type="table",
                    title=f"Table {index}",
                    content=table_text,
                    block_type="career",
                    score=1,
                )
            )

    return blocks


def diagnose_table(table: list[list[str]]) -> dict:
    row_count = len(table)
    column_count = max((len(row) for row in table), default=0)
    cell_count = sum(len(row) for row in table)
    empty_cell_count = sum(1 for row in table for cell in row if not cell.strip())
    short_cell_count = sum(1 for row in table for cell in row if 0 < len(cell.strip()) <= 2)
    ragged_rows = sum(1 for row in table if len(row) != column_count)
    return {
        "rows": row_count,
        "columns": column_count,
        "cells": cell_count,
        "empty_cells": empty_cell_count,
        "short_cells": short_cell_count,
        "ragged_rows": ragged_rows,
        "semantic_type": classify_table_semantic_type(table),
    }


def build_extraction_diagnostics(pages: list[StructuredPage]) -> dict:
    extraction_quality = assess_extraction_quality(pages)
    document_type_profile = infer_document_type(pages)
    document_blocks = extract_document_blocks(pages)
    career_blocks = [block for block in document_blocks if block.block_type in {"career", "achievement", "task", "skill"}]
    page_reports = []
    total_tables = 0
    total_empty_cells = 0
    total_ragged_rows = 0

    for page in pages:
        table_reports = [diagnose_table(table) for table in page.tables]
        total_tables += len(table_reports)
        total_empty_cells += sum(report["empty_cells"] for report in table_reports)
        total_ragged_rows += sum(report["ragged_rows"] for report in table_reports)
        page_reports.append(
            {
                "page": page.page_number,
                "characters": len(page.raw_text),
                "line_count": len([line for line in page.raw_text.splitlines() if line.strip()]),
                "section_count": len(page.sections),
                "section_headings": [section.heading for section in page.sections],
                "section_heading_scores": [
                    {"heading": section.heading, "score": score_heading(section.heading)}
                    for section in page.sections
                    if section.heading != "Page Body"
                ],
                "table_count": len(page.tables),
                "tables": table_reports,
                "keywords": page.keywords,
            }
        )

    return {
        "page_count": len(pages),
        "total_characters": sum(len(page.raw_text) for page in pages),
        "total_sections": sum(len(page.sections) for page in pages),
        "total_tables": total_tables,
        "total_empty_cells": total_empty_cells,
        "total_ragged_rows": total_ragged_rows,
        "extraction_quality": extraction_quality.to_dict(),
        "document_type": document_type_profile.to_dict(),
        "document_block_count": len(document_blocks),
        "document_blocks": [
            {
                "page": block.page_number,
                "source_type": block.source_type,
                "title": block.title,
                "block_type": block.block_type,
                "score": block.score,
                "signals": block.signals,
            }
            for block in document_blocks
        ],
        "career_block_count": len(career_blocks),
        "career_blocks": [
            {
                "page": block.page_number,
                "source_type": block.source_type,
                "title": block.title,
                "block_type": block.block_type,
                "score": block.score,
                "signals": block.signals,
            }
            for block in career_blocks
        ],
        "pages": page_reports,
    }


def build_structured_context(pages: list[StructuredPage], max_characters: int = 6000) -> str:
    blocks: list[str] = []
    for page in pages:
        blocks.append(f"[Page {page.page_number}]")
        if page.keywords:
            blocks.append("Keywords: " + ", ".join(page.keywords))
        for section in page.sections:
            blocks.append(f"Section: {section.heading}\n{section.content}")
        for index, table in enumerate(page.tables, start=1):
            rows = [" | ".join(cell for cell in row if cell) for row in table]
            table_text = "\n".join(row for row in rows if row.strip())
            if table_text:
                blocks.append(f"Table {index}:\n{table_text}")
        if not page.sections and page.raw_text:
            blocks.append(page.raw_text)

    context = "\n\n".join(blocks)
    if len(context) <= max_characters:
        return context
    return context[:max_characters] + "\n\n[TRUNCATED]"


def build_resume_summary_prompt(pages: list[StructuredPage], user_instruction: str | None = None) -> str:
    return build_structured_summary_prompt(pages, user_instruction=user_instruction)
