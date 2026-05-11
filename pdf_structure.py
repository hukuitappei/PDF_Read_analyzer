from pathlib import Path
from typing import BinaryIO

import pdfplumber

from pdf_reader_core import StructuredPage, diagnose_table, infer_keywords, infer_sections, normalize_table


TABLE_SETTINGS_CANDIDATES = (
    None,
    {
        "vertical_strategy": "lines",
        "horizontal_strategy": "lines",
        "intersection_tolerance": 5,
    },
    {
        "vertical_strategy": "text",
        "horizontal_strategy": "text",
        "snap_tolerance": 3,
        "join_tolerance": 3,
        "min_words_vertical": 2,
        "min_words_horizontal": 1,
    },
)


def table_quality_score(tables: list[list[list[str]]]) -> tuple[int, int, int]:
    diagnostics = [diagnose_table(table) for table in tables]
    useful_tables = [report for report in diagnostics if report["columns"] >= 2]
    non_empty_cells = sum(report["cells"] - report["empty_cells"] for report in useful_tables)
    empty_cells = sum(report["empty_cells"] for report in diagnostics)
    low_quality_tables = sum(1 for report in diagnostics if report["columns"] < 2 or report["empty_cells"] > report["cells"] * 0.5)
    table_count = len(tables)
    return non_empty_cells, -low_quality_tables, -empty_cells, table_count


def extract_best_tables(page) -> list[list[list[str]]]:
    best_tables: list[list[list[str]]] = []
    best_score = (-1, -10**9, -1)
    for settings in TABLE_SETTINGS_CANDIDATES:
        raw_tables = page.extract_tables(table_settings=settings) if settings else page.extract_tables()
        tables = [normalize_table(table) for table in raw_tables]
        tables = [table for table in tables if table]
        score = table_quality_score(tables)
        if score > best_score:
            best_score = score
            best_tables = tables
    return best_tables


def extract_structured_pdf(pdf_source: str | Path | BinaryIO) -> list[StructuredPage]:
    pages: list[StructuredPage] = []
    with pdfplumber.open(pdf_source) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            raw_text = page.extract_text() or ""
            pages.append(
                StructuredPage(
                    page_number=page_number,
                    raw_text=raw_text.strip(),
                    sections=infer_sections(raw_text),
                    tables=extract_best_tables(page),
                    keywords=infer_keywords(raw_text),
                )
            )
    return pages
