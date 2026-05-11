from pdf_reader_core import (
    build_document_id,
    build_extraction_diagnostics,
    build_career_blocks_context,
    build_resume_summary_prompt,
    build_structured_summary_prompt,
    build_structured_context,
    classify_table_semantic_type,
    extract_career_blocks,
    flatten_structured_pages,
    flatten_structured_pages_with_ids,
    infer_document_type,
    infer_sections,
    normalize_table,
    prune_empty_table_edges,
    score_heading,
    StructuredPage,
)


def test_normalize_table_converts_none_to_empty_string():
    table = normalize_table([["A", None], ["  B  ", 123]])

    assert table == [["A", ""], ["B", "123"]]


def test_prune_empty_table_edges_removes_empty_rows_and_columns():
    table = prune_empty_table_edges([["", "A", ""], ["", "", ""], ["", "B", "C"]])

    assert table == [["A", ""], ["B", "C"]]


def test_infer_sections_detects_known_headings():
    sections = infer_sections("\u8077\u52d9\u8981\u7d04\nPython\u3067\u81ea\u52d5\u5316\u3057\u307e\u3057\u305f\u3002\n\u30b9\u30ad\u30eb\nSQL\u3068Excel")

    assert [section.heading for section in sections] == ["\u8077\u52d9\u8981\u7d04", "\u30b9\u30ad\u30eb"]
    assert sections[0].content == "Python\u3067\u81ea\u52d5\u5316\u3057\u307e\u3057\u305f\u3002"


def test_score_heading_uses_document_heading_terms():
    assert score_heading("\u8b70\u4e8b\u9332") >= 2
    assert score_heading("\u3053\u308c\u306f\u666e\u901a\u306e\u6587\u7ae0\u3067\u3059\u3002") == 0


def test_classify_table_semantic_type_detects_invoice_table():
    table = [["\u9805\u76ee", "\u6570\u91cf", "\u5358\u4fa1", "\u91d1\u984d"], ["A", "1", "100", "100"]]

    assert classify_table_semantic_type(table) == "invoice"


def test_flatten_structured_pages_adds_metadata():
    pages = [
        StructuredPage(
            page_number=1,
            raw_text="\u8077\u52d9\u8981\u7d04\n\u81ea\u5df1PR\nPython\u3067\u6539\u5584\u3057\u307e\u3057\u305f\u3002",
            sections=infer_sections("\u8077\u52d9\u8981\u7d04\nPython\u3067\u6539\u5584\u3057\u307e\u3057\u305f\u3002"),
            tables=[[["\u671f\u9593", "\u5185\u5bb9"], ["2024", "\u81ea\u52d5\u5316"]]],
            keywords=["Python", "\u6539\u5584"],
        )
    ]

    texts, metadatas = flatten_structured_pages(pages)

    assert len(texts) == 3
    assert metadatas[0]["page"] == 1
    assert metadatas[1]["section_type"] == "section"
    assert metadatas[2]["heading"] == "Table 1"
    assert metadatas[0]["document_type"] == "resume"
    assert metadatas[0]["document_type_score"] >= 2
    assert metadatas[0]["needs_ocr"] is False
    assert metadatas[2]["table_semantic_type"] == "career"


def test_flatten_structured_pages_with_ids_is_stable():
    pages = [
        StructuredPage(
            page_number=1,
            raw_text="\u8077\u52d9\u8981\u7d04\n\u81ea\u5df1PR\nPython\u3067\u6539\u5584\u3057\u307e\u3057\u305f\u3002",
            sections=infer_sections("\u8077\u52d9\u8981\u7d04\nPython\u3067\u6539\u5584\u3057\u307e\u3057\u305f\u3002"),
            tables=[[["\u671f\u9593", "\u5185\u5bb9"], ["2024", "\u81ea\u52d5\u5316"]]],
            keywords=["Python", "\u6539\u5584"],
        )
    ]

    first_texts, first_metadatas, first_ids = flatten_structured_pages_with_ids(pages)
    second_texts, second_metadatas, second_ids = flatten_structured_pages_with_ids(pages)

    assert first_texts == second_texts
    assert first_metadatas == second_metadatas
    assert first_ids == second_ids
    assert len(first_ids) == len(set(first_ids)) == 3
    assert all(metadata["document_id"] == build_document_id(pages) for metadata in first_metadatas)
    assert all("document_type_confidence" in metadata for metadata in first_metadatas)


def test_build_structured_context_uses_sections_and_tables():
    pages = [
        StructuredPage(
            page_number=1,
            raw_text="\u8077\u52d9\u8981\u7d04\nPython",
            sections=infer_sections("\u8077\u52d9\u8981\u7d04\nPython"),
            tables=[[["Tool", "Use"], ["Python", "Automation"]]],
            keywords=["Python"],
        )
    ]

    context = build_structured_context(pages)

    assert "[Page 1]" in context
    assert "Section:" in context
    assert "Table 1:" in context


def test_build_resume_summary_prompt_redacts_personal_information_instruction():
    pages = [StructuredPage(page_number=1, raw_text="\u8077\u52d9\u7d4c\u6b74\n\u81ea\u5df1PR\nPython", keywords=["Python"])]

    prompt = build_resume_summary_prompt(pages, user_instruction="\u77ed\u304f\u8981\u7d04")

    assert "\u500b\u4eba\u540d" in prompt
    assert "\u8077\u7a2e" in prompt
    assert "\u4e3b\u306a\u5f37\u307f" in prompt
    assert "\u77ed\u304f\u8981\u7d04" in prompt


def test_infer_document_type_detects_resume():
    pages = [
        StructuredPage(
            page_number=1,
            raw_text="\u8077\u52d9\u7d4c\u6b74\n\u8077\u52d9\u8981\u7d04\n\u30b9\u30ad\u30eb",
            sections=infer_sections("\u8077\u52d9\u7d4c\u6b74\nPython\u3067\u958b\u767a"),
        )
    ]

    profile = infer_document_type(pages)

    assert profile.document_type == "resume"
    assert profile.score >= 2


def test_infer_document_type_detects_report():
    pages = [
        StructuredPage(
            page_number=1,
            raw_text="\u5831\u544a\u66f8\n\u76ee\u7684\n\u7d50\u679c\n\u8ab2\u984c",
            sections=infer_sections("\u76ee\u7684\n\u8a2d\u5099\u6539\u5584\u306e\u5831\u544a"),
        )
    ]

    profile = infer_document_type(pages)

    assert profile.document_type == "report"
    assert "\u5831\u544a\u66f8" in profile.matched_signals


def test_infer_document_type_detects_specification():
    pages = [
        StructuredPage(
            page_number=1,
            raw_text="\u4ed5\u69d8\u66f8\n\u8981\u4ef6\n\u6a5f\u80fd\n\u5236\u7d04",
            sections=infer_sections("\u4ed5\u69d8\u66f8\n\u6a5f\u80fd\u8981\u4ef6\u3092\u8a18\u8f09"),
        )
    ]

    profile = infer_document_type(pages)

    assert profile.document_type == "specification"
    assert profile.score >= 2


def test_infer_document_type_falls_back_to_general_for_weak_signal():
    pages = [StructuredPage(page_number=1, raw_text="\u624b\u9806\u3060\u3051\u304c\u66f8\u304b\u308c\u305f\u77ed\u3044\u6587\u66f8")]

    profile = infer_document_type(pages)

    assert profile.document_type == "general"
    assert profile.score == 1


def test_build_structured_summary_prompt_uses_document_type_schema():
    pages = [
        StructuredPage(
            page_number=1,
            raw_text="\u624b\u9806\n\u64cd\u4f5c\n\u6ce8\u610f",
            sections=infer_sections("\u624b\u9806\n\u88c5\u7f6e\u3092\u8d77\u52d5\u3059\u308b"),
        )
    ]

    prompt = build_structured_summary_prompt(pages, user_instruction="\u8981\u70b9\u3060\u3051")

    assert "\u63a8\u5b9a\u6587\u66f8\u30bf\u30a4\u30d7: manual" in prompt
    assert "\u5bfe\u8c61\u4f5c\u696d" in prompt
    assert "\u8981\u70b9\u3060\u3051" in prompt


def test_build_structured_summary_prompt_uses_contract_schema():
    pages = [
        StructuredPage(
            page_number=1,
            raw_text="\u5951\u7d04\u66f8\n\u7532\n\u4e59\n\u6709\u52b9\u671f\u9593",
            sections=infer_sections("\u5951\u7d04\u66f8\n\u4e3b\u306a\u6761\u9805"),
        )
    ]

    prompt = build_structured_summary_prompt(pages)

    assert "\u63a8\u5b9a\u6587\u66f8\u30bf\u30a4\u30d7: contract" in prompt
    assert "\u6ce8\u610f\u3059\u3079\u304d\u6761\u9805" in prompt


def test_build_extraction_diagnostics_does_not_include_raw_text():
    pages = [
        StructuredPage(
            page_number=1,
            raw_text="Sensitive resume text",
            sections=infer_sections("\u8077\u52d9\u8981\u7d04\nPython"),
            tables=[[["A", ""], ["B", "C"]]],
            keywords=["Python"],
        )
    ]

    diagnostics = build_extraction_diagnostics(pages)

    assert diagnostics["page_count"] == 1
    assert diagnostics["total_tables"] == 1
    assert diagnostics["total_empty_cells"] == 1
    assert diagnostics["extraction_quality"]["needs_ocr"] is False
    assert diagnostics["document_type"]["document_type"] == "general"
    assert diagnostics["pages"][0]["section_heading_scores"][0]["score"] >= 2
    assert diagnostics["pages"][0]["tables"][0]["semantic_type"] == "unknown"
    assert "Sensitive resume text" not in str(diagnostics)


def test_extract_career_blocks_from_sections_and_tables():
    pages = [
        StructuredPage(
            page_number=1,
            raw_text="\u696d\u52d9\u5185\u5bb9\n\u8a2d\u5099\u7dad\u6301\u3068\u6539\u5584",
            sections=infer_sections("\u696d\u52d9\u5185\u5bb9\n\u8a2d\u5099\u7dad\u6301\u3068\u6539\u5584"),
            tables=[[["\u8077\u52d9\u7d4c\u6b74", "\u62c5\u5f53"], ["2024", "\u958b\u767a\u3068\u6539\u5584"]]],
            keywords=["\u6539\u5584"],
        )
    ]

    blocks = extract_career_blocks(pages)
    context = build_career_blocks_context(blocks)

    assert len(blocks) >= 2
    assert "Document Block" in context
    assert any(block.source_type == "table" for block in blocks)
