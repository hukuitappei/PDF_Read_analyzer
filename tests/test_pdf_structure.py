from pdf_structure import table_quality_score


def test_table_quality_score_prefers_more_non_empty_cells():
    sparse = [[["A", ""], ["", ""]]]
    dense = [[["A", "B"], ["C", "D"]]]

    assert table_quality_score(dense) > table_quality_score(sparse)


def test_table_quality_score_penalizes_single_column_noise():
    single_column_noise = [[["A"], [""], ["B"], ["C"]]]
    compact_table = [[["A", "B"], ["", "C"]]]

    assert table_quality_score(compact_table) > table_quality_score(single_column_noise)
