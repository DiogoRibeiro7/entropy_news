from entropy_news.evaluation import alignment_score, balance_score, modality_contribution


def test_modality_contribution_normalizes() -> None:
    contrib = modality_contribution([2.0, 2.0])
    assert contrib == {"text": 0.5, "market": 0.5}


def test_balance_score_prefers_equal_weights() -> None:
    assert balance_score([1.0, 1.0]) == 1.0
    assert balance_score([1.0, 0.0]) == 0.0


def test_alignment_score_penalises_mismatch() -> None:
    aligned = alignment_score([1.0, 1.0], [1.0, 1.0], [0.5, 0.5])
    misaligned = alignment_score([10.0, 10.0], [0.1, 0.1], [0.1, 0.9])
    assert aligned > misaligned
