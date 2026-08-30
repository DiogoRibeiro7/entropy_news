import pytest

pytest.importorskip("torch")

from entropy_news.paper_rolling import run_paper_reproduction


def _months(count: int = 19) -> list[str]:
    months: list[str] = []
    year, month = 2022, 1
    for _ in range(count):
        months.append(f"{year:04d}-{month:02d}")
        month += 1
        if month == 13:
            month = 1
            year += 1
    return months


def test_paper_runner_rejects_too_few_months(tmp_path) -> None:
    with pytest.raises(ValueError, match="at least 19 ordered months"):
        run_paper_reproduction(
            _months(18),
            str(tmp_path),
            glove_path=str(tmp_path / "unused-glove.txt"),
            show_progress=False,
        )


def test_paper_runner_rejects_non_zero_padded_month(tmp_path) -> None:
    months = _months()
    months[0] = "2022-1"
    with pytest.raises(ValueError, match="zero-padded YYYY-MM"):
        run_paper_reproduction(
            months,
            str(tmp_path),
            glove_path=str(tmp_path / "unused-glove.txt"),
            show_progress=False,
        )
