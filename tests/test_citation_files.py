"""Verify citation metadata."""

from pathlib import Path


def test_citation_contains_license_and_email() -> None:
    text = Path("CITATION").read_text()
    assert "MIT License" in text
    assert "dfr@esmad.ipp.pt" in text
    assert "diogo.debastos.ribeiro@gmail.com" not in text
    assert "github.com/DiogoRibeiro7/entropy_news" in text


def test_citation_cff_metadata() -> None:
    text = Path("CITATION.cff").read_text()
    assert "license: MIT" in text
    assert "dfr@esmad.ipp.pt" in text
    assert "diogo.debastos.ribeiro@gmail.com" not in text
    assert "url: https://github.com/DiogoRibeiro7/entropy_news" in text
