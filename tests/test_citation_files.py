"""Verify citation metadata."""

from pathlib import Path


CURRENT_AFFILIATION = "Faculty of Media Arts and Design, Technical University of Porto"
SOURCE_DOI = "10.2139/ssrn.4555832"


def test_citation_contains_license_contact_and_source() -> None:
    text = Path("CITATION").read_text()
    assert "MIT License" in text
    assert "dfr@esmad.ipp.pt" in text
    assert CURRENT_AFFILIATION in text
    assert SOURCE_DOI in text
    assert "github.com/DiogoRibeiro7/entropy_news" in text
    assert "ESMAD - Instituto Politécnico do Porto" not in text


def test_citation_cff_metadata() -> None:
    text = Path("CITATION.cff").read_text()
    assert "license: MIT" in text
    assert "type: software" in text
    assert "dfr@esmad.ipp.pt" in text
    assert CURRENT_AFFILIATION in text
    assert SOURCE_DOI in text
    assert "repository-code: https://github.com/DiogoRibeiro7/entropy_news" in text
    assert "ESMAD - Instituto Politécnico do Porto" not in text
