from entropy_news.model import ConcatFusion

try:  # pragma: no cover - optional torch dependency
    import torch
except Exception:  # pragma: no cover - torch missing
    torch = None


def test_concat_fusion() -> None:
    fusion = ConcatFusion()
    if torch and hasattr(fusion, "forward"):
        text = torch.tensor([1.0, 2.0])
        market = torch.tensor([3.0])
        result = fusion(text, market)
        assert result.tolist() == [1.0, 2.0, 3.0]
    else:
        result = fusion([1.0, 2.0], [3.0])
        assert result == [1.0, 2.0, 3.0]
