import math
from entropy_news.utils.metrics import perplexity

def test_perplexity_basic():
    assert perplexity(0.0) == 1.0
    assert perplexity(math.log(2)) == 2.0

def test_perplexity_inf():
    assert math.isinf(perplexity(float('inf')))
