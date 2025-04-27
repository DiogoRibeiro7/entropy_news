# entropy_news/utils/metrics.py

import math

def perplexity(entropy: float) -> float:
    """Computes perplexity from cross-entropy."""
    if math.isinf(entropy):
        return float('inf')
    return math.exp(entropy)

# Exemplo de uso:
# entropy_value = 3.0
# perplexity_value = perplexity(entropy_value)
# print(f"Perplexity: {perplexity_value:.2f}")
