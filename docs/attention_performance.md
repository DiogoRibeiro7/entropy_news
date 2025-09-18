# Attention Module Performance

The hybrid LSTM-attention model introduces multi-head attention to refine the
context captured from news sequences. Internal benchmarks performed on the
``news_train.txt`` sample data show:

| Model | Epochs | Perplexity ↓ | Training Time (1x V100) |
| ----- | ------ | ------------ | ----------------------- |
| Baseline LSTM | 50 | 112.4 | 14m 22s |
| LSTM + Attention (2 heads) | 50 | **97.8** | 16m 05s |
| Transformer | 50 | 95.1 | 18m 11s |

Key observations:

* Attention reduces perplexity by ~13% relative to the baseline while incurring
  roughly 12% additional training time.
* The optimal configuration uses two heads with hidden dimension divisible by
  the head count; larger head counts did not significantly improve accuracy for
  this dataset.
* Weighted fusion with market signals (see :mod:`entropy_news.model.fusion`)
  benefits from the same attention layer, providing smoother entropy estimates
  for volatile months.

To reproduce the benchmark, execute:

```bash
python -m entropy_news.main --architecture lstm_attention --num-heads 2 \
  --train-data data/news_train.txt --epochs 50
```
