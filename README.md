# Entropy News

Python research software for reproducing and extending the financial-news entropy methodology in **Glasserman, Mamaysky, and Qin (2023), _New News is Bad News_**.

The repository separates two distinct layers:

1. **Paper reproduction core** — a rolling implementation of the paper's monthly entropy measure and its `ENT_NEWS` / `ENT_MODEL` decomposition.
2. **Research and engineering extensions** — alternative architectures, one-month model-update diagnostics, market-data tooling, causal analysis, dashboards, distributed training, monitoring, and deployment support.

That distinction matters: the paper's `ENT` is a 12-month change in monthly article entropy, not a one-month cross-entropy level.

## Paper reproduction

The reproduction path implements the three quantities

\[
A_t = m_{[t-6,t-1]}(t),\qquad
B_t = m_{[t-18,t-13]}(t-12),\qquad
C_t = m_{[t-18,t-13]}(t),
\]

and then computes

\[
ENT_t = A_t-B_t,
\]

\[
ENT\_NEWS_t = C_t-B_t,
\]

\[
ENT\_MODEL_t = A_t-C_t,
\]

so that

\[
ENT_t = ENT\_NEWS_t + ENT\_MODEL_t
\]

by construction.

The implementation follows the paper's reported rolling training design:

- month `t` is **never used to fit the model weights that score month `t`**;
- the initial language model is trained on six prior months;
- subsequent monthly updates warm-start the previous model and use all articles from `t-1`, then exponentially downsampled articles from `t-2` through `t-6`;
- complete articles are used rather than truncating each article to its first sequence;
- recurrent state is carried across chunks of the same article and reset between articles;
- monthly entropy is the **equal-weighted mean of article entropies**, so long articles do not receive larger weights merely because they contain more tokens;
- the 12-month lagged model is retained so `ENT_NEWS` and `ENT_MODEL` use the paper's actual decomposition rather than two entropy levels from the same month;
- requested months must be strictly consecutive calendar months;
- the paper path requires an explicit GloVe file rather than silently falling back to random embeddings;
- following the paper literally, the vocabulary is selected from the **whole requested corpus** before rolling model training.

The last point is important. The rolling model weights remain causal, but vocabulary membership can reflect later corpus observations because the paper defines its retained vocabulary over the whole corpus. This is therefore a paper-reproduction choice, not a strictly real-time vocabulary construction rule.

The scientific contract lives in:

- `entropy_news/paper_reproduction.py`
- `entropy_news/paper_rolling.py`
- `tests/test_paper_reproduction.py`

### Run the paper path

Monthly files are expected as:

```text
data/news_YYYY-MM.txt
```

with one article per non-empty line. The month arguments must form an unbroken monthly sequence.

Install the project and PyTorch dependencies, then run:

```bash
entropy-news-paper \
  2020-01 2020-02 2020-03 ... 2022-12 \
  --base-data-dir data/ \
  --glove-path /path/to/glove.6B.100d.txt \
  --output-dir output/paper_reproduction
```

At least 19 consecutive months are required before a paper `ENT` observation can be formed. The runner writes:

```text
output/paper_reproduction/paper_entropy_results.csv
output/paper_reproduction/paper_protocol.json
output/paper_reproduction/paper_vocabulary.json
output/paper_reproduction/paper_run_manifest.json
```

The CSV contains:

```text
month
current_entropy
year_ago_entropy
year_ago_model_on_current
ENT
ENT_NEWS
ENT_MODEL
```

`paper_vocabulary.json` stores the exact token-to-index mapping used by the run. `paper_run_manifest.json` records the execution environment, Git revision, learning rate, protocol, vocabulary scope and actual size, every requested monthly input file, raw and qualifying article counts, SHA-256 hashes and byte sizes for all monthly inputs and the GloVe file, plus SHA-256 hashes of the result, protocol and vocabulary outputs.

The paper text says it retains the 10,000 most frequent words and maps other words to `UNK`, while also reporting a 10,000-unit output layer. The current implementation exposes both the configured top-word count and the actual model-vocabulary entry count in the manifest rather than silently resolving that wording ambiguity. A future architecture change should only be made if the original implementation or supplementary source establishes the intended class-count convention.

## One-month diagnostics are not paper ENT

`entropy-news-forecast` remains useful for comparing a baseline model and an updated model on a single dataset. It reports:

```text
baseline_entropy
updated_entropy
model_update_delta
```

These are entropy-level diagnostics. They are deliberately **not** named `ENT`, `ENT_NEWS`, or `ENT_MODEL`, because those paper quantities require the 12-month lagged design above.

Likewise, `entropy-news-rolling` is retained as a generic rolling/fine-tuning workflow for experiments and operational use. Use `entropy-news-paper` when the goal is scientific reproduction of Glasserman, Mamaysky, and Qin.

## Model and data layer

The repository includes:

- LSTM language modelling with optional frozen GloVe embeddings outside the strict paper path;
- alternative LSTM-attention and Transformer architectures for extensions;
- configurable preprocessing and vocabularies;
- streaming datasets for larger corpora;
- entropy and perplexity utilities;
- market-data connectors and correlation tooling;
- causal-analysis helpers;
- Streamlit visualisation;
- Docker and deployment material;
- distributed/orchestration utilities;
- tests, CI, citation metadata, and documentation.

The dashboard prefers `ENT`, `ENT_NEWS`, and `ENT_MODEL` when a paper-reproduction CSV is supplied. It can also display the separately named one-month diagnostic metrics.

## Installation

```bash
git clone https://github.com/DiogoRibeiro7/entropy_news.git
cd entropy_news
poetry install
```

Install the PyTorch dependency group as required by your Poetry version/configuration.

For GloVe 100-dimensional embeddings, use the Stanford GloVe distribution and pass the extracted vector file to `--glove-path`.

## Other commands

| Command | Purpose |
| --- | --- |
| `entropy-news-paper` | Paper reproduction rolling entropy path |
| `entropy-news-train` | Train a configurable language model |
| `entropy-news-forecast` | One-month baseline-vs-updated entropy diagnostics |
| `entropy-news-rolling` | Generic rolling/fine-tuning research workflow |
| `entropy-news-eval` | Entropy/perplexity evaluation for a saved model |
| `entropy-news-orchestrate` | Distributed-training/orchestration tooling |

## Testing

```bash
pytest -q
```

The reproduction tests specifically lock:

- Equation (15) decomposition identity;
- complete-article chunk coverage;
- deterministic exponential historical sampling;
- equal weighting across articles;
- separation of paper metrics from one-month diagnostic labels;
- mandatory GloVe for the strict paper path;
- consecutive-month validation;
- whole-requested-corpus vocabulary construction;
- byte-level input/output provenance in the run manifest.

## Data and reproducibility

The original Reuters corpus used in the paper is not distributed with this repository. Users must supply a legally obtained corpus with monthly article boundaries. A run that uses another news source can validate the implementation but should be described as a **methodological reproduction on alternative data**, not an empirical replication of the paper's Reuters results.

Every paper run with an output directory records the protocol, vocabulary and provenance manifest. Keep these files with any manuscript-facing result so the exact data files, GloVe bytes, filtering counts, random seed, vocabulary, software revision and generated outputs can be audited later.

## Extensions

The following are intentionally outside the narrow paper reproduction contract:

- Transformer and attention architectures;
- multimodal text/market fusion;
- causal estimators and counterfactual tooling;
- Streamlit dashboards;
- quantisation and ONNX export;
- Docker/monitoring/orchestration infrastructure.

They remain useful research and engineering extensions, but they should not be cited as part of the original Glasserman–Mamaysky–Qin estimator.

## Reference

Paul Glasserman, Harry Mamaysky, and Jimmy Qin. (2023). _New News is Bad News: Information, Expectations, and Financial Markets_. SSRN 4555832. DOI: `10.2139/ssrn.4555832`. Also available as arXiv:2309.05560.

## Author

Diogo Ribeiro — Faculty of Media Arts and Design, Technical University of Porto  
ORCID: 0009-0001-2022-7072

## License

MIT. See `LICENSE`.
