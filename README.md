# Entropy News

Python research software for reproducing and extending the financial-news entropy methodology in **Glasserman, Mamaysky, and Qin, _New News is Bad News_**.

The repository separates two distinct layers:

1. **Paper reproduction core** — a rolling implementation of the paper's monthly entropy measure and its `ENT_NEWS` / `ENT_MODEL` decomposition.
2. **Research and engineering extensions** — alternative architectures, one-month model-update diagnostics, market-data tooling, causal analysis, dashboards, distributed training, monitoring, and deployment support.

That distinction matters: the paper's `ENT` is a 12-month change in monthly article entropy, not a one-month cross-entropy level.

## Paper reproduction

The reproduction path implements

\[
A_t = m_{[t-6,t-1]}(t),\qquad
B_t = m_{[t-18,t-13]}(t-12),\qquad
C_t = m_{[t-18,t-13]}(t),
\]

with

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

The strict paper path currently enforces the following contract:

- month `t` is **never used to fit the model weights that score month `t`**;
- the initial language model is trained on six prior months;
- subsequent monthly updates warm-start the previous model and use all articles from `t-1`, then exponentially downsampled articles from `t-2` through `t-6`;
- complete articles are used rather than truncating each article to its first sequence;
- recurrent state is carried across evaluation chunks of the same article and reset between articles;
- the first observed word is scored from the unconditional output distribution at the zero initial hidden state, with no synthetic recurrent transition before `w_1`;
- monthly entropy is the **equal-weighted mean of article entropies**, so long articles do not receive larger weights merely because they contain more tokens;
- the 12-month lagged model is retained so `ENT_NEWS` and `ENT_MODEL` use the paper's decomposition rather than two entropy levels from the same month;
- requested months must be strictly consecutive calendar months;
- the paper path requires an explicit GloVe file rather than silently falling back to random embeddings;
- following the paper's whole-corpus vocabulary definition, vocabulary membership is selected from the **whole requested corpus** before rolling model training;
- the exact paper LSTM uses 100-dimensional frozen embeddings, hidden dimension 16, one bias vector per gate block, and **177,488 trainable parameters** at the default 10,000-class specification;
- the strict architecture uses **10,000 predictive classes total**, including `UNK`; padding is a separate zero-vector input row and is not part of the predictive softmax.

The last two points involve an important distinction between reported facts and implementation inference. The paper reports a 10,000-unit output layer and maps out-of-vocabulary words to `UNK`. The strict implementation therefore uses 10,000 predictive classes total, with `UNK` occupying one class and the remaining classes lexical. This choice is recorded explicitly rather than presented as wording stated verbatim by the paper.

The paper also does not specify the embedding vector assigned to `UNK`. The strict path therefore fixes an explicit implementation convention: a deterministic seeded `N(0,1)` vector. The convention, seed, paper-specified flag (`false`), and SHA-256 of the exact vector are stored in `paper_vocabulary.json`. Padding remains a separate all-zero row.

The whole-corpus vocabulary choice means the rolling model weights remain causal, but vocabulary membership can reflect later corpus observations. This is a paper-reproduction choice, not a strictly real-time vocabulary construction rule.

The scientific contract lives primarily in:

- `entropy_news/paper_reproduction.py`
- `entropy_news/paper_architecture.py`
- `entropy_news/model/paper_lstm.py`
- `entropy_news/paper_rolling.py`
- `entropy_news/corpus_contract.py`
- `tests/test_paper_reproduction.py`
- `tests/test_paper_unk_embedding.py`

### Corpus classification

The software distinguishes two run classes:

- `methodological_reproduction` — the default when no external corpus manifest is supplied;
- `empirical_reuters_replication` — available only when the supplied corpus contract declares the Reuters-specific source/filtering conditions required by the strict validator.

The validator checks internal consistency of the declaration. It does **not** certify the provenance of proprietary Reuters bytes, and the run manifest records `provenance_certified_by_software: false` accordingly.

### Run the paper path

Monthly files are expected as:

```text
data/news_YYYY-MM.txt
```

with one article per non-empty line. The month arguments must form an unbroken monthly sequence.

Install the strict dependencies and run:

```bash
poetry install --with torch --with dev

entropy-news-paper \
  2020-01 2020-02 2020-03 ... 2022-12 \
  --base-data-dir data/ \
  --glove-path /path/to/glove.6B.100d.txt \
  --output-dir output/paper_reproduction
```

An optional `--corpus-manifest` JSON file can be supplied to bind an explicit corpus contract to the run.

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

`paper_vocabulary.json` stores the exact token-to-index mapping and strict vocabulary metadata, including the auditable `UNK` embedding convention. `paper_run_manifest.json` records the execution environment, Git revision, learning rate, protocol, corpus classification, vocabulary scope and size, every requested monthly input file, raw and qualifying article counts, SHA-256 hashes and byte sizes for all monthly inputs and the GloVe file, plus SHA-256 hashes of the result, protocol and vocabulary outputs.

## One-month diagnostics are not paper ENT

`entropy-news-forecast` remains useful for comparing a baseline model and an updated model on a single dataset. It reports:

```text
baseline_entropy
updated_entropy
model_update_delta
```

These are entropy-level diagnostics. They are deliberately **not** named `ENT`, `ENT_NEWS`, or `ENT_MODEL`, because those paper quantities require the 12-month lagged design above.

Likewise, `entropy-news-rolling` is retained as a generic rolling/fine-tuning workflow for experiments and operational use. Use `entropy-news-paper` when the goal is scientific reproduction of the Glasserman–Mamaysky–Qin method.

## Model and data layer

The repository includes:

- the exact strict paper LSTM plus configurable language models for extensions;
- optional frozen GloVe embeddings outside the strict paper path;
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
poetry install --with torch --with dev
```

The repository commits `poetry.lock`. CI runs on Python 3.11 with Poetry 1.7.1, executes `poetry check --lock`, and installs from the committed lockfile. Changes to `pyproject.toml` that are not accompanied by a current lockfile fail CI.

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
- exact 177,488-parameter paper architecture;
- 10,000 predictive classes with `UNK` inside the softmax and padding outside it;
- unconditional first-word scoring;
- complete-article chunk coverage;
- deterministic exponential historical sampling;
- equal weighting across articles;
- mandatory GloVe for the strict paper path;
- deterministic and provenance-bound `UNK` embedding convention;
- strict vocabulary cardinality;
- consecutive-month validation;
- whole-requested-corpus vocabulary construction;
- corpus-contract validation;
- byte-level input/output provenance in the run manifest;
- repository-wide and paper-core coverage gates.

## Data and reproducibility

The original Reuters corpus used in the paper is not distributed with this repository. Users must supply a legally obtained corpus with monthly article boundaries.

A run without an externally supplied qualifying Reuters corpus contract is classified as a **methodological reproduction**, not as an empirical replication of the paper's Reuters results. Even with an `empirical_reuters_replication` declaration, the software validates the contract but does not independently certify proprietary source provenance.

Every paper run with an output directory records the protocol, vocabulary and provenance manifest. Keep these files with any manuscript-facing result so the exact data files, GloVe bytes, filtering counts, random seed, vocabulary, implementation conventions, software revision and generated outputs can be audited later.

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

Paul Glasserman, Harry Mamaysky, and Jimmy Qin. _New News is Bad News: Information, Expectations, and Financial Markets_. Working paper. SSRN 4555832. DOI: `10.2139/ssrn.4555832`. Also available as arXiv:2309.05560.

## Author

Diogo Ribeiro — Faculty of Media Arts and Design, Technical University of Porto  
ORCID: 0009-0001-2022-7072

## License

MIT. See `LICENSE`.
