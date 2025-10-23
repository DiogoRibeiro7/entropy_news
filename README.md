# Entropy News

Professional implementation of the methodology from the paper **"New News is Bad News"** (Paul Glasserman, Harry Mamaysky, and Jimmy Qin, 2023).

This project computes the **entropy of financial news** using a **Recurrent Neural Network (LSTM)** and uses that signal to predict market returns.

Full documentation is available at [entropy-news.readthedocs.io](https://entropy-news.readthedocs.io/).

## 📖 Documentation Highlights

- **API Reference** – Auto-generated module documentation lives under
  `docs/api/index.md` with pages for core CLIs, model helpers, utilities, and the
  causal research toolkit.
- **Tutorials** – Step-by-step walkthroughs in `docs/tutorials/` cover training,
  forecasting insights, dashboard storytelling, orchestration, and inference
  delivery scenarios.
- **Scenario Playbooks** – Operational guides in `docs/playbooks/` map
  enterprise rollout, incident response, causal reviews, and research registry
  processes to the underlying runbooks and APIs.
- **Multimedia Hub** – Production, accessibility, and storyboard checklists in
  `docs/media/` ensure Release 1.0 includes captioned videos and transcripts.

## 📦 Project Structure

```
.
├── docs/                      # Deployment, testing, performance, and data-format guides
├── docker/                    # Dockerfile and compose profile for containerised runs
├── entropy_news/
│   ├── dashboard/
│   │   └── app.py             # Streamlit analytics with correlation views and exports
│   ├── data/
│   │   ├── dataset.py         # In-memory datasets with automatic padding
│   │   ├── preprocessor.py    # Text cleaning, tokenisation, GloVe loading, vocab save/load
│   │   └── streaming_dataset.py
│   │                          # Incremental dataset indexing utilities
│   ├── evaluation/
│   │   ├── entropy_calculator.py
│   │   ├── multimodal_metrics.py
│   │   └── news_model_update.py
│   │                          # Metrics for decomposition and multimodal scoring
│   ├── model/
│   │   ├── config.py          # Configuration dataclasses with validation helpers
│   │   ├── distributed.py     # Monitoring, stress, and checkpoint helpers for distributed runs
│   │   ├── factory.py         # Unified factory for LSTM, attention, and transformer models
│   │   ├── fusion.py          # Configurable multimodal fusion layers
│   │   ├── inference.py       # Quantisation and ONNX export utilities
│   │   ├── orchestration.py   # Enterprise scheduler, launch plans, and health endpoints
│   │   ├── lstm_attention.py
│   │   ├── lstm_entropy.py
│   │   └── transformer_entropy.py
│   ├── research/
│   │   └── causal/             # Causal data assembly, estimators, and reporting helpers
│   ├── utils/
│   │   ├── cli.py             # Shared CLI configuration and checkpoint loading helpers
│   │   ├── device.py          # CUDA stream / autocast context managers
│   │   ├── migration.py       # Legacy argument & config migration utilities
│   │   └── ...
│   ├── migrate_config.py      # CLI entry-point for migrating legacy configs
│   ├── main.py                # Training CLI entry-point
│   ├── main_evaluate.py       # Evaluation CLI entry-point
│   ├── main_forecast.py       # Forecast CLI entry-point
│   └── rolling_train_forecast.py
├── notebooks/                 # Research collateral (e.g., correlation analysis)
├── monitoring/                # Prometheus, Grafana, and alerting templates
├── tests/                     # Unit, integration, and performance suites
└── pyproject.toml             # Poetry-managed dependencies
```

## 🏗 Architecture Overview

```
News Texts -> TextPreprocessor -> Dataset -> ModelFactory -> Trainer -> Metrics
```
The pipeline cleans and tokenizes raw texts, builds datasets, instantiates a
model through the configuration-driven `ModelFactory`, trains it with the
`Trainer`, and finally evaluates entropy-based metrics.

## ⚙️ Installation

```bash
git clone <repository>
cd entropy_news
poetry install
```

Also download GloVe:
```bash
wget http://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip
```

## 🧭 Quick Start Tutorial

1. **Prepare Data** – Place your training news in `data/news_train.txt` and
   download GloVe embeddings as shown above.
2. **Train** – Run `entropy-news-train` to fit an LSTM on the news corpus.
3. **Forecast** – Use `entropy-news-forecast` to compute `ENT`, `ENT_news`, and
   `ENT_model` for new articles.
4. **Evaluate** – Run `entropy-news-eval` to measure perplexity on held‑out
   data.
5. **Iterate** – Adjust hyperparameters or switch architectures via
   configuration files.

## 📊 Interactive Dashboard

Launch the Streamlit dashboard to explore rolling forecast outputs with
correlation heatmaps, summary statistics, and downloadable reports:

```bash
streamlit run entropy_news/dashboard/app.py
```

Filter by month, inspect correlations between `entropy`, `entropy_news`, and
`entropy_model`, swap between CSV files via the sidebar (or upload new results),
and export both the filtered dataset and the generated summary report for
stakeholder distribution.


## 🔍 Causal Research Toolkit

The `entropy_news.research.causal` package bundles everything required to evaluate
news-driven interventions:

1. **Assemble panels** – Merge entropy metrics with market data using
   `assemble_causal_panel` and generate propensity inputs with
   `build_propensity_features`.
2. **Estimate effects** – Apply `difference_in_differences`,
   `two_stage_least_squares`, or `synthetic_control` to quantify counterfactual
   scenarios. Each helper returns a dataclass with diagnostics (confidence
   intervals, F-statistics, weight vectors).
3. **Report findings** – Produce tabular summaries with
   `build_summary_table`, craft narratives via `format_policy_narrative`, and
   forward aligned time series to dashboards through
   `prepare_counterfactual_series`.

See `docs/causal_methodology.md` and the accompanying notebooks for example
workflows and methodological guardrails.

## 📚 Data
The training news files referenced in the examples are not distributed with this
repository. Ensure you have permission to use any dataset you supply. The
[GloVe embeddings](https://nlp.stanford.edu/projects/glove/) are available from
the Stanford NLP Group and released under the
[Public Domain Dedication and License](https://nlp.stanford.edu/data/).

Example sources of open data include Kaggle's
["Financial News Dataset"](https://www.kaggle.com/aaron7sun/stocknews). Download
the archive by first accepting its license on Kaggle and then running:

```bash
kaggle datasets download -d aaron7sun/stocknews -p data/
unzip data/stocknews.zip -d data/
```

After extracting, provide the file path via ``--train-data`` when running
``entropy-news-train``.

Any dataset you provide for training must be legally obtained and its license
terms respected.

## 🚀 How to Use

### 1. Train the Model
You can invoke the training script directly or via the provided console command:
```bash
entropy-news-train
```
- Trains the LSTM using training news (`data/news_train.txt`).
- Saves the trained model in `output/model_final.pth`.
  You can customise the inputs and hyperparameters:
```bash
entropy-news-train --train-data my_train.txt --epochs 10 --batch-size 64 \
                   --learning-rate 0.0005
```

You can supply a JSON configuration via `--model-config` to reuse
Transformer or attention hyperparameters, and the resolved settings are saved to
`--config-out` for evaluation and forecasting runs.

### 2. Forecast Entropies
```bash
entropy-news-forecast
```
- Calculates `ENT`, `ENT_news`, `ENT_model` using new news (`data/news_new.txt`).
- Exports results to `output/forecast_results.csv`.
  Example with options:
```bash
entropy-news-forecast --new-data other.txt --output-csv results.csv \
                      --fine-tune-epochs 8 --batch-size 2
```

### 3. Evaluate a Saved Model
```bash
entropy-news-eval
```
- Computes the average entropy and perplexity on `data/news_new.txt`.
- Optionally writes the values to ``output/metrics.csv``.
  Example with options:
```bash
entropy-news-eval --data other.txt --batch-size 4 --output-csv metrics.csv
```
- For legacy checkpoints saved with ``torch.save(model)``, re-run with
  ``--allow-unsafe-load`` after verifying the file is trusted; this flag
  restores the older pickle-based loading path. As we migrate remaining
  artifacts to safe formats the fallback will be disabled in a future release,
  so plan to refresh any custom checkpoints accordingly.

### 4. Orchestrate Multi-Node Training

Render distributed launch plans (and optionally execute them) using the
enterprise orchestrator CLI:

```bash
# Preview the launch plan without executing the job
entropy-news-orchestrate --topology configs/cluster.json

# Execute the plan with the built-in launcher and wait for completion
entropy-news-orchestrate --topology configs/cluster.json --launch
```

Each launched rank receives `MASTER_ADDR`, `MASTER_PORT`, `RANK`, and
`WORLD_SIZE` in its environment so PyTorch's distributed runtime is correctly
configured. Press `Ctrl+C` to interrupt running jobs; the orchestrator will
terminate the spawned processes and exit with status 130. Add `--health-server`
to expose the JSON liveness endpoint while the plan is executing.

### 5. Reuse Vocabulary
You can save the built vocabulary for later runs and reload it instead of
recomputing every time:

```python
from entropy_news.data import TextPreprocessor

preprocessor = TextPreprocessor()
preprocessor.build_vocab(train_texts)
preprocessor.save_vocab("output/vocab.json")

# Later
preprocessor.load_vocab("output/vocab.json")
```


## 🔃 Migrating Legacy Configurations

Use `python -m entropy_news.migrate_config <legacy.json>` to convert older
training metadata into the new `ModelConfig` format. The command writes the
resolved configuration to `output/model_config.json`, which can then be loaded
by the training, evaluation, forecasting, and rolling CLIs via `--model-config`.

## 🚢 Deployment

A production-focused Dockerfile and compose profile live in the repository. The
[`deployment` guide](docs/deployment.md) explains how to build the image, export
quantised/ONNX models with the inference helpers, and run the CLIs inside the
containerised environment.

## 🧪 Testing

Unit tests remain the default entry point:

```bash
pytest -q
```

Integration and performance suites are tagged and can be executed separately:

```bash
pytest -m integration -q
pytest -m performance -q
pytest -m stress -q
```

See [`docs/testing_strategy.md`](docs/testing_strategy.md) for the full
automation plan and CI matrix.

### 🔐 Checkpoint Safety

All CLIs load checkpoints with PyTorch's ``weights_only=True`` safeguard by
default. This prevents arbitrary pickle execution when consuming untrusted
artifacts. When you must load legacy checkpoints saved via ``torch.save(model)``,
explicitly opt-in with ``--allow-unsafe-load`` (or set the
``ENTROPY_NEWS_ALLOW_UNSAFE_LOAD=1`` environment variable) and ensure that the
checkpoint originates from a trusted source. The application logs a warning when
falling back to the unsafe path so operators can audit usage.

## 📈 Rolling Window Pipeline (Example)

To process multiple months:

1. For each month `t`:
   - Use data from `[t-6, t-1]` to train (6-month rolling window).
   - Predict entropy on month `t`.
   - Fine-tune the model including month `t`.
2. Store `ENT`, `ENT_news`, `ENT_model` month by month.

This can be automated into a single pipeline.

Run it from the command line with ``entropy-news-rolling``:

```bash
entropy-news-rolling 2023-01 2023-02 2023-03 --base-data-dir data/ --output-dir output/
```

The ``Trainer`` class displays a progress bar via ``tqdm`` and supports optional
early stopping when a validation set is provided:

```python
trainer.train(train_dataset, epochs=100, val_dataset=val_dataset,
              early_stopping=True, patience=3)
```

You can also compute perplexity directly using ``EntropyCalculator``:

```python
calculator = EntropyCalculator(model)
perplex = calculator.compute_perplexity(dataset)
```

## 🧠 Model Selection Guidelines

- **LSTM** – Good baseline for smaller datasets or when GPU resources are
  limited.
- **LSTM with Attention** – Keeps recurrent inductive bias while capturing
  longer‑range dependencies. Use when sequence order matters but context is
  broad.
- **Transformer** – Best for large datasets and when parallel training is
  desired. Requires head dimensions that divide the hidden size evenly.

Choose the architecture by setting `architecture` in `ModelConfig` and passing
the configuration to `ModelFactory`.

## 🛠 Command-line Interface

The project exposes convenient console entry points:

| Command | Purpose |
| --- | --- |
| `entropy-news-train` | Train a model using news text |
| `entropy-news-forecast` | Forecast entropy measures for new data |
| `entropy-news-eval` | Evaluate a saved model |
| `entropy-news-rolling` | Run a rolling‑window training/forecast pipeline |

Each command accepts `--help` to list all options. Key arguments include
`--train-data` for data paths, `--epochs` and `--batch-size` for training
control, and `--model-out`/`--output-csv` for result locations.

## 📝 Logging

All scripts configure logging with ``utils.setup_logger``. The helper will
ignore duplicate handlers for the same file, but it is best to call it once per
process and reuse the returned logger throughout the script:

```python
from entropy_news.utils import setup_logger

logger = setup_logger("train_logger", "logs/train.log")  # or ``None`` to disable file logging

# reuse ``logger`` across modules
```

Calling ``setup_logger`` multiple times with the same arguments is harmless but
may incur a small overhead.

## ✅ Testing
The project includes a suite of unit tests that can be executed with ``pytest``.
Several tests rely on ``torch`` and ``numpy``. If those packages are not
installed, they will be skipped. Install them to run the full suite and gather
coverage:

```bash
poetry install --with torch,numpy
pytest --cov=entropy_news --cov-fail-under=95 -q
```
Continuous integration runs this command on Linux x86_64 alongside additional
Linux ARM64 and Windows jobs so coverage gating (set to 95%) and parity checks
remain visible in GitHub Actions summary reports.

## 📦 Releasing
This project is packaged with [Poetry](https://python-poetry.org/) and follows
[Semantic Versioning](https://semver.org/).

1. Bump the version number using ``poetry version <patch|minor|major>``.
2. Build the distribution with ``poetry build``.
3. Publish to PyPI via ``poetry publish`` (configure your credentials first).
4. Create a Git tag for the new version and push it along with ``main``.
5. A GitHub Actions workflow builds and uploads the package whenever a tag
   matching ``v*`` is pushed. Ensure the ``PYPI_TOKEN`` secret is configured on
   your repository.

## 📚 Reference
- Paul Glasserman, Harry Mamaysky, and Jimmy Qin. (2023). *New News is Bad News: Information, Expectations, and Financial Markets*. [SSRN 4555832](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4555832)

## Author
- **Diogo Ribeiro** (DiogoRibeiro7)
- Affiliation: ESMAD - Instituto Politécnico do Porto
- ORCID: [0009-0001-2022-7072](https://orcid.org/0009-0001-2022-7072)
- Professional email: dfr@esmad.ipp.pt

## License
This project is released under the MIT License. See [LICENSE](LICENSE) for details.
