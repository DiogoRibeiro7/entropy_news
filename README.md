# Entropy News

Professional implementation of the methodology from the paper **"New News is Bad News"** (Paul Glasserman, Harry Mamaysky, and Jimmy Qin, 2023).

This project computes the **entropy of financial news** using a **Recurrent Neural Network (LSTM)** and uses that signal to predict market returns.

## 📦 Project Structure

```
entropy_news/
├── data/
│   ├── preprocessor.py        # Text cleaning, tokenization, vocabulary creation, GloVe loading, vocab save/load
│   └── dataset.py             # PyTorch Dataset with automatic padding for sequence training
│
├── model/
│   ├── lstm_entropy.py        # LSTM architecture for next-token prediction
│   └── trainer.py             # Training and fine-tuning with Adam optimizer and CrossEntropyLoss
│
├── evaluation/
│   ├── entropy_calculator.py  # Average entropy computation on new data
│   └── news_model_update.py   # Decomposition of entropy: ENT, ENT_news, ENT_model
│
├── utils/
│   └── metrics.py             # Auxiliary functions (e.g., perplexity computation)
│
├── main.py                    # Main script for training the initial model
├── main_forecast.py           # Script for predicting ENT, ENT_news, ENT_model, and exporting to CSV
│
├── output/
│   └── (generated models and results)
│
├── pyproject.toml             # Project dependencies managed by Poetry
└── README.md                  # Documentation and usage instructions
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
### 4. Reuse Vocabulary
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
pytest --cov=entropy_news -q
```
Continuous integration runs the same command, ensuring coverage is tracked for
every pull request. Coverage statistics are displayed in the GitHub Actions
summary for convenient review.

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
