# Entropy News

Professional implementation of the methodology from the paper **"New News is Bad News"** (Cieslak et al., 2023).

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

### 3. Reuse Vocabulary
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

### 3. Reuse Vocabulary
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

## 📝 Logging

All scripts configure logging with ``utils.setup_logger``. The helper will
ignore duplicate handlers for the same file, but it is best to call it once per
process and reuse the returned logger throughout the script:

```python
from entropy_news.utils import setup_logger

logger = setup_logger("train_logger", "logs/train.log")

# reuse ``logger`` across modules
```

Calling ``setup_logger`` multiple times with the same arguments is harmless but
may incur a small overhead.

## 📚 Reference
- Cieslak, L., Lussange, J., & Thesmar, D. (2023). *New News is Bad News: Information, Expectations, and Financial Markets*. [arXiv:2309.05560](https://arxiv.org/abs/2309.05560)
