# README.md for entropy_news

# Entropy News

Professional implementation of the methodology from the paper **"New News is Bad News"** (Cieslak et al., 2023).

This project computes the **entropy of financial news** using a **Recurrent Neural Network (LSTM)** and uses that signal to predict market returns.

## 📦 Project Structure

```
entropy_news/
├── data/
│   ├── preprocessor.py        # Text cleaning, tokenization, vocabulary creation, GloVe loading
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
```bash
python entropy_news/main.py
```
- Trains the LSTM using training news (`data/news_train.txt`).
- Saves the trained model in `output/model_final.pth`.

### 2. Forecast Entropies
```bash
python main_forecast.py
```
- Calculates `ENT`, `ENT_news`, `ENT_model` using new news (`data/news_new.txt`).
- Exports results to `output/forecast_results.csv`.

## 📈 Rolling Window Pipeline (Example)

To process multiple months:

1. For each month `t`:
   - Use data from `[t-6, t-1]` to train (6-month rolling window).
   - Predict entropy on month `t`.
   - Fine-tune the model including month `t`.
2. Store `ENT`, `ENT_news`, `ENT_model` month by month.

This can be automated into a single pipeline.

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
