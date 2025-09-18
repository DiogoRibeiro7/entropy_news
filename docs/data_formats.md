# Standardized Data Formats

Entropy News expects newline-delimited UTF-8 text files for model training and
forecasting. Each line corresponds to an individual article. The rolling
pipeline consumes files named `news_<YYYY-MM>.txt` and stores the resulting
metrics in CSV format with the following columns:

| Column | Description |
| ------ | ----------- |
| `month` | Reporting month in `YYYY-MM` format |
| `entropy` | Average entropy on the evaluation month |
| `entropy_news` | News-driven contribution |
| `entropy_model` | Model update contribution |

Market connectors produce pandas DataFrames with a standard schema:

| Column | Type | Notes |
| ------ | ---- | ----- |
| `date` | datetime64 | Trading day |
| `open` | float | Opening price |
| `high` | float | Daily high |
| `low` | float | Daily low |
| `close` | float | Adjusted close |
| `volume` | int | Trading volume |

## API Usage

* Yahoo Finance loader: `load_yahoo_data("AAPL", start="2022-01-01", end="2022-12-31")`
* Alpha Vantage loader: supply your API key via the `ALPHA_VANTAGE_API_KEY`
  environment variable before invoking `load_alpha_vantage_data`.
* CSV loader: `load_csv_data(Path("market.csv"))` expects headers matching the
  schema above.

All loaders return data sorted by ascending date and can be combined with the
rolling pipeline for joint modelling with news features.
