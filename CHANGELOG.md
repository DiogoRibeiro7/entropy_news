# Changelog

## Unreleased
- Added correlation visualisations, summary exports, and CSV upload controls to the Streamlit dashboard
- Introduced Docker deployment assets and documentation covering quantisation/ONNX workflows
- Created integration/performance test suites with CI automation and pytest markers
- Extended distributed utilities with throughput monitoring, checkpoint rotation, and stress testing helpers
- Consolidated CLI configuration resolution via reusable helpers shared across training and rolling pipelines

## 0.1.0 - 2025-07-08
- Initial release.

## 0.1.1 - 2025-07-09
- Added a GitHub Actions workflow for automatic PyPI publishing
- Improved documentation on obtaining datasets legally
- Harmonised docstrings and expanded tests for training and forecasting

## 0.1.2 - 2025-07-09
- Added unit tests for rolling-text helpers
- Verified that `Trainer.train` updates model parameters

## 0.1.3 - 2025-07-09
- Enabled coverage reporting in CI
- Cleaned up a duplicate import in logger tests

