# Changelog

## Unreleased

## 0.2.2 - 2026-08-31
- Rewrote the Zenodo record as scholarly research-software metadata rather than repository boilerplate.
- Adopted the descriptive title `Entropy News: Reproducible Financial-News Entropy Estimation and Decomposition` across Zenodo and citation metadata.
- Replaced the generic deposit description with a methodological abstract covering the rolling entropy estimator, `ENT_NEWS` / `ENT_MODEL` decomposition, reported LSTM specification, and the boundary between methodological reproduction and empirical Reuters replication.
- Tightened discovery keywords around financial-news entropy, rolling estimation, language models, and reproducible computational research.
- No estimator, model, data, dependency, provenance, or scientific-result logic changed.

## 0.2.1 - 2026-08-31
- Added authoritative `.zenodo.json` metadata for Zenodo GitHub archiving.
- Aligned Zenodo, citation, and package metadata at version 0.2.1.
- Added the source working paper as an `isDerivedFrom` related identifier without predeclaring a Zenodo DOI.
- No estimator, model, data, dependency, provenance, or scientific-result logic changed.

## 0.2.0 - 2026-08-31
- Added the audited strict paper-reproduction path for `ENT`, `ENT_NEWS`, and `ENT_MODEL`, including the 12-month lagged-model decomposition.
- Matched the reported LSTM architecture with 100-dimensional inputs, 16 hidden units, one bias vector per gate block, and 177,488 trainable parameters for the default model.
- Standardised the strict predictive vocabulary at 10,000 classes total including `UNK`, with padding outside the softmax.
- Corrected first-word entropy scoring to use the unconditional distribution at the zero initial hidden state, with no synthetic recurrent transition before `w_1`.
- Added whole-requested-corpus vocabulary construction, explicit `UNK` embedding provenance, and strict vocabulary-cardinality validation.
- Added methodological-reproduction and empirical-Reuters corpus contracts while avoiding software-only claims of proprietary provenance certification.
- Added byte-level input/output provenance, Git revision, environment, architecture, and vocabulary metadata to the paper run manifest.
- Committed a Poetry-generated lockfile and aligned CI on Python 3.11, Poetry 1.7.1, `poetry check --lock`, repository coverage, and a dedicated paper-reproduction coverage gate.
- Updated public documentation to distinguish causal rolling model weights from the paper-style future-informed whole-corpus vocabulary space.
- Updated citation metadata and current institutional affiliation.

## 0.1.3 - 2025-07-09
- Enabled coverage reporting in CI
- Cleaned up a duplicate import in logger tests

## 0.1.2 - 2025-07-09
- Added unit tests for rolling-text helpers
- Verified that `Trainer.train` updates model parameters

## 0.1.1 - 2025-07-09
- Added a GitHub Actions workflow for automatic PyPI publishing
- Improved documentation on obtaining datasets legally
- Harmonised docstrings and expanded tests for training and forecasting

## 0.1.0 - 2025-07-08
- Initial release.

### Earlier unreleased engineering work
- Added correlation visualisations, summary exports, and CSV upload controls to the Streamlit dashboard
- Introduced Docker deployment assets and documentation covering quantisation/ONNX workflows
- Created integration/performance test suites with CI automation and pytest markers
- Extended distributed utilities with throughput monitoring, checkpoint rotation, and stress testing helpers
- Consolidated CLI configuration resolution via reusable helpers shared across training and rolling pipelines
