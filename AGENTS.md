# AGENTS Instructions

Welcome to the **Entropy News** repository.

This file defines guidelines for future changes. All modifications in the tree from this directory must follow these rules.

## Code Style

- Use Python 3.10 or higher with *type hints* and *docstrings* in all functions.
- Prefer `snake_case` for variables and functions.
- Comments and messages should be brief and, when possible, in English.

## Recommended Features

- The scripts `main.py` and `main_forecast.py` already support `argparse` to allow command-line parameterization.
- GloVe loading automatically detects the embedding dimension (can be overridden with an argument).
- The `EntropyLSTM` class allows configuring `num_layers` and `dropout`.
- Unit tests for `EntropyCalculator` and `NewsModelUpdateCalculator` are located in `tests/`.

Any new feature should remain compatible with these extensions.

## Tests

- When modifying Python code, run:

```bash
  pytest -q
```


- From the repository root and ensure that all tests pass.
- If new behaviors are added, include corresponding tests inside `tests/`.

## Pull Requests

- Commit messages should be concise (e.g., `feat: add cli support`).
- The PR body should briefly describe the changes and note whether the tests passed.
