"""Sphinx configuration for the Entropy News documentation."""

from __future__ import annotations

import importlib.metadata as metadata
import os
import sys

# Add project root to sys.path to allow autodoc to locate modules.
sys.path.insert(0, os.path.abspath(".."))

project = "Entropy News"
author = "Diogo Ribeiro"

# Attempt to obtain the installed package version; fallback to default.
try:
    release = metadata.version("entropy-news")
except metadata.PackageNotFoundError:  # pragma: no cover - fallback when package isn't installed
    release = "0.1.0"

extensions: list[str] = [
    "myst_parser",
    "sphinx.ext.autodoc",
]

templates_path = ["_templates"]
exclude_patterns: list[str] = []

# Use the Read the Docs theme if available.
html_theme = "sphinx_rtd_theme"

