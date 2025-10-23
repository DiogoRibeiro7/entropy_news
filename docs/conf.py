"""Sphinx configuration for the Entropy News documentation."""

from __future__ import annotations

import importlib.metadata as metadata
import os
import sys
from datetime import datetime

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
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
]

autosummary_generate = True
autodoc_typehints = "description"
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "linkify",
    "smartquotes",
]

myst_heading_anchors = 3

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", {}),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable", {}),
    "numpy": ("https://numpy.org/doc/stable/", {}),
    "torch": ("https://pytorch.org/docs/stable", {}),
}

templates_path = ["_templates"]
exclude_patterns: list[str] = []

# Use the Read the Docs theme if available.
html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "collapse_navigation": False,
    "style_external_links": True,
    "navigation_depth": 3,
}

html_static_path = ["_static"]

html_context = {
    "last_updated": datetime.utcnow().strftime("%Y-%m-%d"),
}

