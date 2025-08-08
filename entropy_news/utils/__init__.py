"""Utility helpers for logging and file I/O."""

from .logger import setup_logger
from .metrics import perplexity
from .io import load_texts, save_texts
from .device import autocast, cuda_stream, get_cuda_stream, get_device
from .correlation import correlation, plot_correlation, rolling_correlation
from .memory import measure_peak_memory
from .attention import plot_attention_weights

__all__ = [
    "setup_logger",
    "perplexity",
    "load_texts",
    "save_texts",
    "get_device",
    "get_cuda_stream",
    "cuda_stream",
    "autocast",
    "correlation",
    "plot_correlation",
    "rolling_correlation",
    "plot_attention_weights",
    "measure_peak_memory",
]
