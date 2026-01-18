"""
Benchmarking and evaluation module for compression algorithms.
Separate from core compression logic.
"""

from .evaluator import evaluate_compression, benchmark_thresholds

__all__ = [
    'evaluate_compression',
    'benchmark_thresholds'
]
