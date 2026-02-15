"""Mixture Sequential Probability Ratio Test (mSPRT).

A lightweight, dependency-minimal implementation of the mSPRT for
sequential A/B testing with always-valid inference.

Based on Johari, Pekelis & Walsh (2017/2022).
"""

from importlib.metadata import PackageNotFoundError, version

from msprt.core import MsprtResult, compute_tau, msprt_statistic, msprt_test

try:
    __version__ = version("msprt")
except PackageNotFoundError:
    __version__ = "0.0.0-unknown"

__all__ = [
    "MsprtResult",
    "compute_tau",
    "msprt_statistic",
    "msprt_test",
]
