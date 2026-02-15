# A/B Test Harness for MCM Pipeline Analysis Stats

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("ab-harness")
except PackageNotFoundError:
    __version__ = "0.0.0-unknown"
