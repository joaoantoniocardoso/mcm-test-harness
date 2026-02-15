"""Unit tests for ab_harness.metadata module."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from ab_harness.metadata import (
    _run,
    collect_metadata,
    git_branch,
    git_hash,
    git_short_hash,
    save_metadata,
)

# ---------------------------------------------------------------------------
# save_metadata
# ---------------------------------------------------------------------------


class TestSaveMetadata:
    """Tests for save_metadata persistence."""

    @pytest.fixture
    def sample_meta(self) -> dict:
        return {
            "experiment": "test_exp",
            "label": "baseline",
            "timestamp": "2025-01-01T00:00:00+00:00",
            "mcm_git_hash": "abc123",
            "deploy_mode": "local",
        }

    def test_creates_file(self, tmp_path: Path, sample_meta: dict) -> None:
        """save_metadata should write metadata.json to the given directory."""
        result_path = save_metadata(sample_meta, tmp_path)
        assert result_path.exists(), "metadata.json was not created"
        assert result_path.name == "metadata.json"

    def test_valid_json_with_expected_keys(self, tmp_path: Path, sample_meta: dict) -> None:
        """The saved file should be valid JSON containing all input keys."""
        save_metadata(sample_meta, tmp_path)
        data = json.loads((tmp_path / "metadata.json").read_text())

        for key in sample_meta:
            assert key in data, f"Expected key '{key}' missing from saved JSON"
            assert data[key] == sample_meta[key], (
                f"Value mismatch for '{key}': expected {sample_meta[key]!r}, got {data[key]!r}"
            )


# ---------------------------------------------------------------------------
# collect_metadata
# ---------------------------------------------------------------------------


class TestCollectMetadata:
    """Tests for collect_metadata builder."""

    REQUIRED_KEYS = {
        # Package versions
        "harness_version",
        "msprt_version",
        "command_line",
        "experiment",
        "label",
        "timestamp",
        # MCM repo
        "mcm_git_hash",
        "mcm_git_short_hash",
        "mcm_git_branch",
        "mcm_git_diff_stat",
        "mcm_git_diff",
        # Harness repo
        "harness_git_hash",
        "harness_git_short_hash",
        "harness_git_branch",
        "harness_git_dirty",
        # Config
        "deploy_mode",
        "host",
        "port",
        "duration_seconds",
        "repetitions",
        "warmup_seconds",
        "env_overrides",
        "extra_mcm_args",
    }

    def _make_meta(self, **overrides: Any) -> dict:
        """Helper to build metadata with sensible defaults."""
        defaults: dict[str, Any] = dict(
            label="test",
            experiment="unit_test",
            deploy_mode="local",
            duration_seconds=60,
            repetitions=10,
            warmup_seconds=5,
            host="localhost",
            port=8554,
        )
        defaults.update(overrides)
        return collect_metadata(**defaults)

    def test_structure(self) -> None:
        """All expected keys should be present in the returned dict."""
        meta = self._make_meta()

        missing = self.REQUIRED_KEYS - set(meta.keys())
        assert not missing, f"Missing keys in metadata: {missing}"

        # Git fields may be empty strings if not in a proper git repo
        assert isinstance(meta["mcm_git_hash"], str)
        assert isinstance(meta["timestamp"], str)
        assert meta["deploy_mode"] == "local"

    def test_with_extra(self) -> None:
        """Extra fields should be merged into the metadata dict."""
        meta = self._make_meta(extra={"foo": "bar", "baz": 42})

        assert meta["foo"] == "bar", "Extra key 'foo' not found"
        assert meta["baz"] == 42, "Extra key 'baz' not found"

    def test_env_overrides_default(self) -> None:
        """Without env_overrides, the field should be an empty dict."""
        meta = self._make_meta()
        assert meta["env_overrides"] == {}

    def test_env_overrides_passed(self) -> None:
        """env_overrides should be stored as-is."""
        overrides = {"GST_DEBUG": "3", "MCM_LOG": "debug"}
        meta = self._make_meta(env_overrides=overrides)
        assert meta["env_overrides"] == overrides

    def test_no_blueos_versions_in_collect_metadata(self) -> None:
        """BlueOS version fields are injected by the runner, not collect_metadata."""
        meta = self._make_meta(deploy_mode="local")
        assert "blueos_core_version" not in meta
        assert "blueos_bootstrap_version" not in meta

        meta_remote = self._make_meta(deploy_mode="remote")
        assert "blueos_core_version" not in meta_remote
        assert "blueos_bootstrap_version" not in meta_remote


# ---------------------------------------------------------------------------
# _run helper
# ---------------------------------------------------------------------------


class TestRunCommand:
    """Tests for the _run subprocess helper."""

    def test_successful_command(self) -> None:
        """A simple command should return its stdout."""
        result = _run(["echo", "hello"], cwd=Path("/tmp"))
        assert result == "hello"

    def test_invalid_command_returns_empty(self) -> None:
        """An invalid/nonexistent command should return an empty string."""
        result = _run(
            ["this_command_does_not_exist_xyz_12345"],
            cwd=Path("/tmp"),
        )
        assert result == "", f"Expected empty string for invalid command, got {result!r}"

    def test_timeout_returns_empty(self) -> None:
        """A command that exceeds the timeout should return empty string."""
        result = _run(["sleep", "60"], cwd=Path("/tmp"), timeout=0.01)
        assert result == "", "Timed-out command should return empty string"


# ---------------------------------------------------------------------------
# Git helper functions
# ---------------------------------------------------------------------------


class TestGitFunctions:
    """Tests for git_hash, git_short_hash, git_branch."""

    @pytest.mark.parametrize(
        "fn",
        [git_hash, git_short_hash, git_branch],
        ids=["git_hash", "git_short_hash", "git_branch"],
    )
    def test_returns_string(self, fn) -> None:
        """All git helper functions should return a string (possibly empty)."""
        result = fn()
        assert isinstance(result, str), (
            f"{fn.__name__} should return str, got {type(result).__name__}"
        )
