"""Capture and store run metadata: git state, container version, config."""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from msprt import __version__ as msprt_version

from ab_harness import __version__ as harness_version
from ab_harness.config import HARNESS_ROOT, MCM_REPO_ROOT

log = logging.getLogger(__name__)


def _run(cmd: list[str], cwd: Path | None = None, timeout: float = 15) -> str:
    """Run a command and return stripped stdout, or empty string on failure."""
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=cwd or MCM_REPO_ROOT,
            timeout=timeout,
        )
        return result.stdout.strip()
    except Exception as exc:
        log.warning("Command %s failed: %s", cmd, exc)
        return ""


# --- MCM repo git info ---


def git_hash() -> str:
    return _run(["git", "rev-parse", "HEAD"])


def git_short_hash() -> str:
    return _run(["git", "rev-parse", "--short", "HEAD"])


def git_branch() -> str:
    return _run(["git", "branch", "--show-current"])


def git_diff() -> str:
    return _run(["git", "diff", "HEAD"])


def git_diff_stat() -> str:
    return _run(["git", "diff", "--stat", "HEAD"])


# --- Harness repo git info ---


def harness_git_hash() -> str:
    return _run(["git", "rev-parse", "HEAD"], cwd=HARNESS_ROOT)


def harness_git_short_hash() -> str:
    return _run(["git", "rev-parse", "--short", "HEAD"], cwd=HARNESS_ROOT)


def harness_git_branch() -> str:
    return _run(["git", "branch", "--show-current"], cwd=HARNESS_ROOT)


def harness_git_dirty() -> bool:
    """Return True if the harness repo has uncommitted changes."""
    status = _run(["git", "status", "--porcelain"], cwd=HARNESS_ROOT)
    return bool(status)


def collect_metadata(
    *,
    label: str,
    experiment: str,
    deploy_mode: str,
    duration_seconds: int,
    repetitions: int,
    warmup_seconds: int,
    host: str,
    port: int,
    env_overrides: dict[str, str] | None = None,
    client_config: Any = None,
    extra_args: list[str] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the full metadata dict for a run."""
    from ab_harness.config import client_config_to_dict

    # Capture git state once to avoid redundant subprocess calls
    _mcm_hash = git_hash()
    _mcm_short = git_short_hash()
    _mcm_branch = git_branch()

    meta: dict[str, Any] = {
        # Package versions
        "harness_version": harness_version,
        "msprt_version": msprt_version,
        # Invocation
        "command_line": sys.argv,
        # MCM repo
        "mcm_git_hash": _mcm_hash,
        "mcm_git_short_hash": _mcm_short,
        "mcm_git_branch": _mcm_branch,
        "mcm_git_diff_stat": git_diff_stat(),
        "mcm_git_diff": git_diff(),
        # Harness repo
        "harness_git_hash": harness_git_hash(),
        "harness_git_short_hash": harness_git_short_hash(),
        "harness_git_branch": harness_git_branch(),
        "harness_git_dirty": harness_git_dirty(),
        # Experiment config
        "experiment": experiment,
        "label": label,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "deploy_mode": deploy_mode,
        "host": host,
        "port": port,
        "duration_seconds": duration_seconds,
        "repetitions": repetitions,
        "warmup_seconds": warmup_seconds,
        "env_overrides": env_overrides or {},
        "extra_mcm_args": extra_args or [],
    }
    if client_config is not None:
        meta["client_config"] = client_config_to_dict(client_config)
    if extra:
        meta.update(extra)
    return meta


def save_metadata(meta: dict[str, Any], run_dir: Path) -> Path:
    """Write metadata.json to the run directory."""
    path = run_dir / "metadata.json"
    path.write_text(json.dumps(meta, indent=2, default=str))
    log.info("Metadata saved to %s", path)
    return path
