"""Orchestrator: deploy -> repetition loop (reset, warmup, collect) -> aggregate -> report."""

from __future__ import annotations

import contextlib
import json
import logging
import random
import re
import shutil
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from ab_harness.collector import collect_one_rep, measure_idle_cpu
from ab_harness.compare import compare_runs, save_comparison
from ab_harness.config import MCM_REPO_ROOT, ClientConfig
from ab_harness.deploy import (
    LocalDeployment,
    RemoteDeployment,
    create_deployer,
)
from ab_harness.kpi import aggregate_reps, summarize_rep
from ab_harness.metadata import collect_metadata, save_metadata
from ab_harness.report import generate_report
from ab_harness.sequential_testing import (
    AdaptiveState,
    MsprtAdaptiveState,
    check_interim,
    check_interim_msprt,
    create_adaptive_state,
    create_msprt_state,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Git source management (checkout / apply / restore)
# ---------------------------------------------------------------------------


class _GitContext:
    """Manages git state for --git-hash, --diff, --github-pr."""

    def __init__(
        self,
        *,
        git_hash: str | None = None,
        diff_path: str | None = None,
        github_pr: str | None = None,
    ):
        self.git_hash = git_hash
        self.diff_path = diff_path
        self.github_pr = github_pr
        self._original_ref: str | None = None
        self._stashed = False
        self._diff_applied = False
        self._active = any([git_hash, diff_path, github_pr])

    def enter(self) -> str:
        """Apply the source change. Returns a label derived from the input."""
        if not self._active:
            return ""

        # Stash uncommitted changes
        status = _git("status", "--porcelain")
        if status.strip():
            log.info("Stashing uncommitted changes...")
            _git("stash", "push", "-m", "ab_harness_auto_stash")
            self._stashed = True

        self._original_ref = _git("rev-parse", "--abbrev-ref", "HEAD")

        if self.git_hash:
            log.info("Checking out git hash: %s", self.git_hash)
            _git("checkout", self.git_hash)
            return f"hash_{self.git_hash[:8]}"

        elif self.diff_path:
            log.info("Applying diff: %s", self.diff_path)
            _git("apply", self.diff_path)
            self._diff_applied = True
            name = Path(self.diff_path).stem
            return f"diff_{name}"

        elif self.github_pr:
            pr = self.github_pr.strip().rstrip("/")
            # Extract PR number from URL if needed
            match = re.search(r"(\d+)$", pr)
            pr_num = match.group(1) if match else pr
            log.info("Checking out GitHub PR #%s", pr_num)
            subprocess.run(
                ["gh", "pr", "checkout", pr_num],
                cwd=MCM_REPO_ROOT,
                check=True,
            )
            return f"pr_{pr_num}"

        return ""

    def exit(self) -> None:
        """Restore original git state."""
        if not self._active:
            return

        try:
            if self._diff_applied and self.diff_path:
                log.info("Reverting applied diff...")
                _git("checkout", ".")
                self._diff_applied = False

            if self._original_ref:
                current = _git("rev-parse", "--abbrev-ref", "HEAD")
                if current != self._original_ref:
                    log.info("Restoring branch: %s", self._original_ref)
                    _git("checkout", self._original_ref)

            if self._stashed:
                log.info("Popping stashed changes...")
                _git("stash", "pop")
                self._stashed = False
        except Exception as exc:
            log.error("Failed to restore git state: %s", exc)
            log.error(
                "Manual cleanup may be needed. Original ref: %s, stashed: %s",
                self._original_ref,
                self._stashed,
            )


def _git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        capture_output=True,
        text=True,
        cwd=MCM_REPO_ROOT,
    )
    if result.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout.strip()


# ---------------------------------------------------------------------------
# Client management
# ---------------------------------------------------------------------------


def _read_client_log_tail(client: Any, max_lines: int = 15) -> str | None:
    """Read the last *max_lines* lines of a client's log file."""
    log_path = getattr(client, "log_path", None)
    if log_path and log_path.exists():
        try:
            lines = log_path.read_text().strip().splitlines()
            return "\n    ".join(lines[-max_lines:])
        except Exception:
            return None
    return None


class _ClientMonitor:
    """Background thread that restarts crashed clients every 1 s."""

    def __init__(self, clients: list[Any]) -> None:
        self._clients = clients
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name="client-monitor")

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        self._thread.join(timeout=5)

    def _run(self) -> None:
        while not self._stop_event.is_set():
            self._stop_event.wait(timeout=1)
            if self._stop_event.is_set():
                break
            for c in self._clients:
                if hasattr(c, "running") and not c.running:
                    name = str(getattr(c, "producer", None) or getattr(c, "path", "unknown"))
                    exit_code = c._process.returncode if c._process else "?"
                    log_tail = _read_client_log_tail(c)
                    detail = f" -- {log_tail.splitlines()[-1]}" if log_tail else ""
                    log.warning(
                        "Client %s crashed (exit code %s), restarting...%s",
                        name,
                        exit_code,
                        detail,
                    )
                    try:
                        c.stop()
                        c.start()
                    except Exception as exc:
                        log.warning("Failed to restart client %s: %s", name, exc)


def _start_clients(
    host: str,
    port: int,
    duration: int,
    client_config: ClientConfig,
    log_dir: Path,
) -> tuple[list[Any], _ClientMonitor | None]:
    """Start optional WebRTC / RTSP clients and a background monitor.

    Returns ``(clients, monitor)``.  The monitor restarts any client that
    crashes for the entire lifetime of the test; call ``monitor.stop()``
    before tearing down the clients.
    """
    clients: list[Any] = []

    for spec in client_config.webrtc:
        for i in range(spec.count):
            try:
                from ab_harness.clients import WebRTCClient

                wc = WebRTCClient(
                    host=host,
                    port=port,
                    producer=spec.name,
                    duration=duration,
                    log_dir=log_dir,
                    log_suffix=f"_{spec.name}_{i}" if spec.count > 1 else f"_{spec.name}",
                )
                wc.start()
                clients.append(wc)
                log.info(
                    "WebRTC client %d/%d started for producer: %s",
                    i + 1,
                    spec.count,
                    spec.name,
                )
            except Exception as exc:
                log.warning("Could not start WebRTC client for %s: %s", spec.name, exc)

    for spec in client_config.rtsp:
        for i in range(spec.count):
            try:
                from ab_harness.clients import RTSPClient

                rc = RTSPClient(
                    host=host,
                    path=spec.name,
                    log_dir=log_dir,
                    log_suffix=f"_{spec.name}_{i}" if spec.count > 1 else f"_{spec.name}",
                )
                rc.start()
                clients.append(rc)
                log.info(
                    "RTSP client %d/%d started for path: %s",
                    i + 1,
                    spec.count,
                    spec.name,
                )
            except Exception as exc:
                log.warning("Could not start RTSP client for %s: %s", spec.name, exc)

    monitor: _ClientMonitor | None = None
    if clients:
        monitor = _ClientMonitor(clients)
        monitor.start()

    return clients, monitor


def _stop_clients(clients: list[Any], monitor: _ClientMonitor | None = None) -> None:
    # Stop the monitor first so it doesn't restart clients we're tearing down
    if monitor is not None:
        monitor.stop()

    for c in clients:
        try:
            c.stop()
        except Exception as exc:
            log.debug("Error stopping client: %s", exc)


# ---------------------------------------------------------------------------
# Run helpers
# ---------------------------------------------------------------------------


def _next_iteration_index(iterations_dir: Path) -> int:
    """Find the next iteration number (001, 002, ...)."""
    if not iterations_dir.exists():
        return 1
    existing = sorted(iterations_dir.iterdir())
    max_idx = 0
    for d in existing:
        match = re.match(r"(\d+)_", d.name)
        if match:
            max_idx = max(max_idx, int(match.group(1)))
    return max_idx + 1


def _run_collection_loop(
    *,
    base_url: str,
    run_dir: Path,
    duration: int,
    repetitions: int,
    warmup: int,
    host: str,
    port: int,
    client_config: ClientConfig,
) -> None:
    """Execute the repetition loop: for each rep, reset + warmup + collect."""
    # Total duration for clients = (duration + warmup) * reps + buffer
    total_client_duration = (duration + warmup) * repetitions + 30

    # Start clients once for the entire collection
    clients, monitor = _start_clients(
        host,
        port,
        total_client_duration,
        client_config,
        run_dir,
    )

    try:
        for rep in range(1, repetitions + 1):
            rep_dir = run_dir / f"rep_{rep:03d}"
            rep_dir.mkdir(parents=True, exist_ok=True)

            log.info("=== Repetition %d/%d ===", rep, repetitions)
            collect_one_rep(
                base_url=base_url,
                rep_dir=rep_dir,
                duration_seconds=duration,
                warmup_seconds=warmup,
                client_config=client_config,
            )
    finally:
        _stop_clients(clients, monitor)


def _collect_single_rep(
    *,
    base_url: str,
    run_dir: Path,
    rep_index: int,
    duration: int,
    warmup: int,
    host: str,
    port: int,
    client_config: ClientConfig,
) -> None:
    """Collect a single repetition (used in interleaved mode).

    Starts clients, collects one rep, then stops clients.
    """
    rep_dir = run_dir / f"rep_{rep_index:03d}"
    rep_dir.mkdir(parents=True, exist_ok=True)

    # Client duration = warmup + duration + buffer
    client_duration = duration + warmup + 15

    clients, monitor = _start_clients(
        host,
        port,
        client_duration,
        client_config,
        run_dir,
    )
    try:
        log.info("=== Repetition %d ===", rep_index)
        collect_one_rep(
            base_url=base_url,
            rep_dir=rep_dir,
            duration_seconds=duration,
            warmup_seconds=warmup,
            client_config=client_config,
        )
    finally:
        _stop_clients(clients, monitor)


def _measure_and_save_idle_cpu(
    exp_dir: Path,
    deployer: RemoteDeployment,
) -> dict[str, Any] | None:
    """Measure idle system CPU on the Pi and save to system_baseline.json."""
    baseline_path = exp_dir / "system_baseline.json"
    if baseline_path.exists():
        log.info("system_baseline.json already exists, skipping idle measurement.")
        data: dict[str, Any] = json.loads(baseline_path.read_text())
        return data
    try:
        result = measure_idle_cpu(
            deployer.host,
            ssh_user=deployer.ssh_user,
            ssh_pwd=deployer.ssh_pwd,
            ssh_port=deployer.ssh_port,
            container=deployer.container,
        )
        baseline_path.write_text(json.dumps(result, indent=2))
        log.info("System baseline saved to %s", baseline_path)
        return result
    except Exception as exc:
        log.warning("Could not measure idle CPU: %s", exc)
        return None


def _snapshot_mcm_config(
    deployer: LocalDeployment | RemoteDeployment,
    dest_dir: Path,
    label: str,
) -> Path | None:
    """Download MCM's settings.json and save as ``settings_{label}.json``.

    Works with both local and remote deployers.
    Returns the local path on success, ``None`` otherwise.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    result = deployer.download_mcm_config(dest_dir)
    if result is None:
        return None
    # Rename to include the label (e.g. settings_pre.json, settings_post.json)
    target = dest_dir / f"settings_{label}.json"
    result.rename(target)
    log.info("MCM config snapshot saved: %s", target)
    return target


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_baseline(
    *,
    experiment: str,
    host: str,
    port: int,
    duration: int,
    repetitions: int,
    warmup: int,
    deploy_mode: str,
    env_overrides: dict[str, str],
    client_config: ClientConfig | None = None,
    extra_args: list[str] | None = None,
    runs_dir: Path,
    isolated: bool = True,
    restore: bool = True,
    blueos_tag: str | None = None,
    mcm_config: Path | None = None,
) -> Path:
    """Run the baseline (B) measurement."""
    if client_config is None:
        client_config = ClientConfig()

    exp_dir = runs_dir / experiment
    run_dir = exp_dir / "baseline"
    run_dir.mkdir(parents=True, exist_ok=True)

    log.info("=== BASELINE for experiment '%s' ===", experiment)

    # Metadata
    meta = collect_metadata(
        label="baseline",
        experiment=experiment,
        deploy_mode=deploy_mode,
        duration_seconds=duration,
        repetitions=repetitions,
        warmup_seconds=warmup,
        host=host,
        port=port,
        env_overrides=env_overrides,
        client_config=client_config,
        extra_args=extra_args,
    )

    # Deploy
    deployer = create_deployer(
        deploy_mode,
        host=host,
        port=port,
        env_overrides=env_overrides,
        extra_args=extra_args,
        isolated=isolated,
        blueos_tag=blueos_tag,
        mcm_config=mcm_config,
    )

    # Capture BlueOS versions while containers are still running, then
    # prepare isolated container if requested (stops all other services).
    is_isolated = isinstance(deployer, RemoteDeployment) and deployer.isolated
    if isinstance(deployer, RemoteDeployment):
        deployer.capture_blueos_versions()
    if isinstance(deployer, RemoteDeployment) and is_isolated:
        deployer.prepare_isolated_container()

    try:
        meta["isolated"] = is_isolated
        if blueos_tag:
            meta["blueos_tag"] = blueos_tag
        if mcm_config:
            meta["mcm_config"] = str(mcm_config)
        if isinstance(deployer, RemoteDeployment):
            meta["blueos_core_version"] = deployer.blueos_core_version
            meta["blueos_bootstrap_version"] = deployer.blueos_bootstrap_version

        config_dir = run_dir / "mcm_config"

        if isinstance(deployer, LocalDeployment):
            save_metadata(meta, run_dir)
            deployer.build()
            deployer.start()
        elif is_isolated:
            # Parallel: build locally + harden Pi (different machines).
            # Thermal warmup is the last Pi-side step so the SoC stays
            # hot right up until the first measurement.
            with ThreadPoolExecutor(max_workers=2) as pool:
                build_future = pool.submit(
                    deployer.build_only,
                    "baseline",
                    dest_dir=run_dir,
                )
                harden_future = pool.submit(deployer.harden_system)
            binary_path = build_future.result()
            warmup_info = harden_future.result()
            meta["thermal_warmup"] = warmup_info
            save_metadata(meta, run_dir)

            # Idle CPU (after warmup, before MCM starts)
            _measure_and_save_idle_cpu(exp_dir, deployer)

            # Deploy: upload + start MCM (config is restored inside swap_and_restart)
            deployer.upload_binary("baseline", binary_path)
            deployer.swap_and_restart("baseline")
        else:
            # Non-isolated remote: deploy script handles build + start
            # (config is restored inside build_and_deploy)
            save_metadata(meta, run_dir)
            _measure_and_save_idle_cpu(exp_dir, deployer)
            deployer.build_and_deploy()

        # Snapshot MCM config before collection
        _snapshot_mcm_config(deployer, config_dir, "pre")

        try:
            _run_collection_loop(
                base_url=deployer.base_url,
                run_dir=run_dir,
                duration=duration,
                repetitions=repetitions,
                warmup=warmup,
                host=host,
                port=port,
                client_config=client_config,
            )
        finally:
            # Snapshot MCM config after collection
            _snapshot_mcm_config(deployer, config_dir, "post")
            # Collect logs
            deployer.collect_logs(run_dir / "container_logs")
            if isinstance(deployer, LocalDeployment):
                deployer.stop()

        # Aggregate
        aggregate_reps(run_dir)
        log.info("Baseline complete: %s", run_dir)
        return run_dir

    finally:
        if isinstance(deployer, RemoteDeployment) and deployer.isolated:
            # teardown_isolated_container handles restore_remote_settings
            deployer.teardown_isolated_container(restore=restore)
        elif isinstance(deployer, RemoteDeployment):
            # Non-isolated: restore the device's original settings
            deployer.restore_remote_settings()


def run_iteration(
    *,
    experiment: str,
    label: str | None,
    host: str,
    port: int,
    duration: int,
    repetitions: int,
    warmup: int,
    deploy_mode: str,
    env_overrides: dict[str, str],
    client_config: ClientConfig | None = None,
    extra_args: list[str] | None = None,
    runs_dir: Path,
    git_hash: str | None = None,
    diff_path: str | None = None,
    github_pr: str | None = None,
    sequential: bool = False,
    adaptive: bool = False,
    adaptive_method: str = "gst",
    adaptive_look_every: int = 0,
    adaptive_kpis: list[str] | None = None,
    adaptive_no_futility: bool = False,
    no_build: bool = False,
    isolated: bool = True,
    restore: bool = True,
    blueos_tag: str | None = None,
    mcm_config: Path | None = None,
    strict: bool = False,
    seed: int | None = None,
) -> Path:
    """Run an iteration (A) measurement.

    By default uses interleaved ABBA execution for remote deployments.
    Pass ``sequential=True`` to fall back to the original sequential mode.

    Adaptive stopping (``adaptive=True``) enables sequential testing to
    stop early when:
    - **Efficacy**: a statistically significant result is found.
    - **Futility**: there is no realistic chance of reaching significance.

    Two adaptive methods are available (``adaptive_method``):
    - ``"msprt"``: Mixture Sequential Probability Ratio Test.
      Checks every trial with zero statistical penalty.
    - ``"gst"`` (default): Group Sequential Testing with Lan-DeMets (rho=3)
      alpha-spending.  Checks at pre-scheduled look points.

    ``repetitions`` becomes the *maximum* number of reps per side.

    ``no_build=True`` skips cross-compilation entirely and expects
    pre-built binaries in ``<exp_dir>/binaries/``.  Useful for retrying
    a crashed experiment without rebuilding.
    """
    if client_config is None:
        client_config = ClientConfig()

    exp_dir = runs_dir / experiment
    baseline_dir = exp_dir / "baseline"
    iterations_dir = exp_dir / "iterations"
    iterations_dir.mkdir(parents=True, exist_ok=True)

    # Git source management
    git_ctx = _GitContext(git_hash=git_hash, diff_path=diff_path, github_pr=github_pr)
    source_label = git_ctx.enter()

    try:
        # Derive label
        if not label:
            label = source_label or "unnamed"

        idx = _next_iteration_index(iterations_dir)
        dir_name = f"{idx:03d}_{label}"
        run_dir = iterations_dir / dir_name
        run_dir.mkdir(parents=True, exist_ok=True)

        log.info("=== ITERATION %s for experiment '%s' ===", dir_name, experiment)

        # Metadata
        meta = collect_metadata(
            label=label,
            experiment=experiment,
            deploy_mode=deploy_mode,
            duration_seconds=duration,
            repetitions=repetitions,
            warmup_seconds=warmup,
            host=host,
            port=port,
            env_overrides=env_overrides,
            client_config=client_config,
            extra_args=extra_args,
            extra={
                "iteration_index": idx,
                "git_source_hash": git_hash,
                "diff_path": diff_path,
                "github_pr": github_pr,
                "mcm_config": str(mcm_config) if mcm_config else None,
            },
        )
        save_metadata(meta, run_dir)

        deployer = create_deployer(
            deploy_mode,
            host=host,
            port=port,
            env_overrides=env_overrides,
            extra_args=extra_args,
            isolated=isolated,
            blueos_tag=blueos_tag,
            mcm_config=mcm_config,
        )

        # Capture BlueOS versions early (while containers are still up).
        # For isolated mode, prepare_isolated_container will capture them
        # automatically, but for non-isolated we need to do it here.
        if isinstance(deployer, RemoteDeployment):
            deployer.capture_blueos_versions()
            meta["blueos_core_version"] = deployer.blueos_core_version
            meta["blueos_bootstrap_version"] = deployer.blueos_bootstrap_version

        # ---- Decide interleaved vs sequential ----
        use_interleaved = (
            not sequential
            and isinstance(deployer, RemoteDeployment)
            and baseline_dir.exists()
            and (baseline_dir / "aggregate.json").exists()
        )

        if use_interleaved:
            assert isinstance(deployer, RemoteDeployment)
            _run_interleaved(
                deployer=deployer,
                exp_dir=exp_dir,
                baseline_dir=baseline_dir,
                iteration_dir=run_dir,
                duration=duration,
                repetitions=repetitions,
                warmup=warmup,
                host=host,
                port=port,
                client_config=client_config,
                meta=meta,
                env_overrides=env_overrides,
                adaptive=adaptive,
                adaptive_method=adaptive_method,
                adaptive_look_every=adaptive_look_every,
                adaptive_kpis=adaptive_kpis,
                adaptive_no_futility=adaptive_no_futility,
                no_build=no_build,
                restore=restore,
                seed=seed,
            )
        else:
            if use_interleaved is False and not sequential:
                log.info("No baseline found or local mode; using sequential execution.")
            _run_sequential_iteration(
                deployer=deployer,
                run_dir=run_dir,
                duration=duration,
                repetitions=repetitions,
                warmup=warmup,
                host=host,
                port=port,
                client_config=client_config,
            )

        # Aggregate iteration data
        aggregate_reps(run_dir)

        # Compare vs baseline if it exists
        if baseline_dir.exists() and (baseline_dir / "aggregate.json").exists():
            log.info("Comparing against baseline...")
            comparison = compare_runs(run_dir, baseline_dir, strict=strict)
            save_comparison(comparison, run_dir / "comparison.json")

            # Find previous iteration for secondary comparison
            prev_comp = None
            if idx > 1:
                prev_dirs = sorted(iterations_dir.iterdir())
                for pd in reversed(prev_dirs):
                    if pd.name != dir_name and (pd / "aggregate.json").exists():
                        with contextlib.suppress(Exception):
                            prev_comp = compare_runs(run_dir, pd, strict=strict)
                        break

            report_path = generate_report(
                comparison,
                run_dir,
                baseline_dir,
                metadata=meta,
                prev_comparison=prev_comp,
            )
            log.info("Report: %s", report_path)
        else:
            log.warning(
                "No baseline found at %s. Run 'baseline' first to enable comparison.",
                baseline_dir,
            )

        log.info("Iteration complete: %s", run_dir)
        return run_dir

    finally:
        git_ctx.exit()
        # Non-isolated remote: restore the device's original settings.
        # (Isolated remote is handled by teardown_isolated_container
        #  inside _run_interleaved.)
        if isinstance(deployer, RemoteDeployment) and not deployer.isolated:
            deployer.restore_remote_settings()


# ---------------------------------------------------------------------------
# Sequential iteration (original behaviour)
# ---------------------------------------------------------------------------


def _run_sequential_iteration(
    *,
    deployer: LocalDeployment | RemoteDeployment,
    run_dir: Path,
    duration: int,
    repetitions: int,
    warmup: int,
    host: str,
    port: int,
    client_config: ClientConfig,
) -> None:
    """Build, deploy, then collect all reps in sequence (original flow)."""
    if isinstance(deployer, LocalDeployment):
        deployer.build()
        deployer.start()
    else:
        deployer.build_and_deploy()

    config_dir = run_dir / "mcm_config"
    _snapshot_mcm_config(deployer, config_dir, "pre")

    try:
        _run_collection_loop(
            base_url=deployer.base_url,
            run_dir=run_dir,
            duration=duration,
            repetitions=repetitions,
            warmup=warmup,
            host=host,
            port=port,
            client_config=client_config,
        )
    finally:
        _snapshot_mcm_config(deployer, config_dir, "post")
        deployer.collect_logs(run_dir / "container_logs")
        if isinstance(deployer, LocalDeployment):
            deployer.stop()


# ---------------------------------------------------------------------------
# Live per-rep mean extraction (for adaptive interim analysis)
# ---------------------------------------------------------------------------


def _compute_live_per_rep_means(
    iteration_dir: Path,
    baseline_dir: Path,
    completed_iter_reps: set[int],
    completed_base_reps: set[int],
) -> tuple[dict[str, list[float]], dict[str, list[float]]]:
    """Compute per-rep KPI means from data collected so far.

    Reads the rep directories that have been completed and extracts
    per-rep means for each KPI, suitable for feeding into
    :func:`sequential_testing.check_interim`.

    Returns (a_per_rep, b_per_rep) where each is a dict mapping
    KPI name to a list of per-rep mean values.
    """
    from ab_harness.kpi import KPI_BY_NAME

    a_per_rep: dict[str, list[float]] = {k: [] for k in KPI_BY_NAME}
    b_per_rep: dict[str, list[float]] = {k: [] for k in KPI_BY_NAME}

    for rep_idx in sorted(completed_iter_reps):
        rep_dir = iteration_dir / f"rep_{rep_idx:03d}"
        if not rep_dir.exists():
            continue
        try:
            summary = summarize_rep(rep_dir, rep_idx)
            for kpi_name in KPI_BY_NAME:
                mean_val = summary.kpis.get(kpi_name, {}).get("mean")
                if mean_val is not None:
                    a_per_rep[kpi_name].append(mean_val)
        except Exception as exc:
            log.debug("Could not summarize iter rep %d: %s", rep_idx, exc)

    for rep_idx in sorted(completed_base_reps):
        rep_dir = baseline_dir / f"rep_{rep_idx:03d}"
        if not rep_dir.exists():
            continue
        try:
            summary = summarize_rep(rep_dir, rep_idx)
            for kpi_name in KPI_BY_NAME:
                mean_val = summary.kpis.get(kpi_name, {}).get("mean")
                if mean_val is not None:
                    b_per_rep[kpi_name].append(mean_val)
        except Exception as exc:
            log.debug("Could not summarize base rep %d: %s", rep_idx, exc)

    return a_per_rep, b_per_rep


# ---------------------------------------------------------------------------
# Interleaved ABBA execution
# ---------------------------------------------------------------------------


def _run_interleaved(
    *,
    deployer: RemoteDeployment,
    exp_dir: Path,
    baseline_dir: Path,
    iteration_dir: Path,
    duration: int,
    repetitions: int,
    warmup: int,
    host: str,
    port: int,
    client_config: ClientConfig,
    meta: dict,
    env_overrides: dict[str, str] | None = None,
    adaptive: bool = False,
    adaptive_method: str = "gst",
    adaptive_look_every: int = 0,
    adaptive_kpis: list[str] | None = None,
    adaptive_no_futility: bool = False,
    no_build: bool = False,
    restore: bool = True,
    seed: int | None = None,
) -> None:
    """Interleaved A/B data collection.

    Ordering strategy (auto-selected):
    - N <= 10: ABBA (cancels linear drift, better for small samples)
    - N >  10: Randomized balanced (protects against any temporal
      pattern, gold standard for large samples)

    Binary management (build-once):
    1. Both binaries are stored in ``<exp_dir>/binaries/mcm_baseline``
       and ``<exp_dir>/binaries/mcm_iteration``.  If they already exist
       the build step is skipped entirely.
    2. Binaries are uploaded to ``pi:/tmp/mcm_{tag}`` once.  If the
       remote file already exists the upload is skipped.
    3. ``swap_and_restart`` does a fast ``docker cp`` from ``/tmp`` into
       the container -- no rebuild or re-upload per trial.

    Pass ``no_build=True`` (CLI: ``--no-build``) to require pre-built
    binaries and skip compilation entirely.

    When ``adaptive=True``, interim analyses decide when to stop early.
    Two methods are available via ``adaptive_method``:

    - ``"msprt"``: Mixture Sequential Probability Ratio Test.
      Checks after *every* completed A/B pair (no look schedule needed).
      The mSPRT statistic is recomputed incrementally; each new pair's
      KPI means are added and the likelihood ratio is evaluated.

    - ``"gst"`` (default): Group Sequential Testing with Lan-DeMets (rho=3).
      Checks at pre-scheduled look points (every ``adaptive_look_every``
      reps per side).  Uses a Welch's t-test at each look point with
      alpha-spending boundaries.

    Baseline reps are re-collected fresh so both sides face the same
    temporal conditions. The old baseline data is left untouched; the
    new interleaved baseline data is written to a separate directory
    and then *replaces* the baseline aggregate at the end.
    """
    use_randomized = repetitions > 10
    strategy = "randomized balanced" if use_randomized else "ABBA"
    total_trials = repetitions * 2
    log.info(
        "=== INTERLEAVED %s mode (%d reps/side, %d total trials) ===",
        strategy,
        repetitions,
        total_trials,
    )

    # Prepare isolated container if requested (stops all other services).
    # Hardening (clear memory, governor, thermal warmup) is deferred to
    # _run_interleaved_body() so it runs *after* the local build phase,
    # keeping the Pi thermally primed right before the first measurement.
    if deployer.isolated:
        deployer.prepare_isolated_container()

    try:
        _run_interleaved_body(
            deployer=deployer,
            exp_dir=exp_dir,
            baseline_dir=baseline_dir,
            iteration_dir=iteration_dir,
            duration=duration,
            repetitions=repetitions,
            warmup=warmup,
            host=host,
            port=port,
            client_config=client_config,
            meta=meta,
            env_overrides=env_overrides,
            adaptive=adaptive,
            adaptive_method=adaptive_method,
            adaptive_look_every=adaptive_look_every,
            adaptive_kpis=adaptive_kpis,
            adaptive_no_futility=adaptive_no_futility,
            no_build=no_build,
            use_randomized=use_randomized,
            strategy=strategy,
            total_trials=total_trials,
            seed=seed,
        )
    finally:
        if deployer.isolated:
            deployer.teardown_isolated_container(restore=restore)


def _run_interleaved_body(
    *,
    deployer: RemoteDeployment,
    exp_dir: Path,
    baseline_dir: Path,
    iteration_dir: Path,
    duration: int,
    repetitions: int,
    warmup: int,
    host: str,
    port: int,
    client_config: ClientConfig,
    meta: dict,
    env_overrides: dict[str, str] | None = None,
    adaptive: bool = False,
    adaptive_method: str = "gst",
    adaptive_look_every: int = 0,
    adaptive_kpis: list[str] | None = None,
    adaptive_no_futility: bool = False,
    no_build: bool = False,
    use_randomized: bool = False,
    strategy: str = "",
    total_trials: int = 0,
    seed: int | None = None,
) -> None:
    """Inner body of interleaved execution (separated for isolated teardown)."""

    # Canonical binary storage -- build products live here and are
    # reused across retries so we never rebuild or re-upload unnecessarily.
    binaries_dir = exp_dir / "binaries"
    binaries_dir.mkdir(parents=True, exist_ok=True)

    # --- Phase 1: Build both binaries (skipped if cached in binaries_dir) ---
    a_cached = binaries_dir / "mcm_iteration"
    b_cached = binaries_dir / "mcm_baseline"

    if no_build:
        # Explicit --no-build: expect pre-built binaries
        if not a_cached.exists() or not b_cached.exists():
            raise FileNotFoundError(
                f"--no-build requires both binaries to exist in {binaries_dir}. "
                f"Found: iteration={a_cached.exists()}, baseline={b_cached.exists()}"
            )
        log.info("Phase 1: Using pre-built binaries from %s (--no-build)", binaries_dir)
        a_path = a_cached
        b_path = b_cached
    elif a_cached.exists() and b_cached.exists():
        log.info("Phase 1: Reusing cached binaries from %s", binaries_dir)
        a_path = a_cached
        b_path = b_cached
    else:
        log.info("Phase 1: Building iteration (A) binary...")
        a_path = deployer.build_only("iteration", dest_dir=binaries_dir)

        log.info("Phase 1: Building baseline (B) binary...")
        _stash_and_build_baseline(deployer, dest_dir=binaries_dir)
        b_path = b_cached

    log.info(
        "Binaries: A=%s (%d bytes), B=%s (%d bytes)",
        a_path.name,
        a_path.stat().st_size,
        b_path.name,
        b_path.stat().st_size,
    )

    # --- Phase 1b: Harden Pi (after builds, right before measurements) ---
    # Runs after the local build so the Pi stays thermally primed.
    if deployer.isolated:
        warmup_info = deployer.harden_system()
        meta["thermal_warmup"] = warmup_info

    # Idle CPU (after warmup, before any MCM)
    _measure_and_save_idle_cpu(exp_dir, deployer)

    # --- Phase 1c: Configure per-tag env vars for swap_and_restart ---
    # Iteration tag gets user's --env overrides; baseline tag uses defaults.
    # NOTE: MCM_RTSPSRC_* env vars are passed to the container but are NOT
    # currently read by the binary (pipeline params are hardcoded in source).
    if env_overrides:
        deployer.set_tag_env("iteration", env_overrides)

    # --- Phase 2: Upload both to the Pi (skipped if already present) ---
    log.info("Phase 2: Uploading binaries to Pi (skipped if already present)...")
    deployer.upload_binary("iteration", a_path)
    deployer.upload_binary("baseline", b_path)

    # --- Phase 3: Generate trial sequence ---
    interleaved_baseline_dir = exp_dir / "baseline_interleaved"
    interleaved_baseline_dir.mkdir(parents=True, exist_ok=True)

    # Build the flat trial list: (tag, target_dir, rep_index)
    trials: list[tuple[str, Path, int]] = []
    for r in range(1, repetitions + 1):
        trials.append(("baseline", interleaved_baseline_dir, r))
        trials.append(("iteration", iteration_dir, r))

    if use_randomized:
        # Use a local RNG for reproducibility.  If the user provided a
        # seed we use it; otherwise we generate one and log it so the
        # experiment can be reproduced post-hoc.
        if seed is None:
            seed = random.randint(0, 2**32 - 1)
        rng = random.Random(seed)
        rng.shuffle(trials)
        meta["randomization_seed"] = seed
        log.info("Randomized %d trials (seed=%d).", len(trials), seed)
    else:
        # ABBA ordering: rearrange into B-A-A-B-B-A-A-B-...
        abba_trials: list[tuple[str, Path, int]] = []
        for r in range(1, repetitions + 1):
            b_trial = ("baseline", interleaved_baseline_dir, r)
            a_trial = ("iteration", iteration_dir, r)
            if r % 2 == 1:
                abba_trials.extend([b_trial, a_trial])
            else:
                abba_trials.extend([a_trial, b_trial])
        trials = abba_trials
        log.info("ABBA ordering for %d trials.", len(trials))

    # Save the trial sequence for auditability / reproducibility
    sequence_data: dict[str, Any] = {
        "strategy": strategy,
        "seed": meta.get("randomization_seed"),
        "trials": [
            {"trial": i + 1, "tag": tag, "rep_index": rep_idx}
            for i, (tag, _, rep_idx) in enumerate(trials)
        ],
    }
    seq_path = exp_dir / "trial_sequence.json"
    seq_path.write_text(json.dumps(sequence_data, indent=2))
    log.info("Trial sequence saved to %s", seq_path)

    # Estimate total time
    est_per_trial = duration + warmup + 15  # 15s swap overhead
    est_total = len(trials) * est_per_trial
    log.info(
        "Estimated total collection time: %d trials x ~%ds = ~%.1fh",
        len(trials),
        est_per_trial,
        est_total / 3600,
    )

    # --- Adaptive stopping state ---
    adaptive_state: AdaptiveState | MsprtAdaptiveState | None = None
    use_msprt = adaptive and adaptive_method == "msprt"
    if adaptive:
        if use_msprt:
            adaptive_state = create_msprt_state(
                max_reps=repetitions,
                alpha=0.05,
                target_kpis=adaptive_kpis,
                futility=not adaptive_no_futility,
            )
            log.info(
                "Adaptive stopping ENABLED (mSPRT): checks every trial, max %d reps/side",
                repetitions,
            )
        else:
            adaptive_state = create_adaptive_state(
                max_reps=repetitions,
                look_every=adaptive_look_every,
                alpha=0.05,
                target_kpis=adaptive_kpis,
                futility=not adaptive_no_futility,
            )
            assert isinstance(adaptive_state, AdaptiveState)
            log.info(
                "Adaptive stopping ENABLED (GST): will check at %d look points, max %d reps/side",
                len(adaptive_state.boundaries),
                repetitions,
            )

    # --- Phase 3b: Snapshot MCM config before trials ---
    # Push the pristine config into the container so the snapshot can read it
    # (no swap has happened yet in this phase, so the override path is empty).
    if isinstance(deployer, RemoteDeployment):
        deployer._restore_mcm_config()
    config_dir = iteration_dir / "mcm_config"
    _snapshot_mcm_config(deployer, config_dir, "pre")

    # --- Phase 4: Execute trials ---
    log.info("Phase 3: Executing %d trials...", len(trials))

    # Track completed reps per side for adaptive interim analysis
    completed_baseline_reps: set[int] = set()
    completed_iteration_reps: set[int] = set()
    early_stopped = False
    # For mSPRT: track which reps we've already fed into the model
    _msprt_last_reps_per_side = 0

    for i, (tag, target_dir, rep_idx) in enumerate(trials, 1):
        log.info(
            "--- Trial %d/%d: %s rep %d ---",
            i,
            len(trials),
            tag,
            rep_idx,
        )
        deployer.swap_and_restart(tag)
        _collect_single_rep(
            base_url=deployer.base_url,
            run_dir=target_dir,
            rep_index=rep_idx,
            duration=duration,
            warmup=warmup,
            host=host,
            port=port,
            client_config=client_config,
        )
        deployer.collect_logs(
            target_dir / "container_logs" / f"trial_{i:04d}_{tag}_rep{rep_idx:03d}"
        )

        # Track completed reps per side
        if tag == "baseline":
            completed_baseline_reps.add(rep_idx)
        else:
            completed_iteration_reps.add(rep_idx)

        # --- Adaptive interim analysis ---
        if adaptive_state and not adaptive_state.stopped:
            reps_per_side = min(len(completed_baseline_reps), len(completed_iteration_reps))

            if use_msprt and isinstance(adaptive_state, MsprtAdaptiveState):
                # mSPRT: feed new pairs incrementally and check every trial
                if reps_per_side > _msprt_last_reps_per_side:
                    # New pair(s) completed -- compute their KPI means
                    # and feed them into the mSPRT model one at a time
                    a_per_rep_all, b_per_rep_all = _compute_live_per_rep_means(
                        iteration_dir,
                        interleaved_baseline_dir,
                        completed_iteration_reps,
                        completed_baseline_reps,
                    )
                    # Feed only the new pairs since last check
                    for pair_idx in range(_msprt_last_reps_per_side, reps_per_side):
                        kpi_a: dict[str, float] = {}
                        kpi_b: dict[str, float] = {}
                        for kpi_name, vals in a_per_rep_all.items():
                            if pair_idx < len(vals):
                                kpi_a[kpi_name] = vals[pair_idx]
                        for kpi_name, vals in b_per_rep_all.items():
                            if pair_idx < len(vals):
                                kpi_b[kpi_name] = vals[pair_idx]
                        interim = check_interim_msprt(
                            adaptive_state,
                            kpi_a,
                            kpi_b,
                            pair_idx + 1,
                        )
                        if interim and interim.stopped:
                            break

                    _msprt_last_reps_per_side = reps_per_side
                    adaptive_state.save(exp_dir / "adaptive_state.json")

                    if adaptive_state.stopped:
                        early_stopped = True
                        log.info(
                            "*** ADAPTIVE EARLY STOP at trial %d/%d (%s, %d reps/side) ***",
                            i,
                            len(trials),
                            adaptive_state.stop_reason,
                            reps_per_side,
                        )
                        break

            elif (
                isinstance(adaptive_state, AdaptiveState)
                and reps_per_side in adaptive_state.boundaries
            ):
                # GST: check only at pre-scheduled look points
                a_per_rep, b_per_rep = _compute_live_per_rep_means(
                    iteration_dir,
                    interleaved_baseline_dir,
                    completed_iteration_reps,
                    completed_baseline_reps,
                )
                interim = check_interim(
                    adaptive_state,
                    a_per_rep,
                    b_per_rep,
                    reps_per_side,
                )
                adaptive_state.save(exp_dir / "adaptive_state.json")

                if interim and interim.stopped:
                    early_stopped = True
                    log.info(
                        "*** ADAPTIVE EARLY STOP at trial %d/%d (%s, %d reps/side) ***",
                        i,
                        len(trials),
                        adaptive_state.stop_reason,
                        reps_per_side,
                    )
                    break

        # Progress estimate
        elapsed_trials = i
        remaining = len(trials) - elapsed_trials
        log.info(
            "Progress: %d/%d (%.0f%%), ~%.1fh remaining",
            elapsed_trials,
            len(trials),
            100 * elapsed_trials / len(trials),
            remaining * est_per_trial / 3600,
        )

    if adaptive_state:
        if not early_stopped:
            log.info("Adaptive: all %d trials completed without early stopping.", len(trials))
        adaptive_state.save(exp_dir / "adaptive_state.json")

    # --- Phase 4b: Snapshot MCM config after trials ---
    _snapshot_mcm_config(deployer, config_dir, "post")

    # --- Phase 5: Re-aggregate baseline with interleaved data ---
    log.info("Phase 4: Aggregating interleaved baseline data...")
    aggregate_reps(interleaved_baseline_dir)

    # Copy the interleaved aggregate over the baseline's aggregate so
    # comparisons use the temporally-matched data.
    interleaved_agg = interleaved_baseline_dir / "aggregate.json"
    if interleaved_agg.exists():
        shutil.copy2(interleaved_agg, baseline_dir / "aggregate.json")
        log.info("Baseline aggregate replaced with interleaved baseline data.")

    # Save metadata noting this was interleaved
    meta["isolated"] = deployer.isolated
    meta["interleaved"] = True
    meta["interleaved_strategy"] = strategy
    meta["interleaved_rounds"] = repetitions
    meta["total_trials"] = total_trials
    if adaptive_state:
        meta["adaptive"] = True
        meta["adaptive_method"] = adaptive_method
        meta["adaptive_stopped"] = adaptive_state.stopped
        meta["adaptive_stop_reason"] = adaptive_state.stop_reason
        meta["adaptive_stop_at_rep"] = adaptive_state.stop_at_rep
        meta["adaptive_stop_kpi"] = adaptive_state.stop_kpi
        meta["adaptive_look_count"] = len(adaptive_state.interim_results)
        meta["actual_reps_per_side"] = min(
            len(completed_baseline_reps),
            len(completed_iteration_reps),
        )
    save_metadata(meta, iteration_dir)

    log.info("Interleaved collection complete.")


def _stash_and_build_baseline(
    deployer: RemoteDeployment,
    *,
    dest_dir: Path | None = None,
) -> None:
    """Temporarily stash iteration changes, build baseline binary, then restore.

    This handles the case where _GitContext has already checked out the
    iteration code.  We stash, build, then pop.

    If *dest_dir* is given it is forwarded to ``build_only`` so the binary
    is saved (and cached) in the canonical experiment binaries folder.
    """
    # Check if working tree is dirty (iteration changes applied as diff)
    status = _git("status", "--porcelain")
    has_local_changes = bool(status.strip())

    if has_local_changes:
        log.info("Stashing iteration changes to build baseline binary...")
        _git("stash", "push", "-m", "ab_interleaved_baseline_build")

    try:
        deployer.build_only("baseline", dest_dir=dest_dir)
    finally:
        if has_local_changes:
            log.info("Restoring iteration changes...")
            _git("stash", "pop")


# ---------------------------------------------------------------------------
# Report regeneration
# ---------------------------------------------------------------------------


def regenerate_report(
    *,
    experiment: str,
    iteration: str,
    runs_dir: Path,
) -> Path:
    """Regenerate the report for an existing iteration."""
    exp_dir = runs_dir / experiment
    baseline_dir = exp_dir / "baseline"
    iteration_dir = exp_dir / "iterations" / iteration

    if not iteration_dir.exists():
        raise FileNotFoundError(f"Iteration not found: {iteration_dir}")

    # Load metadata
    meta_path = iteration_dir / "metadata.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}

    # Re-aggregate if needed
    if not (iteration_dir / "aggregate.json").exists():
        aggregate_reps(iteration_dir)

    # Compare
    comparison = compare_runs(iteration_dir, baseline_dir)
    save_comparison(comparison, iteration_dir / "comparison.json")

    report_path = generate_report(
        comparison,
        iteration_dir,
        baseline_dir,
        metadata=meta,
    )
    log.info("Report regenerated: %s", report_path)
    return report_path
