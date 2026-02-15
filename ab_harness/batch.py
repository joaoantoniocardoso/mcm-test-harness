"""Batch and overnight A/B test orchestration.

Extracted from ``cli.py`` to keep that module focused on argparse wiring.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import logging
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Any

log = logging.getLogger("ab_harness")


# ---------------------------------------------------------------------------
# Batch YAML config loading
# ---------------------------------------------------------------------------

BATCH_DEFAULTS: dict[str, Any] = {
    "host": "blueos.local",
    "port": 6020,
    "duration": 60,
    "repetitions": 30,
    "warmup": 5,
    "deploy": "remote",
    "isolated": True,
    "adaptive_method": "gst",
    "adaptive_look_every": 0,
    "adaptive_kpis": None,
    "adaptive_no_futility": False,
    "no_build": False,
    "env": {},
    "webrtc": [],
    "rtsp": [],
    "thumbnails": [],
    "thumbnail_interval": 1,
    "extra_args": [],
    "blueos_tag": None,
    "mcm_config": None,
}


def load_batch_config(config_path: Path) -> dict[str, Any]:
    """Load and validate a batch YAML config file.

    Returns a dict with keys ``defaults`` (merged with built-in defaults)
    and ``experiments`` (list of per-experiment dicts, each merged with
    defaults).  Also includes ``_raw_bytes`` for hashing.
    """
    import yaml

    raw_bytes = config_path.read_bytes()
    data = yaml.safe_load(raw_bytes)
    if not isinstance(data, dict):
        raise ValueError(f"Batch config must be a YAML mapping, got {type(data).__name__}")

    # Merge user defaults over built-in defaults
    user_defaults = data.get("defaults", {}) or {}
    defaults = {**BATCH_DEFAULTS, **user_defaults}

    experiments_raw = data.get("experiments")
    if not experiments_raw or not isinstance(experiments_raw, list):
        raise ValueError("Batch config must have a non-empty 'experiments' list")

    experiments: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    for idx, exp_raw in enumerate(experiments_raw):
        if not isinstance(exp_raw, dict):
            raise ValueError(
                f"experiments[{idx}]: expected a mapping, got {type(exp_raw).__name__}"
            )
        if "experiment" not in exp_raw:
            raise ValueError(f"experiments[{idx}]: missing required key 'experiment'")

        name = exp_raw["experiment"]
        if name in seen_names:
            raise ValueError(f"experiments[{idx}]: duplicate experiment name '{name}'")
        seen_names.add(name)

        # Merge: built-in defaults < config defaults < per-experiment overrides
        merged = {**defaults, **exp_raw}

        # Validate source mutual exclusivity
        sources = [s for s in ("github_pr", "git_hash", "diff") if merged.get(s) is not None]
        if len(sources) > 1:
            raise ValueError(
                f"experiments[{idx}] ('{name}'): only one of github_pr, git_hash, "
                f"diff may be set, got: {', '.join(sources)}"
            )

        experiments.append(merged)

    config_hash = hashlib.sha256(raw_bytes).hexdigest()
    return {
        "defaults": defaults,
        "experiments": experiments,
        "config_hash": config_hash,
        "config_file": str(config_path),
    }


# ---------------------------------------------------------------------------
# Single overnight run
# ---------------------------------------------------------------------------


def _parse_env(env_list: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in env_list:
        if "=" not in item:
            log.warning("Ignoring malformed --env value: %s", item)
            continue
        k, v = item.split("=", 1)
        result[k] = v
    return result


def run_single_overnight(params: dict[str, Any], *, restore: bool = True) -> dict[str, Any]:
    """Run a single overnight A/B test (baseline + adaptive iteration + replay).

    *params* is a flat dict with the same keys as the YAML per-experiment
    config (see ``load_batch_config``).  Returns a result dict with keys:
    ``experiment``, ``report_path`` (str | None), ``replay_path`` (str | None).
    """
    from ab_harness.config import build_client_config
    from ab_harness.kpi import load_aggregate
    from ab_harness.runner import run_baseline, run_iteration
    from ab_harness.sequential_testing import (
        format_replay_summary,
        replay_all_methods,
        serialize_replay_results,
    )

    experiment = params["experiment"]
    runs_dir = Path(params.get("runs_dir") or _default_runs_dir())
    env_overrides = params.get("env", {}) or {}
    if isinstance(env_overrides, list):
        env_overrides = _parse_env(env_overrides)
    no_build = params.get("no_build", False)
    exp_dir = runs_dir / experiment
    baseline_agg = exp_dir / "baseline" / "aggregate.json"

    client_config = build_client_config(
        webrtc=params.get("webrtc") or [],
        rtsp=params.get("rtsp") or [],
        thumbnails=params.get("thumbnails") or [],
        thumbnail_interval=params.get("thumbnail_interval", 1),
    )

    print("=" * 60)
    print(f"Overnight A/B Test: {experiment}")
    print(f"Started: {datetime.datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"Max reps: {params['repetitions']} per side, {params['duration']}s each")
    print(f"Adaptive: {params.get('adaptive_method', 'gst')}")
    if no_build:
        print("Build: SKIPPED (--no-build, reusing cached binaries)")
    print("=" * 60)
    print()

    # --- Step 1: Bootstrap baseline (1 rep, skipped if exists) ----------------
    if no_build and baseline_agg.exists():
        print(">>> Step 1: Baseline already exists, skipping (--no-build)")
    else:
        print(">>> Step 1: Bootstrap baseline (1 rep)")
        run_baseline(
            experiment=experiment,
            host=params["host"],
            port=params["port"],
            duration=params["duration"],
            repetitions=1,
            warmup=params["warmup"],
            deploy_mode=params["deploy"],
            env_overrides=env_overrides,
            client_config=client_config,
            extra_args=params.get("extra_args") or [],
            runs_dir=runs_dir,
            isolated=params.get("isolated", True),
            restore=False,  # keep container up for iteration
            blueos_tag=params.get("blueos_tag"),
            mcm_config=Path(params["mcm_config"]) if params.get("mcm_config") else None,
        )

    print()
    print(f">>> Step 2: Adaptive iteration (max {params['repetitions']} reps/side)")
    run_iteration(
        experiment=experiment,
        label=params.get("label"),
        host=params["host"],
        port=params["port"],
        duration=params["duration"],
        repetitions=params["repetitions"],
        warmup=params["warmup"],
        deploy_mode=params["deploy"],
        env_overrides=env_overrides,
        client_config=client_config,
        extra_args=params.get("extra_args") or [],
        runs_dir=runs_dir,
        git_hash=params.get("git_hash"),
        diff_path=str(Path(params["diff"]).resolve()) if params.get("diff") else None,
        github_pr=params.get("github_pr"),
        sequential=False,
        adaptive=True,
        adaptive_method=params.get("adaptive_method", "gst"),
        adaptive_look_every=params.get("adaptive_look_every", 0),
        adaptive_kpis=params.get("adaptive_kpis"),
        adaptive_no_futility=params.get("adaptive_no_futility", False),
        no_build=no_build,
        isolated=params.get("isolated", True),
        restore=restore,
        blueos_tag=params.get("blueos_tag"),
        mcm_config=Path(params["mcm_config"]) if params.get("mcm_config") else None,
    )

    # --- Step 3: Replay through sequential testing methods --------------------
    print()
    print(">>> Step 3: Replaying through sequential testing methods")
    replay_kpis = params.get("adaptive_kpis") or ["system_cpu_pct"]
    report_path: str | None = None
    replay_path: str | None = None

    iterations_dir = exp_dir / "iterations"
    if iterations_dir.is_dir():
        iter_dirs = sorted(iterations_dir.iterdir())
        if iter_dirs:
            iteration_dir = iter_dirs[-1]
            baseline_dir = exp_dir / "baseline"
            try:
                b_agg = load_aggregate(baseline_dir)
                a_agg = load_aggregate(iteration_dir)
            except FileNotFoundError as e:
                print(f"WARNING: Cannot replay, missing aggregate: {e}")
                b_agg, a_agg = None, None

            if a_agg and b_agg:
                a_data = a_agg.get("aggregate", {})
                b_data = b_agg.get("aggregate", {})
                all_results = []
                for kpi_name in replay_kpis:
                    a_reps = a_data.get(kpi_name, {}).get("per_rep_means", [])
                    b_reps = b_data.get(kpi_name, {}).get("per_rep_means", [])
                    if a_reps and b_reps:
                        result = replay_all_methods(
                            a_per_rep=a_reps,
                            b_per_rep=b_reps,
                            kpi_name=kpi_name,
                            alpha=0.05,
                        )
                        all_results.append(result)
                        print(format_replay_summary(result))

                if all_results:
                    output_path = iteration_dir / "sequential_replay.json"
                    output_path.write_text(
                        json.dumps(
                            serialize_replay_results(all_results),
                            indent=2,
                            default=str,
                        )
                    )
                    replay_path = str(output_path)
                    print(f"\nReplay results: {output_path}")

            # Locate report
            rpt = iteration_dir / "report.html"
            if rpt.exists():
                report_path = str(rpt)

    print()
    print("=" * 60)
    print("Overnight A/B Test FINISHED")
    print(f"Ended: {datetime.datetime.now():%Y-%m-%d %H:%M:%S}")
    print("=" * 60)
    if report_path:
        print(f"  Report: {report_path}")

    return {
        "experiment": experiment,
        "report_path": report_path,
        "replay_path": replay_path,
    }


# ---------------------------------------------------------------------------
# Batch state file management
# ---------------------------------------------------------------------------


def _batch_state_path(runs_dir: Path, config_path: Path) -> Path:
    """Return the path to the batch state file for a given config."""
    return runs_dir / f"_batch_state_{config_path.stem}.json"


def _load_batch_state(state_path: Path) -> dict[str, Any]:
    """Load an existing batch state file, or return an empty structure."""
    if state_path.exists():
        data: dict[str, Any] = json.loads(state_path.read_text())
        return data
    return {}


def _save_batch_state(state_path: Path, state: dict[str, Any]) -> None:
    """Atomically write the batch state file."""
    tmp = state_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2, default=str))
    tmp.replace(state_path)


def _init_batch_state(
    state_path: Path,
    config: dict[str, Any],
    *,
    fresh: bool = False,
) -> dict[str, Any]:
    """Create or resume a batch state file.

    If *fresh* is True, discard any existing state and start over.
    If the config hash has changed since the last run, raise an error
    unless *fresh* is True.
    """
    existing = {} if fresh else _load_batch_state(state_path)

    if existing:
        old_hash = existing.get("config_hash", "")
        new_hash = config["config_hash"]
        if old_hash and old_hash != new_hash and not fresh:
            raise RuntimeError(
                f"Batch config has changed since the last run "
                f"(hash {old_hash[:12]}... -> {new_hash[:12]}...). "
                f"Use --fresh to discard old state and start over."
            )

    # If we have existing state with matching experiments, reuse it
    if existing and not fresh:
        exp_states = {e["experiment"]: e for e in existing.get("experiments", [])}
        experiments_state = []
        for exp in config["experiments"]:
            name = exp["experiment"]
            prev = exp_states.get(name, {})
            if prev.get("status") == "completed":
                experiments_state.append(prev)
            else:
                experiments_state.append({"experiment": name, "status": "pending"})
        state = {
            "config_file": config["config_file"],
            "config_hash": config["config_hash"],
            "started_at": existing.get("started_at", datetime.datetime.now().isoformat()),
            "experiments": experiments_state,
        }
    else:
        state = {
            "config_file": config["config_file"],
            "config_hash": config["config_hash"],
            "started_at": datetime.datetime.now().isoformat(),
            "experiments": [
                {"experiment": e["experiment"], "status": "pending"} for e in config["experiments"]
            ],
        }

    _save_batch_state(state_path, state)
    return state


def _update_experiment_state(
    state_path: Path,
    state: dict[str, Any],
    idx: int,
    update: dict[str, Any],
) -> None:
    """Update one experiment entry in the state and persist."""
    state["experiments"][idx].update(update)
    _save_batch_state(state_path, state)


# ---------------------------------------------------------------------------
# Batch-wide pre-build
# ---------------------------------------------------------------------------


def _prebuild_all_binaries(
    config: dict[str, Any],
    runs_dir: Path,
) -> None:
    """Pre-build every binary before any experiment runs.

    Builds the baseline binary once (clean tree) and each experiment's
    iteration binary upfront, then distributes them to the per-experiment
    cache locations so that ``run_baseline()`` and ``_run_interleaved_body()``
    find cached binaries and skip their build phases entirely.

    This avoids cross-compilation happening between the thermal-warmup and
    measurement phases of each experiment.
    """
    from ab_harness.deploy import RemoteDeployment
    from ab_harness.runner import _GitContext

    experiments = config["experiments"]
    defaults = config["defaults"]

    # Nothing to do when every experiment opts out of building.
    if all(exp.get("no_build", False) for exp in experiments):
        log.info("Pre-build: all experiments have no_build=True, skipping.")
        return

    print("=" * 60)
    print("Pre-build: Building all binaries before experiments")
    print("=" * 60)

    # Deployer used only for local cross-compilation (build_only).
    deployer = RemoteDeployment(
        host=defaults.get("host", "blueos.local"),
        port=defaults.get("port", 6020),
        isolated=defaults.get("isolated", True),
    )

    cache_dir = runs_dir / ".batch_binaries"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. Build baseline binary (clean tree, once) -------------------------
    baseline_src = cache_dir / "mcm_baseline"
    if not baseline_src.exists():
        print("Pre-build: Building baseline binary (clean tree)...")
        deployer.build_only("baseline", dest_dir=cache_dir)
    else:
        print("Pre-build: Reusing cached baseline binary.")

    # --- 2. Build each experiment's iteration binary -------------------------
    for exp in experiments:
        exp_name = exp["experiment"]

        if exp.get("no_build", False):
            print(f"Pre-build: [{exp_name}] skipped (no_build)")
            continue

        iter_src = cache_dir / f"mcm_iter_{exp_name}"
        if iter_src.exists():
            print(f"Pre-build: [{exp_name}] reusing cached iteration binary.")
            continue

        diff_path = exp.get("diff")
        git_hash = exp.get("git_hash")
        github_pr = exp.get("github_pr")

        if not any([diff_path, git_hash, github_pr]):
            print(f"Pre-build: [{exp_name}] no source change, copying baseline.")
            shutil.copy2(baseline_src, iter_src)
            continue

        # _git() runs with cwd=MCM_REPO_ROOT, so relative diff paths must
        # be resolved to absolute (they are relative to the harness root).
        abs_diff = str(Path(diff_path).resolve()) if diff_path else None

        print(f"Pre-build: [{exp_name}] building iteration binary...")
        git_ctx = _GitContext(
            git_hash=git_hash,
            diff_path=abs_diff,
            github_pr=github_pr,
        )
        git_ctx.enter()
        try:
            # build_only("iteration") creates <cache_dir>/mcm_iteration;
            # rename to an experiment-specific name so the next experiment
            # does not short-circuit on the stale generic file.
            deployer.build_only("iteration", dest_dir=cache_dir)
            generic = cache_dir / "mcm_iteration"
            shutil.move(str(generic), str(iter_src))
        finally:
            git_ctx.exit()

    # --- 3. Distribute to per-experiment cache locations ----------------------
    for exp in experiments:
        exp_name = exp["experiment"]

        if exp.get("no_build", False):
            continue

        exp_dir = runs_dir / exp_name
        iter_src = cache_dir / f"mcm_iter_{exp_name}"

        # run_baseline() looks for <exp_dir>/baseline/mcm_baseline
        baseline_dst_dir = exp_dir / "baseline"
        baseline_dst_dir.mkdir(parents=True, exist_ok=True)
        baseline_dst = baseline_dst_dir / "mcm_baseline"
        if not baseline_dst.exists():
            shutil.copy2(baseline_src, baseline_dst)

        # _run_interleaved_body() looks for <exp_dir>/binaries/mcm_{iteration,baseline}
        binaries_dir = exp_dir / "binaries"
        binaries_dir.mkdir(parents=True, exist_ok=True)

        iter_dst = binaries_dir / "mcm_iteration"
        if not iter_dst.exists():
            shutil.copy2(iter_src, iter_dst)

        base_dst = binaries_dir / "mcm_baseline"
        if not base_dst.exists():
            shutil.copy2(baseline_src, base_dst)

    print(f"Pre-build: {len(experiments)} experiment(s) ready (cache: {cache_dir})")
    print()


# ---------------------------------------------------------------------------
# cmd_batch entry point
# ---------------------------------------------------------------------------


def cmd_batch(
    config_path: Path,
    runs_dir: Path,
    *,
    fresh: bool = False,
    no_restore: bool = False,
) -> None:
    """Run a series of independent A/B experiments from a YAML config file."""
    if not config_path.exists():
        print(f"ERROR: Config file not found: {config_path}")
        sys.exit(1)

    config = load_batch_config(config_path)
    runs_dir.mkdir(parents=True, exist_ok=True)

    state_path = _batch_state_path(runs_dir, config_path)
    state = _init_batch_state(state_path, config, fresh=fresh)

    experiments = config["experiments"]
    total = len(experiments)
    completed_count = sum(1 for e in state["experiments"] if e.get("status") == "completed")

    print("=" * 60)
    print(f"Batch A/B Test: {config_path.name}")
    print(f"Started: {datetime.datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"Experiments: {total} ({completed_count} already completed)")
    print(f"State file: {state_path}")
    print("=" * 60)
    print()

    # --- Pre-build all binaries so no compilation happens mid-experiment ------
    _prebuild_all_binaries(config, runs_dir)

    failed_experiments: list[str] = []

    for idx, exp_params in enumerate(experiments):
        exp_name = exp_params["experiment"]
        exp_state = state["experiments"][idx]

        if exp_state.get("status") == "completed":
            print(f"[{idx + 1}/{total}] {exp_name}: SKIPPED (already completed)")
            print()
            continue

        is_last = idx == total - 1
        # Only restore on the very last experiment (unless --no-restore)
        should_restore = is_last and not no_restore

        print(f"[{idx + 1}/{total}] {exp_name}: STARTING")
        print("-" * 60)

        # Inject runs_dir into params
        exp_params_with_dir = {**exp_params, "runs_dir": str(runs_dir)}

        _update_experiment_state(
            state_path,
            state,
            idx,
            {
                "status": "running",
                "started_at": datetime.datetime.now().isoformat(),
            },
        )

        t0 = time.monotonic()
        try:
            result = run_single_overnight(
                exp_params_with_dir,
                restore=should_restore,
            )
            elapsed = time.monotonic() - t0
            _update_experiment_state(
                state_path,
                state,
                idx,
                {
                    "status": "completed",
                    "elapsed_s": round(elapsed, 1),
                    "report": result.get("report_path"),
                },
            )
            print(f"[{idx + 1}/{total}] {exp_name}: COMPLETED in {elapsed:.0f}s")
        except Exception as exc:
            elapsed = time.monotonic() - t0
            tb = traceback.format_exc()
            log.error("Experiment '%s' failed after %.0fs: %s", exp_name, elapsed, exc)
            log.debug("Traceback:\n%s", tb)
            _update_experiment_state(
                state_path,
                state,
                idx,
                {
                    "status": "failed",
                    "elapsed_s": round(elapsed, 1),
                    "error": str(exc),
                },
            )
            failed_experiments.append(exp_name)
            print(f"[{idx + 1}/{total}] {exp_name}: FAILED ({exc})")

        print()

    # --- Final restore if a mid-batch failure left things dirty ---------------
    if not no_restore:
        # Check if the last experiment actually ran and restored
        last_state = state["experiments"][-1]
        if last_state.get("status") != "completed":
            # Last experiment failed or was skipped -- restore now
            try:
                host = experiments[0].get("host", "blueos.local")
                from ab_harness.deploy import RemoteDeployment

                deployer = RemoteDeployment(host=host, isolated=False)
                deployer._ssh(
                    f"docker stop {RemoteDeployment._ISOLATED_CONTAINER} 2>/dev/null || true",
                    timeout=30,
                )
                deployer._ssh(
                    f"docker rm {RemoteDeployment._ISOLATED_CONTAINER} 2>/dev/null || true",
                )
                deployer.restore_blueos()
                print("BlueOS restored after batch completion.")
            except Exception as exc:
                log.warning("Failed to restore BlueOS: %s", exc)

    # --- Generate batch summary report ----------------------------------------
    try:
        from ab_harness.report import generate_batch_summary

        summary_path = generate_batch_summary(state, runs_dir, config_path.stem)
        print(f"Batch summary: {summary_path}")
    except Exception as exc:
        log.warning("Could not generate batch summary: %s", exc)

    # --- Final banner ---------------------------------------------------------
    print()
    print("=" * 60)
    completed_final = sum(1 for e in state["experiments"] if e.get("status") == "completed")
    failed_final = sum(1 for e in state["experiments"] if e.get("status") == "failed")
    print(f"Batch FINISHED: {completed_final}/{total} completed, {failed_final} failed")
    print(f"Ended: {datetime.datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"State: {state_path}")
    print("=" * 60)

    if failed_experiments:
        print(f"\nFailed experiments: {', '.join(failed_experiments)}")
        print("Re-run the same command to retry failed experiments.")
        sys.exit(1)


def _default_runs_dir() -> Path:
    from ab_harness.config import HARNESS_ROOT

    return HARNESS_ROOT / "runs"
