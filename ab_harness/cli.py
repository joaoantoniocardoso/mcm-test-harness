"""CLI entry point: python -m ab_harness <subcommand> [options]."""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from ab_harness import __version__
from ab_harness.batch import run_single_overnight
from ab_harness.config import HARNESS_ROOT, ClientConfig, build_client_config

log = logging.getLogger("ab_harness")

DEFAULT_RUNS_DIR = HARNESS_ROOT / "runs"


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    """Add flags shared by baseline and iteration subcommands."""
    parser.add_argument(
        "--experiment",
        "-e",
        required=True,
        help="Experiment name (groups baseline + iterations together)",
    )
    parser.add_argument("--host", default="blueos.local", help="MCM host (default: blueos.local)")
    parser.add_argument("--port", type=int, default=6020, help="MCM HTTP port (default: 6020)")
    parser.add_argument(
        "--duration", type=int, default=60, help="Seconds per repetition (default: 60)"
    )
    parser.add_argument(
        "--repetitions", type=int, default=3, help="Number of collection reps (default: 3)"
    )
    parser.add_argument(
        "--warmup", type=int, default=5, help="Warmup seconds after reset (default: 5)"
    )
    parser.add_argument(
        "--deploy",
        choices=["local", "remote"],
        default="remote",
        help="Deploy mode (default: remote)",
    )
    parser.add_argument(
        "--env",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Environment variable override (repeatable)",
    )
    parser.add_argument(
        "--webrtc",
        action="append",
        default=[],
        metavar="PRODUCER[:COUNT]",
        help="WebRTC consumer: producer name substring and optional client count "
        "(repeatable, e.g. --webrtc RadCam:3 --webrtc 'UDP Stream':2)",
    )
    parser.add_argument(
        "--rtsp",
        action="append",
        default=[],
        metavar="PATH[:COUNT]",
        help="RTSP consumer: stream path and optional client count "
        "(repeatable, e.g. --rtsp video_stream_0:3)",
    )
    parser.add_argument(
        "--thumbnail",
        action="append",
        default=[],
        metavar="SOURCE[:COUNT]",
        help="Thumbnail probe: source name and optional parallel request count "
        "(repeatable, e.g. --thumbnail RadCam:2)",
    )
    parser.add_argument(
        "--thumbnail-interval",
        type=int,
        default=1,
        metavar="SECONDS",
        help="Seconds between thumbnail probes (default: 1)",
    )
    parser.add_argument("--extra-args", nargs="*", default=[], help="Extra MCM CLI args")
    parser.add_argument(
        "--isolated",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="[remote only] Stop all containers and run MCM in a clean isolated "
        "container (no other BlueOS services). Use --no-isolated to inject "
        "into the existing blueos-core container. "
        "(default: --isolated)",
    )
    parser.add_argument(
        "--blueos-tag",
        default=None,
        metavar="TAG",
        help="[remote + isolated only] Override the BlueOS Docker image tag "
        "for the isolated container (e.g. '1.4.3-beta.11'). "
        "By default the tag from the currently running container is used. "
        "Can also be set via BLUEOS_TAG env var.",
    )
    parser.add_argument(
        "--mcm-config",
        type=Path,
        default=None,
        metavar="PATH",
        help="[remote only] Path to a local MCM settings.json to upload "
        "into the container before starting MCM. The current config "
        "is always downloaded before and after each run for traceability.",
    )
    parser.add_argument(
        "--no-restore",
        action="store_true",
        default=False,
        help="[remote + isolated only] After tests, do NOT restart "
        "blueos-bootstrap to bring BlueOS back up. Useful if you plan "
        "to run more tests. Use 'restore' subcommand to bring BlueOS "
        "back up manually later.",
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=DEFAULT_RUNS_DIR,
        help=f"Base directory for run data (default: {DEFAULT_RUNS_DIR})",
    )


def _parse_env(env_list: list[str]) -> dict[str, str]:
    result = {}
    for item in env_list:
        if "=" not in item:
            log.warning("Ignoring malformed --env value: %s", item)
            continue
        k, v = item.split("=", 1)
        result[k] = v
    return result


def _build_client_config(args: argparse.Namespace) -> ClientConfig:
    """Build a ClientConfig from parsed CLI arguments."""
    return build_client_config(
        webrtc=args.webrtc,
        rtsp=args.rtsp,
        thumbnails=args.thumbnail,
        thumbnail_interval=args.thumbnail_interval,
    )


def cmd_baseline(args: argparse.Namespace) -> None:
    from ab_harness.runner import run_baseline

    run_baseline(
        experiment=args.experiment,
        host=args.host,
        port=args.port,
        duration=args.duration,
        repetitions=args.repetitions,
        warmup=args.warmup,
        deploy_mode=args.deploy,
        env_overrides=_parse_env(args.env),
        client_config=_build_client_config(args),
        extra_args=args.extra_args,
        runs_dir=args.runs_dir,
        isolated=args.isolated,
        restore=not args.no_restore,
        blueos_tag=args.blueos_tag,
        mcm_config=args.mcm_config,
    )


def cmd_iteration(args: argparse.Namespace) -> None:
    from ab_harness.runner import run_iteration

    run_iteration(
        experiment=args.experiment,
        label=args.label,
        host=args.host,
        port=args.port,
        duration=args.duration,
        repetitions=args.repetitions,
        warmup=args.warmup,
        deploy_mode=args.deploy,
        env_overrides=_parse_env(args.env),
        client_config=_build_client_config(args),
        extra_args=args.extra_args,
        runs_dir=args.runs_dir,
        git_hash=args.git_hash,
        diff_path=args.diff,
        github_pr=args.github_pr,
        sequential=args.sequential,
        adaptive=args.adaptive,
        adaptive_method=args.adaptive_method,
        adaptive_look_every=args.adaptive_look_every,
        adaptive_kpis=args.adaptive_kpis,
        adaptive_no_futility=args.adaptive_no_futility,
        no_build=args.no_build,
        isolated=args.isolated,
        restore=not args.no_restore,
        blueos_tag=args.blueos_tag,
        mcm_config=args.mcm_config,
        strict=args.strict,
        seed=args.seed,
    )


def cmd_report(args: argparse.Namespace) -> None:
    from ab_harness.runner import regenerate_report

    regenerate_report(
        experiment=args.experiment,
        iteration=args.iteration,
        runs_dir=args.runs_dir,
    )


def cmd_compare(args: argparse.Namespace) -> None:
    from ab_harness.compare import compare_runs, save_comparison
    from ab_harness.report import generate_report

    run_a = Path(args.run_a)
    run_b = Path(args.run_b)

    comparison = compare_runs(run_a, run_b, strict=args.strict)
    save_comparison(comparison, run_a / "comparison.json")
    generate_report(comparison, run_a, run_b)
    print(f"Report: {run_a / 'report.html'}")


def cmd_replay(args: argparse.Namespace) -> None:
    """Replay collected data through all sequential testing methods."""
    import json

    from ab_harness.kpi import load_aggregate
    from ab_harness.sequential_testing import (
        format_replay_summary,
        replay_all_methods,
        serialize_replay_results,
    )

    exp_dir = args.runs_dir / args.experiment
    baseline_dir = exp_dir / "baseline"

    # Find iteration dir
    if args.iteration:
        iteration_dir = exp_dir / "iterations" / args.iteration
    else:
        # Use the latest iteration
        iterations_dir = exp_dir / "iterations"
        if not iterations_dir.exists():
            print("ERROR: No iterations found")
            sys.exit(1)
        latest = sorted(iterations_dir.iterdir())[-1]
        iteration_dir = latest
        print(f"Using latest iteration: {iteration_dir.name}")

    # Determine baseline source (prefer interleaved)
    interleaved_dir = exp_dir / "baseline_interleaved"
    if interleaved_dir.exists() and (interleaved_dir / "aggregate.json").exists():
        b_source = interleaved_dir
        print(f"Using interleaved baseline: {b_source}")
    else:
        b_source = baseline_dir
        print(f"Using original baseline: {b_source}")

    # Load aggregates
    agg_a = load_aggregate(iteration_dir)
    agg_b = load_aggregate(b_source)

    a_data = agg_a.get("aggregate", {})
    b_data = agg_b.get("aggregate", {})

    # Determine KPIs to replay
    kpis = args.kpis or ["system_cpu_pct"]

    all_results = []
    for kpi_name in kpis:
        a_per_rep = a_data.get(kpi_name, {}).get("per_rep_means", [])
        b_per_rep = b_data.get(kpi_name, {}).get("per_rep_means", [])

        if not a_per_rep or not b_per_rep:
            print(f"WARNING: No per-rep data for {kpi_name}, skipping")
            continue

        result = replay_all_methods(
            a_per_rep=a_per_rep,
            b_per_rep=b_per_rep,
            kpi_name=kpi_name,
            alpha=args.alpha,
        )
        all_results.append(result)
        print(format_replay_summary(result))

    # Save detailed results
    if all_results:
        output_path = iteration_dir / "sequential_replay.json"
        output_path.write_text(
            json.dumps(serialize_replay_results(all_results), indent=2, default=str)
        )
        print(f"\nDetailed results saved to: {output_path}")

        # Save full trajectories separately (can be large)
        traj_path = iteration_dir / "sequential_replay_trajectories.json"
        traj_data = [
            {
                "kpi_name": r.kpi_name,
                "gst_trajectory": r.gst.get("trajectory", []),
                "msprt_trajectory": r.msprt.get("trajectory", []),
            }
            for r in all_results
        ]
        traj_path.write_text(json.dumps(traj_data, indent=2, default=str))
        print(f"Trajectories saved to: {traj_path}")


def cmd_recommend(args: argparse.Namespace) -> None:
    """Recommend sample sizes based on prior data or user-supplied CV."""
    import json

    from ab_harness.compare import recommend_sample_size
    from ab_harness.kpi import KPI_REGISTRY

    mde_list = args.min_effect  # list of floats
    target_power = args.power
    alpha = 0.05

    # --- Auto-run baseline if requested -----------------------------------
    if args.run_baseline:
        if not args.experiment:
            print("ERROR: --run-baseline requires --experiment")
            sys.exit(1)
        exp_dir = args.runs_dir / args.experiment
        baseline_agg_path = exp_dir / "baseline" / "aggregate.json"
        if baseline_agg_path.exists():
            print(f"Baseline already exists for {args.experiment}, skipping collection.\n")
        else:
            from ab_harness.runner import run_baseline

            print(f"Running baseline ({args.repetitions} reps x {args.duration}s)...\n")
            run_baseline(
                experiment=args.experiment,
                host=args.host,
                port=args.port,
                duration=args.duration,
                repetitions=args.repetitions,
                warmup=args.warmup,
                deploy_mode=args.deploy,
                env_overrides={},
                client_config=build_client_config(
                    webrtc=args.webrtc,
                    rtsp=args.rtsp,
                    thumbnails=args.thumbnail,
                    thumbnail_interval=args.thumbnail_interval,
                ),
                extra_args=[],
                runs_dir=args.runs_dir,
                isolated=args.isolated,
                restore=not args.no_restore,
            )
            print()

    if args.experiment:
        # Load existing aggregate data
        exp_dir = args.runs_dir / args.experiment
        baseline_agg_path = exp_dir / "baseline" / "aggregate.json"
        if not baseline_agg_path.exists():
            print(f"ERROR: No baseline aggregate found at {baseline_agg_path}")
            print("Hint: run with --run-baseline to collect one automatically.")
            sys.exit(1)
        agg = json.loads(baseline_agg_path.read_text())
        agg_data = agg.get("aggregate", {})

        # Build header
        mde_headers = "".join(f"  {m:>5g}%" for m in mde_list)
        robust_tag = " [robust/MAD]" if args.robust else ""
        print(f"\nSample-size recommendations (power={target_power}, alpha={alpha}){robust_tag}")
        print(f"Based on: {args.experiment}/baseline\n")
        print(f"{'KPI':<28} {'Std':>8} {'Mean':>8} {'CV%':>7}{mde_headers}")
        print("-" * (55 + 8 * len(mde_list)))

        for kpi_def in KPI_REGISTRY:
            kpi_data = agg_data.get(kpi_def.name, {})
            mean = kpi_data.get("mean")
            std = kpi_data.get("std")
            if mean is not None and std is not None and mean != 0:
                cv_pct = abs(std / mean) * 100
                reps_cols = ""
                for mde in mde_list:
                    per_rep = kpi_data.get("per_rep_means", [])
                    n = recommend_sample_size(
                        std=std,
                        mde_pct=mde,
                        baseline_mean=mean,
                        alpha=alpha,
                        target_power=target_power,
                        robust=args.robust,
                        per_rep_means=per_rep if args.robust else None,
                    )
                    reps_cols += f"  {n:>6d}"
                print(f"{kpi_def.name:<28} {std:>8.3f} {mean:>8.3f} {cv_pct:>6.1f}%{reps_cols}")
            else:
                na_cols = "".join(f"  {'N/A':>6}" for _ in mde_list)
                print(f"{kpi_def.name:<28} {'N/A':>8} {'N/A':>8} {'N/A':>7}{na_cols}")

        print("\nColumns show reps per side needed to detect each MDE.")

    elif args.cv is not None:
        # User-supplied coefficient of variation
        cv = args.cv / 100.0
        from scipy.stats import norm

        z_alpha = norm.ppf(1.0 - alpha / 2)
        z_beta = norm.ppf(target_power)
        import math

        print(f"\nWith CV={args.cv}%, power={target_power}, alpha={alpha}:")
        for mde in mde_list:
            mde_frac = mde / 100.0
            if cv > 0 and mde_frac > 0:
                n = max(2, math.ceil(2.0 * ((z_alpha + z_beta) / (mde_frac / cv)) ** 2))
            else:
                n = 2
            print(f"  MDE={mde}% -> {n} reps per side")
    else:
        print("ERROR: Provide either --experiment (with optional --run-baseline) or --cv")
        sys.exit(1)


def cmd_overnight(args: argparse.Namespace) -> None:
    """Run a full overnight A/B test: bootstrap baseline + adaptive iteration + replay."""
    params: dict[str, Any] = {
        "experiment": args.experiment,
        "host": args.host,
        "port": args.port,
        "duration": args.duration,
        "repetitions": args.repetitions,
        "warmup": args.warmup,
        "deploy": args.deploy,
        "env": _parse_env(args.env),
        "webrtc": args.webrtc,
        "rtsp": args.rtsp,
        "thumbnails": args.thumbnail,
        "thumbnail_interval": args.thumbnail_interval,
        "extra_args": args.extra_args,
        "runs_dir": str(args.runs_dir),
        "isolated": args.isolated,
        "no_build": args.no_build,
        "adaptive_method": args.adaptive_method,
        "adaptive_look_every": args.adaptive_look_every,
        "adaptive_kpis": args.adaptive_kpis,
        "adaptive_no_futility": args.adaptive_no_futility,
        "git_hash": args.git_hash,
        "diff": args.diff,
        "github_pr": args.github_pr,
        "label": args.label,
        "blueos_tag": args.blueos_tag,
        "mcm_config": str(args.mcm_config) if args.mcm_config else None,
        "strict": args.strict,
    }
    run_single_overnight(params, restore=not args.no_restore)


def _cmd_batch(args: argparse.Namespace) -> None:
    """Run a series of independent A/B experiments from a YAML config file."""
    from ab_harness.batch import cmd_batch as _batch_impl

    _batch_impl(
        config_path=Path(args.config).resolve(),
        runs_dir=args.runs_dir,
        fresh=args.fresh,
        no_restore=args.no_restore,
    )


def cmd_check(args: argparse.Namespace) -> None:
    """Verify that the environment is ready for running tests."""
    import importlib
    import shutil

    ok_mark = "  [OK]   "
    warn_mark = "  [WARN] "
    fail_mark = "  [FAIL] "
    all_ok = True

    print("A/B Test Harness -- Environment Check")
    print("=" * 50)

    # 1. Python version
    v = sys.version_info
    if v >= (3, 10):
        print(f"{ok_mark}Python {v.major}.{v.minor}.{v.micro}")
    else:
        print(f"{fail_mark}Python {v.major}.{v.minor}.{v.micro} (need >= 3.10)")
        all_ok = False

    # 2. Required packages
    required = [
        ("matplotlib", "matplotlib"),
        ("pandas", "pandas"),
        ("numpy", "numpy"),
        ("jinja2", "jinja2"),
        ("scipy", "scipy"),
        ("spotify_confidence", "spotify-confidence"),
        ("msprt", "msprt"),
    ]
    for mod_name, pip_name in required:
        try:
            importlib.import_module(mod_name)
            print(f"{ok_mark}{pip_name}")
        except ImportError:
            print(f"{fail_mark}{pip_name} (pip install {pip_name})")
            all_ok = False

    # 3. Optional: playwright
    try:
        importlib.import_module("playwright")
        print(f"{ok_mark}playwright (optional, for --webrtc)")
    except ImportError:
        print(f"{warn_mark}playwright not installed (optional, for --webrtc)")
        print("         pip install playwright && playwright install chromium")

    # 4. System tools
    for tool, purpose, required_flag in [
        ("cross", "cross-compilation (remote mode)", True),
        ("sshpass", "SSH password auth (remote mode)", True),
        ("gst-launch-1.0", "RTSP client (optional)", False),
    ]:
        if shutil.which(tool):
            print(f"{ok_mark}{tool}")
        elif required_flag:
            print(f"{fail_mark}{tool} not found ({purpose})")
            all_ok = False
        else:
            print(f"{warn_mark}{tool} not found ({purpose})")

    # 5. CROSS_TARGET
    import os

    target = os.environ.get("CROSS_TARGET", "armv7-unknown-linux-gnueabihf")
    print(f"{ok_mark}CROSS_TARGET={target}")

    # 6. SSH connectivity (if --host given)
    host = getattr(args, "host", None)
    if host:
        from ab_harness.config import SSH_PORT, SSH_PWD, SSH_USER

        print()
        print(f"Remote connectivity ({host}):")
        print("-" * 50)
        # SSH
        ssh_user = SSH_USER
        ssh_pwd = SSH_PWD
        ssh_port = str(SSH_PORT)
        try:
            import subprocess

            result = subprocess.run(
                [
                    "sshpass",
                    "-p",
                    ssh_pwd,
                    "ssh",
                    "-o",
                    "StrictHostKeyChecking=no",
                    "-o",
                    "ConnectTimeout=5",
                    "-p",
                    ssh_port,
                    f"{ssh_user}@{host}",
                    "echo ok",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                print(f"{ok_mark}SSH to {ssh_user}@{host}:{ssh_port}")
            else:
                print(f"{fail_mark}SSH to {ssh_user}@{host}:{ssh_port}: {result.stderr.strip()}")
                all_ok = False
        except FileNotFoundError:
            print(f"{fail_mark}sshpass not found, cannot test SSH")
            all_ok = False
        except Exception as e:
            print(f"{fail_mark}SSH: {e}")
            all_ok = False

        # MCM API
        port = getattr(args, "port", 6020)
        api_url = f"http://{host}:{port}/stats/pipeline-analysis"
        try:
            import urllib.request

            req = urllib.request.Request(api_url, method="GET")
            with urllib.request.urlopen(req, timeout=5) as resp:
                if resp.status == 200:
                    print(f"{ok_mark}MCM API at {host}:{port}")
                else:
                    print(f"{warn_mark}MCM API returned status {resp.status}")
        except Exception:
            print(f"{warn_mark}MCM API not reachable at {host}:{port} (MCM may not be running)")

    print()
    if all_ok:
        print("All required checks passed.")
    else:
        print("Some required checks FAILED. Fix the issues above before running tests.")
        sys.exit(1)


def cmd_restore(args: argparse.Namespace) -> None:
    """Restart blueos-bootstrap to bring the full BlueOS stack back up."""
    from ab_harness.deploy import RemoteDeployment

    deployer = RemoteDeployment(host=args.host, isolated=False)
    # Clean up leftover isolated container if present
    deployer._ssh(
        f"docker stop {RemoteDeployment._ISOLATED_CONTAINER} 2>/dev/null || true",
        timeout=30,
    )
    deployer._ssh(
        f"docker rm {RemoteDeployment._ISOLATED_CONTAINER} 2>/dev/null || true",
    )
    deployer.restore_blueos()
    print(f"BlueOS restore triggered on {args.host}. It will be up shortly.")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="ab_harness",
        description="A/B Test Harness for MCM Pipeline Analysis Stats",
    )
    parser.add_argument(
        "-V",
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- baseline ---
    p_base = sub.add_parser("baseline", help="Run the baseline (B) measurement")
    _add_common_args(p_base)
    p_base.set_defaults(func=cmd_baseline)

    # --- iteration ---
    p_iter = sub.add_parser("iteration", help="Run an iteration (A) measurement")
    _add_common_args(p_iter)
    p_iter.add_argument(
        "--label", "-l", default=None, help="Iteration label (auto-derived if omitted)"
    )
    p_iter.add_argument(
        "--sequential",
        action="store_true",
        default=False,
        help="Disable interleaved ABBA execution; collect all reps sequentially (faster but less robust)",
    )
    # Adaptive stopping
    p_iter.add_argument(
        "--adaptive",
        action="store_true",
        default=False,
        help="Enable adaptive stopping (sequential testing to stop early). "
        "--repetitions becomes the maximum number of reps per side.",
    )
    p_iter.add_argument(
        "--adaptive-method",
        choices=["msprt", "gst"],
        default="msprt",
        help="Adaptive stopping method: 'msprt' (mixture SPRT, checks every trial, "
        "default) or 'gst' (Group Sequential Testing, checks at scheduled intervals)",
    )
    p_iter.add_argument(
        "--adaptive-look-every",
        type=int,
        default=0,
        help="[GST only] Run interim analysis every N reps per side (default: auto-select ~8 looks)",
    )
    p_iter.add_argument(
        "--adaptive-kpis",
        nargs="+",
        default=None,
        help="KPI names to monitor for adaptive stopping (default: system_cpu_pct)",
    )
    p_iter.add_argument(
        "--adaptive-no-futility",
        action="store_true",
        default=False,
        help="Disable futility stopping (only stop for efficacy)",
    )
    p_iter.add_argument(
        "--no-build",
        action="store_true",
        default=False,
        help="Skip cross-compilation; reuse binaries cached in <exp>/binaries/. "
        "Useful for retrying a crashed experiment without rebuilding.",
    )
    p_iter.add_argument(
        "--strict",
        action="store_true",
        default=False,
        help="Apply Holm-Bonferroni multiple comparisons correction across all KPIs, "
        "controlling the family-wise error rate (FWER) at alpha=0.05.",
    )
    p_iter.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for trial ordering (default: true random). "
        "Enables reproducible interleaved experiments.",
    )
    source = p_iter.add_mutually_exclusive_group()
    source.add_argument("--git-hash", default=None, help="Git commit hash to checkout and test")
    source.add_argument("--diff", default=None, help="Path to a .patch/.diff file to apply")
    source.add_argument("--github-pr", default=None, help="GitHub PR number or URL to fetch")
    p_iter.set_defaults(func=cmd_iteration)

    # --- report ---
    p_report = sub.add_parser("report", help="Regenerate report for an existing iteration")
    p_report.add_argument("--experiment", "-e", required=True)
    p_report.add_argument(
        "--iteration", "-i", required=True, help="Iteration dir name (e.g., 001_my_label)"
    )
    p_report.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    p_report.set_defaults(func=cmd_report)

    # --- compare ---
    p_compare = sub.add_parser("compare", help="Compare two arbitrary runs")
    p_compare.add_argument("--run-a", required=True, help="Path to run A directory")
    p_compare.add_argument("--run-b", required=True, help="Path to run B directory")
    p_compare.add_argument(
        "--strict",
        action="store_true",
        default=False,
        help="Apply Holm-Bonferroni multiple comparisons correction across all KPIs.",
    )
    p_compare.set_defaults(func=cmd_compare)

    # --- replay ---
    p_replay = sub.add_parser(
        "replay",
        help="Replay collected data through sequential testing methods "
        "(GST via spotify-confidence, mSPRT)",
    )
    p_replay.add_argument("--experiment", "-e", required=True)
    p_replay.add_argument(
        "--iteration",
        "-i",
        default=None,
        help="Iteration dir name (e.g., 001_my_label). Defaults to latest.",
    )
    p_replay.add_argument(
        "--kpis",
        nargs="+",
        default=None,
        help="KPI names to replay (default: system_cpu_pct)",
    )
    p_replay.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance level (default: 0.05)",
    )
    p_replay.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    p_replay.set_defaults(func=cmd_replay)

    # --- recommend ---
    p_recommend = sub.add_parser(
        "recommend",
        help="Recommend sample size for a target minimum detectable effect (MDE). "
        "Use --run-baseline to automatically collect a baseline first.",
    )
    p_recommend.add_argument(
        "--experiment",
        "-e",
        default=None,
        help="Experiment name with existing baseline data (used to estimate variance)",
    )
    p_recommend.add_argument(
        "--cv",
        type=float,
        default=None,
        help="Coefficient of variation (%%) if no prior data available",
    )
    p_recommend.add_argument(
        "--min-effect",
        type=float,
        nargs="+",
        default=[5.0],
        help="Minimum detectable effect(s) as %% of baseline mean "
        "(default: 5; pass multiple e.g. --min-effect 2 5 10)",
    )
    p_recommend.add_argument(
        "--power",
        type=float,
        default=0.80,
        help="Target statistical power (default: 0.80)",
    )
    p_recommend.add_argument(
        "--robust",
        action="store_true",
        default=False,
        help="Use MAD-based (robust) standard deviation instead of sample std "
        "for sample-size estimation. More resistant to outliers in baseline data.",
    )
    p_recommend.add_argument(
        "--runs-dir",
        type=Path,
        default=DEFAULT_RUNS_DIR,
        help=f"Base directory for run data (default: {DEFAULT_RUNS_DIR})",
    )
    # Auto-baseline flags
    p_recommend.add_argument(
        "--run-baseline",
        action="store_true",
        default=False,
        help="Run a baseline first (if one does not exist) before computing "
        "recommendations. Requires --experiment.",
    )
    p_recommend.add_argument("--host", default="blueos.local", help="MCM host")
    p_recommend.add_argument("--port", type=int, default=6020, help="MCM HTTP port")
    p_recommend.add_argument(
        "--deploy",
        choices=["local", "remote"],
        default="remote",
        help="Deploy mode (default: remote)",
    )
    p_recommend.add_argument(
        "--duration", type=int, default=30, help="Seconds per baseline rep (default: 30)"
    )
    p_recommend.add_argument(
        "--repetitions", type=int, default=5, help="Baseline reps to collect (default: 5)"
    )
    p_recommend.add_argument("--warmup", type=int, default=5, help="Warmup seconds (default: 5)")
    p_recommend.add_argument("--webrtc", action="append", default=[])
    p_recommend.add_argument("--rtsp", action="append", default=[])
    p_recommend.add_argument("--thumbnail", action="append", default=[])
    p_recommend.add_argument("--thumbnail-interval", type=int, default=1)
    p_recommend.add_argument(
        "--isolated",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p_recommend.add_argument("--no-restore", action="store_true", default=False)
    p_recommend.set_defaults(func=cmd_recommend)

    # --- overnight ---
    p_overnight = sub.add_parser(
        "overnight",
        help="Full overnight A/B test: bootstrap baseline + adaptive iteration + replay",
    )
    _add_common_args(p_overnight)
    p_overnight.add_argument("--label", "-l", default=None, help="Iteration label")
    p_overnight.add_argument(
        "--adaptive-method",
        choices=["msprt", "gst"],
        default="gst",
        help="Adaptive stopping method (default: gst)",
    )
    p_overnight.add_argument(
        "--adaptive-look-every",
        type=int,
        default=0,
        help="[GST only] Interim analysis every N reps per side (default: auto)",
    )
    p_overnight.add_argument(
        "--adaptive-kpis",
        nargs="+",
        default=None,
        help="KPI names for adaptive stopping (default: system_cpu_pct)",
    )
    p_overnight.add_argument(
        "--adaptive-no-futility",
        action="store_true",
        default=False,
        help="Disable futility stopping",
    )
    p_overnight.add_argument(
        "--no-build",
        action="store_true",
        default=False,
        help="Skip cross-compilation; reuse cached binaries",
    )
    p_overnight.add_argument(
        "--strict",
        action="store_true",
        default=False,
        help="Apply Holm-Bonferroni multiple comparisons correction across all KPIs.",
    )
    source = p_overnight.add_mutually_exclusive_group()
    source.add_argument("--git-hash", default=None, help="Git commit hash to test")
    source.add_argument("--diff", default=None, help="Path to a .patch/.diff file")
    source.add_argument("--github-pr", default=None, help="GitHub PR number or URL")
    p_overnight.set_defaults(func=cmd_overnight)

    # --- batch ---
    p_batch = sub.add_parser(
        "batch",
        help="Run a series of independent A/B experiments from a YAML config file",
    )
    p_batch.add_argument(
        "--config",
        "-c",
        required=True,
        help="Path to the batch YAML config file",
    )
    p_batch.add_argument(
        "--runs-dir",
        type=Path,
        default=DEFAULT_RUNS_DIR,
        help=f"Base directory for run data (default: {DEFAULT_RUNS_DIR})",
    )
    p_batch.add_argument(
        "--fresh",
        action="store_true",
        default=False,
        help="Ignore existing state file and start all experiments from scratch",
    )
    p_batch.add_argument(
        "--no-restore",
        action="store_true",
        default=False,
        help="Do NOT restore BlueOS after the last experiment",
    )
    p_batch.set_defaults(func=_cmd_batch)

    # --- restore ---
    p_restore = sub.add_parser(
        "restore",
        help="Restart blueos-bootstrap on the Pi to bring BlueOS back up "
        "(useful after a crashed test or --no-restore run)",
    )
    p_restore.add_argument(
        "--host",
        default="blueos.local",
        help="Pi host address (default: blueos.local)",
    )
    p_restore.set_defaults(func=cmd_restore)

    # --- check ---
    p_check = sub.add_parser(
        "check",
        help="Verify that the environment is ready for running tests",
    )
    p_check.add_argument(
        "--host",
        default=None,
        help="Test SSH and API connectivity to this host (optional)",
    )
    p_check.add_argument(
        "--port",
        type=int,
        default=6020,
        help="MCM HTTP port to check (default: 6020)",
    )
    p_check.set_defaults(func=cmd_check)

    args = parser.parse_args()

    # Configure logging
    level = logging.DEBUG if args.verbose else logging.INFO
    log_fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    logging.basicConfig(
        level=level,
        format=log_fmt,
        datefmt="%H:%M:%S",
    )

    # Add a timestamped file handler under the runs directory
    runs_dir = getattr(args, "runs_dir", DEFAULT_RUNS_DIR)
    runs_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = runs_dir / f"harness_{timestamp}.log"
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(log_fmt, datefmt="%Y-%m-%d %H:%M:%S"))
    logging.getLogger().addHandler(fh)
    log.info("Logging to %s", log_file)

    try:
        args.func(args)
    except KeyboardInterrupt:
        log.info("Interrupted by user.")
        sys.exit(130)
    except Exception as exc:
        log.error("Fatal error: %s", exc, exc_info=True)
        sys.exit(1)
