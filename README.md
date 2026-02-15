# A/B Test Harness for MCM Pipeline Analysis

Measure the impact of code changes on stream quality and system performance by
running controlled A/B tests against the MCM pipeline analysis stats API.

This is a standalone test harness for
[mavlink-camera-manager](https://github.com/bluerobotics/mavlink-camera-manager).
It expects the MCM repository to be cloned alongside this one (or configured via
the `MCM_REPO_ROOT` environment variable).

## How it works

1. **Baseline ("B")**: Build and deploy the current code, collect stats for N
   repetitions of M seconds each.
2. **Iteration ("A")**: Apply a change (code edit, git hash, diff, or GitHub PR),
   build, deploy, and collect the same stats.
3. **Compare**: Aggregate KPIs across repetitions, run Welch's t-test, flag
   regressions and improvements, and generate a self-contained HTML report.

```
            baseline (3 reps x 60s)          iteration (3 reps x 60s)
            ┌──────┐┌──────┐┌──────┐        ┌──────┐┌──────┐┌──────┐
  deploy -> │rep 1 ││rep 2 ││rep 3 │  edit ->│rep 1 ││rep 2 ││rep 3 │-> report.html
            └──────┘└──────┘└──────┘        └──────┘└──────┘└──────┘
                     |                               |
                 aggregate.json                  aggregate.json
                                 \              /
                                  t-test + compare
```

## Directory Layout

```
mavlink-camera-manager-test-harness/   (this repo)
├── ab_harness/                        # Python package
│   ├── __init__.py
│   ├── __main__.py
│   ├── cli.py
│   ├── clients.py
│   ├── collector.py
│   ├── compare.py
│   ├── config.py
│   ├── deploy.py
│   ├── kpi.py
│   ├── metadata.py
│   ├── report.py
│   ├── runner.py
│   ├── sequential_testing.py
│   └── webrtc_client.py
├── libs/
│   └── msprt/                         # mSPRT implementation
│       ├── pyproject.toml
│       ├── src/msprt/
│       │   ├── __init__.py
│       │   └── core.py
│       └── tests/
│           └── test_core.py
├── runs/                              # Test run data (gitignored)
├── requirements.txt
└── README.md

../mavlink-camera-manager/             # MCM repo (sibling directory)
```

## Prerequisites

- **Python 3.10+**
- **Remote mode** (default): `sshpass`, SSH access to the BlueOS device, and
  `cross` installed for ARM cross-compilation
- **Local mode**: GStreamer dev libraries, `cargo`
- **Optional**: `playwright` + Chromium (for `--webrtc`), `gst-launch-1.0`
  (for `--rtsp`)

## Setup

```bash
# 1. Clone both repos side by side
git clone https://github.com/bluerobotics/mavlink-camera-manager.git
git clone https://github.com/bluerobotics/mavlink-camera-manager-test-harness.git

# 2. Set up the test harness
cd mavlink-camera-manager-test-harness
python3 -m venv .venv
source .venv/bin/activate

# 3. Install Python dependencies
pip install -r requirements.txt

# 4. (Optional) Install Playwright for WebRTC testing
pip install playwright
playwright install chromium

# 5. Install cross for ARM cross-compilation (remote mode)
cargo install cross --git https://github.com/cross-rs/cross

# 6. Install sshpass for SSH automation (remote mode)
#    Arch:   sudo pacman -S sshpass
#    Debian: sudo apt install sshpass
#    macOS:  brew install hudochenkov/sshpass/sshpass

# 7. Verify everything is ready
python -m ab_harness check
# With remote connectivity test:
python -m ab_harness check --host 192.168.2.2
```

### MCM Repository Location

By default, the harness expects the MCM repo at `../mavlink-camera-manager`
(a sibling directory). Override this with the `MCM_REPO_ROOT` environment variable:

```bash
export MCM_REPO_ROOT=/path/to/mavlink-camera-manager
```

### Non-Pi 4 boards

The default cross-compilation target is `armv7-unknown-linux-gnueabihf` (Pi 4).
For other architectures (e.g., Pi 5 aarch64), set:

```bash
export CROSS_TARGET=aarch64-unknown-linux-gnu
```

## Configuration

All settings have sensible defaults but can be overridden via environment
variables or CLI flags:

| Variable | Default | Description |
|----------|---------|-------------|
| `MCM_REPO_ROOT` | `../mavlink-camera-manager` | Path to the MCM repository |
| `CROSS_TARGET` | `armv7-unknown-linux-gnueabihf` | Rust cross-compilation target triple |
| `SSH_USER` | `pi` | SSH username for the BlueOS device |
| `SSH_PWD` | `raspberry` | SSH password |
| `SSH_PORT` | `22` | SSH port |
| `BLUEOS_CONTAINER` | `blueos-core` | Docker container name on the device |

## Quick Start

```bash
# From the test harness repo root:

# 1. Run baseline (current code, 3 reps x 60s)
python -m ab_harness baseline \
  --experiment rtspsrc_tuning \
  --host blueos.local \
  --deploy remote

# 2. Make your code change in the MCM repo, then run an iteration
python -m ab_harness iteration \
  --experiment rtspsrc_tuning \
  --label "latency_200ms" \
  --host blueos.local \
  --deploy remote \
  --env MCM_RTSPSRC_LATENCY=200

# 3. Open the report
xdg-open runs/rtspsrc_tuning/iterations/001_latency_200ms/report.html
```

## CLI Reference

### `baseline` -- Run the baseline measurement

```
python -m ab_harness baseline --experiment NAME [options]
```

### `iteration` -- Run an iteration measurement

```
python -m ab_harness iteration --experiment NAME [--label LABEL] [options]
```

Iteration-specific source flags (mutually exclusive):

| Flag | Description |
|------|-------------|
| `--git-hash HASH` | Checkout a specific commit, build, test, then restore |
| `--diff PATH` | Apply a `.patch`/`.diff` file, build, test, then revert |
| `--github-pr NUMBER` | Fetch a GitHub PR via `gh pr checkout`, test, then restore |

### `report` -- Regenerate a report

```
python -m ab_harness report --experiment NAME --iteration DIR_NAME
```

### `compare` -- Compare two arbitrary runs

```
python -m ab_harness compare --run-a PATH --run-b PATH
```

### `check` -- Verify environment readiness

```
python -m ab_harness check [--host HOST] [--port PORT]
```

Runs a diagnostic checklist: Python version, required packages, system tools
(`cross`, `sshpass`, `gst-launch-1.0`, `playwright`), and optionally SSH + API
connectivity to a remote host.

### `recommend` -- How many reps do I need?

```
python -m ab_harness recommend --experiment NAME [--min-effect 2 5 10]
```

Prints a per-KPI table showing how many reps per side are needed to detect
each minimum detectable effect (MDE) with the given statistical power.

Use `--run-baseline` to automatically collect a baseline first if one doesn't
exist:

```bash
# One command: collect 5 baseline reps, then print reps needed for 2%, 5%, 10%
python -m ab_harness recommend \
  --experiment my_test \
  --run-baseline --host 192.168.2.2 --deploy remote \
  --duration 30 --repetitions 5 \
  --min-effect 2 5 10
```

Output:

```
Sample-size recommendations (power=0.8, alpha=0.05)
Based on: my_test/baseline

KPI                             Std     Mean    CV%     2%     5%    10%
-----------------------------------------------------------------------
throughput_fps                 0.012   30.000   0.0%      2      2      2
system_cpu_pct                 1.234   48.500   2.5%     20      4      2
freshness_delay_ms             0.450    5.200   8.7%    120     20      6

Columns show reps per side needed to detect each MDE.
```

Without prior data, you can estimate from a coefficient of variation:

```bash
python -m ab_harness recommend --cv 10 --min-effect 2 5 10
```

### Adaptive stopping

When `--adaptive` is passed with `iteration`, the harness uses sequential testing
to stop early once a statistically significant result (or clear futility) is
detected. `--repetitions` becomes the *maximum* number of reps per side.

Two methods are available via `--adaptive-method`:

| Method | Flag | Description |
|--------|------|-------------|
| **mSPRT** | `--adaptive-method msprt` | Mixture Sequential Probability Ratio Test. Checks after every completed A/B pair with zero statistical penalty. Default for `iterate` (quick interactive runs). |
| **GST** (overnight default) | `--adaptive-method gst` | Group Sequential Testing (Lan-DeMets alpha-spending). Checks at pre-scheduled intervals (`--adaptive-look-every`). Higher power when max sample size is known. Default for `overnight` / `batch` runs. |

Additional adaptive flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--adaptive-kpis` | `system_cpu_pct` | KPI names to monitor for stopping |
| `--adaptive-no-futility` | off | Disable futility stopping |
| `--adaptive-look-every` | auto | [GST only] Check every N reps per side |

Example:

```bash
python -m ab_harness iteration \
  --experiment my_test \
  --adaptive --repetitions 300 \
  --adaptive-kpis system_cpu_pct
```

### Shared Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--experiment, -e` | (required) | Experiment name |
| `--host` | `blueos.local` | MCM host address |
| `--port` | `6020` | MCM HTTP port |
| `--duration` | `60` | Seconds per repetition |
| `--repetitions` | `3` | Number of collection repetitions |
| `--warmup` | `5` | Seconds after stats reset before collecting |
| `--deploy` | `remote` | Deploy mode: `local` or `remote` |
| `--env KEY=VALUE` | (none) | Environment overrides (repeatable) |
| `--webrtc PRODUCER[:COUNT]` | (none) | WebRTC consumer: producer name + optional client count (repeatable) |
| `--rtsp PATH[:COUNT]` | (none) | RTSP consumer: stream path + optional client count (repeatable) |
| `--thumbnail SOURCE[:COUNT]` | (none) | Thumbnail probe: source name + optional parallel request count (repeatable) |
| `--thumbnail-interval SECS` | `1` | Seconds between thumbnail probes |
| `--extra-args` | (none) | Extra MCM CLI args |
| `--runs-dir` | `runs/` | Base directory for all run data |
| `-v, --verbose` | off | Debug logging |
| `--isolated` / `--no-isolated` | `--isolated` | [remote] Run MCM in a clean isolated container (see below) |
| `--no-restore` | off | [remote + isolated] Skip restarting BlueOS after tests |

### Isolated container mode

By default (`--isolated`, remote mode only), the harness:

1. Inspects the running `blueos-core` container to capture its Docker image and
   bind mounts.
2. **Stops all containers** on the Pi (including `blueos-core` and extensions).
3. Starts a new container (`blueos-core-ab-test`) from the same image with the
   same mounts but a **minimal entrypoint** -- only a tmux server, no BlueOS
   services.  MCM is the sole workload, dramatically reducing CPU noise.
4. **Clears memory** -- flushes page cache, dentries/inodes, and swap so every
   experiment starts from the same memory state.
5. **Pins CPU governor to `performance`** -- locks all cores at max frequency,
   removing frequency-scaling jitter.  The original governor is restored
   automatically during teardown.
6. **Thermal warmup** -- stresses all CPU cores until the SoC temperature
   stabilises (< 0.5 C change over 10 s), then waits 5 s for the governor to
   settle.  This eliminates the cold-start thermal ramp that otherwise inflates
   variance in the first several reps.
7. After tests complete (or on crash), **restores the CPU governor**, **tears
   down** the isolated container and restarts `blueos-bootstrap` to bring the
   full BlueOS stack back up.

The thermal warmup parameters (`start_temp_c`, `end_temp_c`, `duration_s`) are
saved in the experiment metadata for reproducibility.

Use `--no-isolated` to skip isolation and inject MCM into the running
`blueos-core` container (legacy behaviour -- noisier measurements).

Use `--no-restore` to skip the automatic `blueos-bootstrap` restart after tests.
This is useful when chaining multiple test runs back-to-back.

### `overnight` -- Full overnight A/B test

Combines `baseline` + adaptive `iteration` + `replay` into a single invocation.
Designed for unattended / `nohup` runs.

```bash
# Full overnight run with adaptive stopping (GST by default)
python -m ab_harness overnight \
  --experiment my_test \
  --host 192.168.2.2 --port 6020 \
  --deploy remote \
  --duration 30 --repetitions 300 --warmup 5 \
  --webrtc "RadCam:2" --webrtc "UDP Stream" \
  --rtsp "video_stream_0:3" \
  --thumbnail "RadCam:2" --thumbnail-interval 5 \
  --label "my_change" \
  --adaptive-kpis system_cpu_pct

# Retry after a crash (skip cross-compilation)
python -m ab_harness overnight ... --no-build
```

The subcommand:
1. Runs a **1-rep bootstrap baseline** (skipped if one already exists and `--no-build`).
2. Runs the **adaptive iteration** (interleaved, randomized, early stopping).
3. **Replays** collected data through all sequential testing methods.

All flags from `iteration` are supported (`--adaptive-method`, `--adaptive-kpis`,
`--no-build`, source flags like `--github-pr`, etc.).

### `batch` -- Run multiple experiments overnight

Run a series of independent A/B experiments from a single YAML config file.
Each experiment gets its own baseline + adaptive iteration + replay, run
sequentially. BlueOS is only restored after the last experiment completes.

```bash
python -m ab_harness batch --config runs/my_batch.yaml
```

**YAML config format:**

```yaml
# Shared defaults (all optional, shown with built-in defaults)
defaults:
  host: blueos.local
  port: 6020
  duration: 60
  repetitions: 30
  warmup: 5
  deploy: remote
  isolated: true
  adaptive_method: gst
  adaptive_kpis: [system_cpu_pct]
  env: {}
  webrtc: []              # e.g. ["RadCam:2", "UDP Stream"]
  rtsp: []                # e.g. ["video_stream_0:3"]
  thumbnails: []           # e.g. ["RadCam:2"]
  thumbnail_interval: 1   # seconds between thumbnail probes
  extra_args: []

# Experiments run sequentially, top to bottom
experiments:
  - experiment: pr568_adaptive
    label: tcp_over_udp
    github_pr: "568"

  - experiment: pr567_shm
    label: shm_over_proxy
    github_pr: "567"
    duration: 120        # override default

  - experiment: commit_test
    label: refactor_sinks
    git_hash: "abc1234"
```

Per-experiment fields override the defaults. Only one of `github_pr`,
`git_hash`, or `diff` may be set per experiment.

**Resume on crash:** The batch command tracks progress in a state file
(`_batch_state_<name>.json` in the runs directory). If the process crashes
or is interrupted, re-running the same command skips already-completed
experiments and retries any that failed.

```bash
# Resume after crash (same command)
python -m ab_harness batch --config runs/my_batch.yaml

# Start fresh (discard state, re-run everything)
python -m ab_harness batch --config runs/my_batch.yaml --fresh
```

**Batch summary:** After all experiments complete, a combined HTML summary
(`batch_summary_<name>.html`) is generated in the runs directory with a
table showing each experiment's verdict, key KPI deltas, elapsed time,
and links to individual reports.

| Flag | Default | Description |
|------|---------|-------------|
| `--config, -c` | (required) | Path to the batch YAML config file |
| `--runs-dir` | `runs/` | Base directory for run data |
| `--fresh` | off | Discard state file, start all experiments from scratch |
| `--no-restore` | off | Skip restoring BlueOS after the last experiment |

### `restore` -- Manually bring BlueOS back up

If a test crashes or you used `--no-restore`, you can manually restore BlueOS:

```bash
python -m ab_harness restore --host 192.168.2.2
```

This cleans up any leftover `blueos-core-ab-test` container and starts
`blueos-bootstrap`, which brings `blueos-core` and all extensions back up.

## Data Layout

```
runs/<experiment>/
  baseline/
    metadata.json          # Git hash, diff, env, config
    rep_001/               # Repetition 1
      manifest.json        # Collection manifest
      static_config.json   # Stats level + window size
      topology.json        # Pipeline topology fingerprint
      series/              # 1Hz JSON snapshots
        pipeline_analysis/
        health/
        root_cause/
        per_pipeline/
        per_element/
        thumbnails/        # (if --thumbnail configured)
    rep_002/
    rep_003/
    aggregate.json         # KPI summary + topology across all reps
    container_logs/        # MCM stdout, debug, frame logs
  iterations/
    001_<label>/
      metadata.json
      rep_001/ ...
      aggregate.json
      comparison.json      # KPI deltas vs baseline
      report.html          # Self-contained HTML report
    002_<label>/
      ...
```

## KPI Reference

| KPI | Unit | Higher is Better | Regression Threshold |
|-----|------|-------------------|---------------------|
| `throughput_fps` | fps | yes | -5% |
| `freshness_delay_ms` | ms | no | +20% |
| `pipeline_cpu_pct` | % | no | +10% |
| `system_cpu_pct` | % | no | +10% |
| `system_load_1m` | - | no | +20% |
| `system_mem_used_pct` | % | no | +10% |
| `system_temperature_c` | C | no | +5% |
| `total_stutter_events` | count | no | +50% |
| `total_freeze_events` | count | no | +50% |
| `max_freeze_ms` | ms | no | +30% |
| `max_stutter_ratio` | ratio | no | +30% |
| `interval_p95_ms` | ms | no | +15% |
| `interval_p99_ms` | ms | no | +20% |
| `interval_std_ms` | ms | no | +20% |
| `top_thread_cpu_pct` | % | no | +10% |
| `restarts` | count | no | +50% |
| `expected_interval_ms` | ms | no | +10% |
| `edge_max_freshness_delay_ms` | ms | no | +20% |
| `edge_min_causal_confidence` | 0-1 | yes | -20% |
| `thumbnail_success_pct` | % | yes | -5% |

A KPI is flagged as a **regression** when:
- The delta exceeds the threshold AND
- The Welch's t-test p-value < 0.05 (statistically significant across reps)

## Topology Change Detection

The harness extracts pipeline topology information from the MCM stats API at the
end of each repetition. This includes:

- **Pipeline names** and expected frame intervals
- **Element-to-thread mapping** -- which processing elements run on which thread
- **Edge graph** -- connections between elements with freshness delay and causal
  confidence metrics
- **Restart count** -- how many times each pipeline restarted

When comparing an iteration against a baseline, the harness automatically detects
topology changes:

| Change Type | Example | Invalidated KPIs |
|-------------|---------|-----------------|
| Pipeline added/removed | A pipeline appears or disappears | All structure- and thread-sensitive KPIs |
| Element thread change | Element X moves from thread 1 to thread 2 | `pipeline_cpu_pct`, `top_thread_cpu_pct` |
| Edge added/removed | A connection between elements appears or disappears | All structure-sensitive KPIs (throughput, delays, stutter, etc.) |

When a topology change is detected:
1. Affected KPIs are marked with verdict `topology_changed` instead of
   regression/improvement.
2. The HTML report shows a prominent warning banner listing the changes.
3. Invalidated KPIs are visually dimmed with a **TOPO** badge in the comparison
   table.

This prevents false regressions caused by architectural changes that make certain
KPI comparisons meaningless. For example, if a code change moves an element to a
different thread, the thread CPU contributions are no longer comparable between
the two runs.

## Multi-Client Configuration

The `--webrtc`, `--rtsp`, and `--thumbnail` flags support the `NAME[:COUNT]`
syntax where COUNT specifies how many parallel client instances to spawn:

```bash
# 2 WebRTC clients consuming RadCam, 1 consuming UDP Stream
python -m ab_harness baseline \
  --experiment multi_client \
  --webrtc "RadCam:2" --webrtc "UDP Stream"

# 3 parallel RTSP clients on the same path
python -m ab_harness baseline \
  --experiment rtsp_load \
  --rtsp "video_stream_0:3"

# Thumbnail probing: 2 parallel requests per source, every 5 seconds
python -m ab_harness baseline \
  --experiment thumb_test \
  --thumbnail "RadCam:2" --thumbnail "USB Camera" \
  --thumbnail-interval 5
```

When COUNT is omitted, it defaults to 1. These flags are available on all
subcommands that accept shared flags (`baseline`, `iteration`, `overnight`,
`recommend`).

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `sshpass: command not found` | sshpass not installed | Install it (see Setup) |
| `cross: command not found` | cross not installed | `cargo install cross --git https://github.com/cross-rs/cross` |
| `ConnectionRefusedError` on port 6020 | MCM is not running on the device | Deploy MCM first or check the host address |
| `Failed to open Zenoh session` (MCM panic) | MCM tries to connect to Zenoh but there is no router | Use `--isolated` mode (default) which disables Zenoh |
| `Text file busy` when replacing binary | MCM process is still holding the file | The harness kills MCM before replacing; if it persists, `docker exec ... pkill -9 mavlink-camera` manually |
| `playwright._impl._errors.Error: Executable doesn't exist` | Chromium not installed for Playwright | Run `playwright install chromium` |
| Build fails with `cross build` | Docker daemon not running, or target image not pulled | Ensure Docker is running; try `cross build --release --target=$CROSS_TARGET` manually |
| SSH timeout | Wrong host, device not on network, or wrong port | Verify with `ping <host>` and `ssh pi@<host>` |
| `check` subcommand reports failures | Missing dependencies or connectivity | Follow the specific instructions printed by `check` |
