"""Poll MCM stats API at 1 Hz and store JSON snapshots for one repetition."""

from __future__ import annotations

import concurrent.futures
import contextlib
import json
import logging
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from ab_harness.config import ClientConfig, ConsumerSpec, SshConfig, ssh_run

log = logging.getLogger(__name__)


def measure_idle_cpu(
    host: str,
    *,
    ssh_user: str | None = None,
    ssh_pwd: str | None = None,
    ssh_port: int | None = None,
    container: str | None = None,
    duration: int = 10,
) -> dict[str, Any]:
    """Measure idle system CPU on the Pi (without MCM running) over *duration* seconds.

    Returns a dict with idle_cpu_pct, baseline_load_1m, baseline_temp_c.
    """
    ssh = SshConfig.from_overrides(
        user=ssh_user,
        pwd=ssh_pwd,
        port=ssh_port,
        container=container,
    )
    log.info("Measuring idle CPU on %s for %ds (MCM stopped) ...", host, duration)

    def _ssh(cmd: str, timeout: int = 60) -> str:
        return ssh_run(host, ssh, cmd, timeout=timeout)

    container = ssh.container

    # Ensure MCM is stopped
    _ssh(
        f"docker exec {container} bash -c '"
        f"tmux kill-session -t video 2>/dev/null; "
        f"pkill -9 -f run-service.video 2>/dev/null; "
        f"pkill -9 mavlink-camera 2>/dev/null'"
    )
    time.sleep(3)

    # Sample /proc/stat to compute CPU usage
    # Read CPU times at start, wait, read again
    cpu_cmd = "head -1 /proc/stat"
    start_line = _ssh(f"docker exec {container} {cpu_cmd}")
    time.sleep(duration)
    end_line = _ssh(f"docker exec {container} {cpu_cmd}")

    idle_cpu_pct: float | None = None
    try:
        # Parse: cpu  user nice system idle iowait irq softirq steal guest guest_nice
        s = [int(x) for x in start_line.split()[1:]]
        e = [int(x) for x in end_line.split()[1:]]
        d = [e[i] - s[i] for i in range(len(s))]
        total = sum(d)
        if total > 0:
            idle = d[3] + d[4]  # idle + iowait
            idle_cpu_pct = round((1.0 - idle / total) * 100, 2)
    except Exception as exc:
        log.warning("Failed to parse /proc/stat: %s", exc)

    # Grab load average
    load_str = _ssh(f"docker exec {container} cat /proc/loadavg")
    baseline_load_1m: float | None = None
    with contextlib.suppress(Exception):
        baseline_load_1m = float(load_str.split()[0])

    # Grab temperature
    temp_str = _ssh(
        f"docker exec {container} cat /sys/class/thermal/thermal_zone0/temp 2>/dev/null"
    )
    baseline_temp_c: float | None = None
    with contextlib.suppress(Exception):
        baseline_temp_c = round(int(temp_str) / 1000.0, 1)

    result = {
        "idle_cpu_pct": idle_cpu_pct,
        "baseline_load_1m": baseline_load_1m,
        "baseline_temp_c": baseline_temp_c,
        "duration_seconds": duration,
    }
    log.info("Idle CPU measurement: %s", result)
    return result


def _fetch_json(url: str, timeout: float = 5.0) -> Any:
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8", errors="replace"))


def _post_json(url: str, body: dict[str, Any] | None = None, timeout: float = 5.0) -> Any:
    data = json.dumps(body).encode() if body else None
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8", errors="replace"))


def _safe_fetch(url: str) -> dict[str, Any]:
    try:
        return {"ok": True, "error": None, "payload": _fetch_json(url)}
    except Exception as exc:
        return {"ok": False, "error": str(exc), "payload": None}


def _safe_post(url: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
    try:
        return {"ok": True, "error": None, "payload": _post_json(url, body)}
    except Exception as exc:
        return {"ok": False, "error": str(exc), "payload": None}


def reset_and_configure(base_url: str) -> None:
    """Reset stats and ensure full level."""
    log.info("Resetting stats and setting level=full ...")
    _safe_post(f"{base_url}/stats/pipeline-analysis/reset")
    _safe_post(f"{base_url}/stats/pipeline-analysis/level", {"level": "full"})


def capture_static_config(base_url: str, out_dir: Path) -> dict[str, Any]:
    """Capture one-time config endpoints."""
    static: dict[str, Any] = {}
    for path in [
        "/stats/pipeline-analysis/level",
        "/stats/pipeline-analysis/window-size",
    ]:
        static[path] = _safe_fetch(f"{base_url}{path}")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "static_config.json").write_text(json.dumps(static, indent=2))
    return static


def _probe_single_thumbnail(
    base_url: str,
    source: str,
    quality: int = 75,
    target_height: int = 150,
    timeout: float = 5.0,
) -> dict[str, Any]:
    """Probe the thumbnail endpoint once for a single source."""
    params = urllib.parse.urlencode(
        {
            "source": source,
            "quality": quality,
            "target_height": target_height,
        }
    )
    url = f"{base_url}/thumbnail?{params}"
    entry: dict[str, Any] = {
        "source": source,
        "ok": False,
        "status_code": None,
        "content_length": 0,
        "latency_ms": 0.0,
        "error": None,
    }
    t0 = time.time()
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read()
            entry["status_code"] = resp.status
            entry["content_length"] = len(body)
            content_type = resp.headers.get("Content-Type", "")
            if resp.status == 200 and "image/jpeg" in content_type and len(body) > 0:
                entry["ok"] = True
            else:
                entry["error"] = (
                    f"Unexpected response: status={resp.status}, "
                    f"content_type={content_type}, body_len={len(body)}"
                )
    except urllib.error.HTTPError as exc:
        entry["status_code"] = exc.code
        entry["error"] = f"HTTP {exc.code}: {exc.reason}"
    except Exception as exc:
        entry["error"] = str(exc)
    finally:
        entry["latency_ms"] = round((time.time() - t0) * 1000, 1)
    return entry


def probe_thumbnails(
    base_url: str,
    specs: list[ConsumerSpec],
    quality: int = 75,
    target_height: int = 150,
    timeout: float = 5.0,
) -> list[dict[str, Any]]:
    """Probe the thumbnail endpoint for each source with parallel requests.

    For each :class:`ConsumerSpec`, fires ``spec.count`` parallel HTTP requests
    and aggregates them into a single per-source result dict containing:
      source, ok, status_code, content_length, latency_ms, error,
      parallel_count, parallel_success
    """
    results: list[dict[str, Any]] = []

    # Build the full list of (source, index) jobs
    jobs: list[tuple[str, int]] = []
    for spec in specs:
        for i in range(spec.count):
            jobs.append((spec.name, i))

    if not jobs:
        return results

    # Fire all probes in parallel
    raw_results: dict[str, list[dict[str, Any]]] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(len(jobs), 1)) as pool:
        future_map = {
            pool.submit(
                _probe_single_thumbnail, base_url, source, quality, target_height, timeout
            ): source
            for source, _idx in jobs
        }
        for future in concurrent.futures.as_completed(future_map):
            source = future_map[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {
                    "source": source,
                    "ok": False,
                    "status_code": None,
                    "content_length": 0,
                    "latency_ms": 0.0,
                    "error": str(exc),
                }
            raw_results.setdefault(source, []).append(result)

    # Aggregate per source
    for spec in specs:
        probes = raw_results.get(spec.name, [])
        success_count = sum(1 for p in probes if p.get("ok"))
        avg_latency = sum(p.get("latency_ms", 0) for p in probes) / len(probes) if probes else 0.0
        # Use the first probe's status info as representative
        first = probes[0] if probes else {}
        results.append(
            {
                "source": spec.name,
                "ok": success_count == len(probes) and len(probes) > 0,
                "status_code": first.get("status_code"),
                "content_length": first.get("content_length", 0),
                "latency_ms": round(avg_latency, 1),
                "error": first.get("error"),
                "parallel_count": spec.count,
                "parallel_success": success_count,
            }
        )
    return results


def _report_thumbnail_health(
    results: list[dict[str, Any]],
    sample_index: int,
    *,
    _prev_failed: set[str] = set(),  # noqa: B006 - mutable default is intentional (state across calls)
) -> None:
    """Log thumbnail probe results. Warns on first failure and on status changes."""
    failed_now = {r.get("source", "?") for r in results if not r.get("ok")}
    ok_now = {r.get("source", "?") for r in results if r.get("ok")}

    if sample_index == 0:
        # First sample: always report full status
        if ok_now:
            log.info("Thumbnail probes OK: %s", ", ".join(sorted(ok_now)))
        if failed_now:
            errors = {
                r.get("source", "?"): r.get("error", "unknown") for r in results if not r.get("ok")
            }
            log.warning(
                "Thumbnail probes FAILED for %d/%d sources: %s",
                len(failed_now),
                len(results),
                "; ".join(f"{s} ({e})" for s, e in sorted(errors.items())),
            )
    else:
        # Subsequent samples: only log changes
        newly_failed = failed_now - _prev_failed
        recovered = _prev_failed - failed_now
        if newly_failed:
            errors = {
                r.get("source", "?"): r.get("error", "unknown")
                for r in results
                if r.get("source") in newly_failed
            }
            log.warning(
                "Thumbnail probes started failing: %s",
                "; ".join(f"{s} ({e})" for s, e in sorted(errors.items())),
            )
        if recovered:
            log.info("Thumbnail probes recovered: %s", ", ".join(sorted(recovered)))

    _prev_failed.clear()
    _prev_failed.update(failed_now)


def collect_one_rep(
    base_url: str,
    rep_dir: Path,
    duration_seconds: int,
    warmup_seconds: int = 5,
    client_config: ClientConfig | None = None,
) -> dict[str, Any]:
    """
    Run one collection repetition:
      1. Reset + set level full
      2. Warmup
      3. Poll at 1 Hz for duration_seconds
      4. Store snapshots under rep_dir/series/
    Returns a manifest dict.
    """
    if client_config is None:
        client_config = ClientConfig()

    thumbnail_specs = client_config.thumbnails
    thumbnail_interval = max(1, client_config.thumbnail_interval)

    reset_and_configure(base_url)

    log.info("Warming up for %ds ...", warmup_seconds)
    time.sleep(warmup_seconds)

    # Prepare directories
    series = rep_dir / "series"
    subdirs = [
        "pipeline_analysis",
        "health",
        "root_cause",
        "per_pipeline",
        "per_element",
    ]
    if thumbnail_specs:
        subdirs.append("thumbnails")
    for subdir in subdirs:
        (series / subdir).mkdir(parents=True, exist_ok=True)

    capture_static_config(base_url, rep_dir)

    manifest: dict[str, Any] = {
        "started_unix": int(time.time()),
        "duration_seconds": duration_seconds,
        "warmup_seconds": warmup_seconds,
        "base_url": base_url,
        "samples": [],
    }

    start = time.time()
    index = 0
    last_analysis_payload: Any = None

    while True:
        now = time.time()
        elapsed = now - start
        if elapsed >= duration_seconds:
            break

        stamp = f"{index:04d}_{int(now)}"
        sample: dict[str, Any] = {
            "index": index,
            "unix": int(now),
            "t_rel_s": round(elapsed, 2),
            "ok": True,
            "errors": [],
            "pipelines": [],
            "pipeline_count": 0,
        }

        # --- Fleet-level endpoints ---
        analysis = _safe_fetch(f"{base_url}/stats/pipeline-analysis")
        health = _safe_fetch(f"{base_url}/stats/pipeline-analysis/health")
        root_cause = _safe_fetch(f"{base_url}/stats/pipeline-analysis/root-cause")

        for endpoint, result in [
            ("pipeline-analysis", analysis),
            ("health", health),
            ("root-cause", root_cause),
        ]:
            if not result["ok"]:
                sample["ok"] = False
                sample["errors"].append({"endpoint": endpoint, "error": result["error"]})

        _write_json(series / "pipeline_analysis" / f"{stamp}.json", analysis.get("payload"))
        _write_json(series / "health" / f"{stamp}.json", health.get("payload"))
        _write_json(series / "root_cause" / f"{stamp}.json", root_cause.get("payload"))

        # Keep last successful analysis payload for topology extraction
        if analysis.get("ok") and analysis.get("payload"):
            last_analysis_payload = analysis["payload"]

        # --- Discover pipelines ---
        pipelines: list[str] = []
        payload = analysis.get("payload")
        if isinstance(payload, list):
            pipelines = [
                str(p.get("pipeline_name"))
                for p in payload
                if isinstance(p, dict) and p.get("pipeline_name")
            ]
        sample["pipelines"] = pipelines
        sample["pipeline_count"] = len(pipelines)

        # --- Per-pipeline and per-element ---
        per_pipeline: dict[str, Any] = {}
        per_element: dict[str, Any] = {}

        for pname in pipelines:
            pname_q = urllib.parse.quote(pname, safe="")

            per_pipeline[pname] = {
                "root_cause": _safe_fetch(
                    f"{base_url}/stats/pipeline-analysis/{pname_q}/root-cause"
                ),
                "samples": _safe_fetch(
                    f"{base_url}/stats/pipeline-analysis/{pname_q}/samples?limit=5"
                ),
            }

            # Find elements in this pipeline's snapshot
            elements: list[str] = []
            if isinstance(payload, list):
                for entry in payload:
                    if isinstance(entry, dict) and entry.get("pipeline_name") == pname:
                        elems = entry.get("elements")
                        if isinstance(elems, dict):
                            elements = list(elems.keys())
                        break

            per_element[pname] = {}
            for ename in elements:
                ename_q = urllib.parse.quote(ename, safe="")
                per_element[pname][ename] = {
                    "diagnostics": _safe_fetch(
                        f"{base_url}/stats/pipeline-analysis/{pname_q}/elements/{ename_q}/diagnostics"
                    ),
                    "samples": _safe_fetch(
                        f"{base_url}/stats/pipeline-analysis/{pname_q}/elements/{ename_q}/samples?limit=5"
                    ),
                }

        _write_json(series / "per_pipeline" / f"{stamp}.json", per_pipeline)
        _write_json(series / "per_element" / f"{stamp}.json", per_element)

        # --- Thumbnail probes (at configurable interval) ---
        if thumbnail_specs and index % thumbnail_interval == 0:
            thumb_results = probe_thumbnails(base_url, thumbnail_specs)
            _write_json(series / "thumbnails" / f"{stamp}.json", thumb_results)
            sample["thumbnail_probes"] = thumb_results
            _report_thumbnail_health(thumb_results, index)

        manifest["samples"].append(sample)
        index += 1

        # Sleep until next 1s tick
        next_tick = start + index
        delay = next_tick - time.time()
        if delay > 0:
            time.sleep(delay)

        if index % 10 == 0:
            log.info("  collected %d/%ds samples ...", index, duration_seconds)

    manifest["ended_unix"] = int(time.time())
    manifest["sample_count"] = len(manifest["samples"])
    (rep_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    # --- Extract and save topology from last analysis snapshot ---
    if last_analysis_payload is not None:
        topology = extract_topology(last_analysis_payload)
        _write_json(rep_dir / "topology.json", topology)

    log.info("Collected %d samples into %s", manifest["sample_count"], rep_dir)
    return manifest


# ---------------------------------------------------------------------------
# Topology extraction
# ---------------------------------------------------------------------------


def extract_topology(pipeline_analysis_payload: Any) -> dict[str, Any]:
    """Extract a topology fingerprint from a ``/stats/pipeline-analysis`` snapshot.

    Returns a dict with per-pipeline topology information:
    ``pipelines`` list, each containing ``pipeline_name``, ``expected_interval_ms``,
    ``restarts``, ``elements`` (name -> element_type, thread_id), and ``edges``.
    """
    topology: dict[str, Any] = {"pipelines": []}
    if not isinstance(pipeline_analysis_payload, list):
        return topology

    for pipe in pipeline_analysis_payload:
        if not isinstance(pipe, dict):
            continue
        pipe_topo: dict[str, Any] = {
            "pipeline_name": pipe.get("pipeline_name", ""),
            "expected_interval_ms": pipe.get("expected_interval_ms"),
            "restarts": pipe.get("restarts", 0),
            "elements": {},
            "edges": [],
        }

        # Elements: name -> {element_type, thread_id}
        elements_map = pipe.get("elements") or {}
        if isinstance(elements_map, dict):
            for ename, edata in elements_map.items():
                if isinstance(edata, dict):
                    pipe_topo["elements"][ename] = {
                        "element_type": edata.get("element_type"),
                        "thread_id": edata.get("thread_id"),
                    }

        # Edges
        edges = pipe.get("edges") or []
        if isinstance(edges, list):
            for edge in edges:
                if isinstance(edge, dict):
                    pipe_topo["edges"].append(
                        {
                            "from_element": edge.get("from_element"),
                            "to_element": edge.get("to_element"),
                            "freshness_delay_ms": edge.get("freshness_delay_ms"),
                            "causal_match_rate": edge.get("causal_match_rate"),
                            "causal_confidence": edge.get("causal_confidence"),
                        }
                    )

        # Optional topology field from MCM
        if pipe.get("topology") is not None:
            pipe_topo["raw_topology"] = pipe["topology"]

        topology["pipelines"].append(pipe_topo)

    return topology


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, default=str))
