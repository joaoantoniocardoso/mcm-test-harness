"""KPI definitions, extraction from snapshots, and aggregation across reps."""

from __future__ import annotations

__all__ = [
    "KPI_BY_NAME",
    "KPI_REGISTRY",
    "AggregateKpis",
    "KpiThreshold",
    "RepSummary",
    "SampleKpis",
    "aggregate_reps",
    "extract_sample_kpis",
    "extract_timeseries",
    "load_aggregate",
    "summarize_rep",
]

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# KPI definitions & thresholds
# ---------------------------------------------------------------------------


@dataclass
class KpiThreshold:
    """Define when a delta is considered a regression."""

    name: str
    unit: str
    higher_is_better: bool
    regression_pct: float  # e.g., 5.0 means >5% worse is regression
    description: str = ""


# Registry of all tracked KPIs with their regression thresholds
KPI_REGISTRY: list[KpiThreshold] = [
    # Pipeline-level
    KpiThreshold("throughput_fps", "fps", True, 5.0, "Frames per second throughput"),
    KpiThreshold("freshness_delay_ms", "ms", False, 20.0, "End-to-end pipeline freshness delay"),
    KpiThreshold("pipeline_cpu_pct", "%", False, 10.0, "Pipeline CPU attribution"),
    # System-level
    KpiThreshold("system_cpu_pct", "%", False, 10.0, "System-wide CPU usage"),
    KpiThreshold("system_load_1m", "", False, 20.0, "1-minute load average"),
    KpiThreshold("system_mem_used_pct", "%", False, 10.0, "System memory usage"),
    KpiThreshold("system_temperature_c", "C", False, 5.0, "SoC temperature"),
    # Element diagnostics (aggregated)
    KpiThreshold("total_stutter_events", "", False, 50.0, "Total stutter events across elements"),
    KpiThreshold("total_freeze_events", "", False, 50.0, "Total freeze events across elements"),
    KpiThreshold("max_freeze_ms", "ms", False, 30.0, "Worst max freeze duration"),
    KpiThreshold("max_stutter_ratio", "", False, 30.0, "Worst stutter ratio across elements"),
    # Distribution (interval)
    KpiThreshold("interval_p95_ms", "ms", False, 15.0, "95th percentile frame interval"),
    KpiThreshold("interval_p99_ms", "ms", False, 20.0, "99th percentile frame interval"),
    KpiThreshold("interval_std_ms", "ms", False, 20.0, "Frame interval standard deviation"),
    # Thread bottleneck
    KpiThreshold("top_thread_cpu_pct", "%", False, 10.0, "Hottest thread CPU usage"),
    # Topology / edges
    KpiThreshold("restarts", "", False, 50.0, "Pipeline restart count"),
    KpiThreshold("expected_interval_ms", "ms", False, 10.0, "Expected frame interval"),
    KpiThreshold(
        "edge_max_freshness_delay_ms",
        "ms",
        False,
        20.0,
        "Worst edge freshness delay across all edges",
    ),
    KpiThreshold(
        "edge_min_causal_confidence",
        "",
        True,
        20.0,
        "Lowest causal confidence across edges (0-1)",
    ),
    # Thumbnail verification
    KpiThreshold("thumbnail_success_pct", "%", True, 5.0, "Thumbnail request success rate"),
]

KPI_BY_NAME: dict[str, KpiThreshold] = {k.name: k for k in KPI_REGISTRY}

# Map categorical causal-confidence strings from the MCM API to numeric 0-1 values.
_CONFIDENCE_MAP: dict[str, float] = {"high": 1.0, "medium": 0.5, "low": 0.0}


# ---------------------------------------------------------------------------
# Per-sample KPI extraction
# ---------------------------------------------------------------------------


@dataclass
class SampleKpis:
    """KPIs extracted from a single 1-second snapshot."""

    t_rel_s: float = 0.0

    # Pipeline
    throughput_fps: float | None = None
    freshness_delay_ms: float | None = None
    pipeline_cpu_pct: float | None = None
    causal_latency_health: str | None = None

    # System
    system_cpu_pct: float | None = None
    system_load_1m: float | None = None
    system_mem_used_pct: float | None = None
    system_temperature_c: float | None = None

    # Element diagnostics (summed/max across all elements)
    total_stutter_events: int | None = None
    total_freeze_events: int | None = None
    max_freeze_ms: float | None = None
    max_stutter_ratio: float | None = None

    # Distribution (from first pipeline's source pad, if available)
    interval_p95_ms: float | None = None
    interval_p99_ms: float | None = None
    interval_std_ms: float | None = None

    # Thread bottleneck
    top_thread_cpu_pct: float | None = None

    # Topology / edges
    restarts: int | None = None
    expected_interval_ms: float | None = None
    edge_max_freshness_delay_ms: float | None = None
    edge_min_causal_confidence: float | None = None

    # Thumbnail verification
    thumbnail_success_pct: float | None = None


def extract_sample_kpis(
    analysis_snapshot: Any,
    health_snapshot: Any,
    element_snapshot: Any,
    t_rel_s: float,
    thumbnail_snapshot: Any = None,
) -> SampleKpis:
    """Extract KPIs from a single snapshot set."""
    kpis = SampleKpis(t_rel_s=t_rel_s)

    # --- Pipeline-level (from first pipeline in the list) ---
    if isinstance(analysis_snapshot, list) and analysis_snapshot:
        pipe = analysis_snapshot[0]
        summary = pipe.get("summary") or {}
        kpis.throughput_fps = summary.get("throughput_fps")
        kpis.freshness_delay_ms = summary.get("total_pipeline_freshness_delay_ms")
        kpis.pipeline_cpu_pct = pipe.get("pipeline_cpu_pct")
        kpis.causal_latency_health = summary.get("causal_latency_health")

        # System
        system = pipe.get("system") or {}
        kpis.system_cpu_pct = system.get("current_cpu_pct")
        kpis.system_load_1m = system.get("current_load_1m")
        kpis.system_mem_used_pct = system.get("current_mem_used_pct")
        kpis.system_temperature_c = system.get("current_temperature_c")

        # Distribution from elements
        _extract_distribution(pipe, kpis)

        # Thread bottleneck
        thread_groups = pipe.get("thread_groups") or []
        if thread_groups:
            kpis.top_thread_cpu_pct = max((tg.get("cpu_pct", 0) or 0) for tg in thread_groups)

        # Topology / edges
        restarts_raw = pipe.get("restarts")
        if isinstance(restarts_raw, dict):
            kpis.restarts = restarts_raw.get("restart_count", 0)
        elif isinstance(restarts_raw, int | float):
            kpis.restarts = int(restarts_raw)
        kpis.expected_interval_ms = pipe.get("expected_interval_ms")

        edges = pipe.get("edges") or []
        if isinstance(edges, list) and edges:
            delays: list[Any] = [
                e.get("freshness_delay_ms")
                for e in edges
                if isinstance(e, dict) and e.get("freshness_delay_ms") is not None
            ]
            confidences: list[float] = []
            for e in edges:
                if not isinstance(e, dict):
                    continue
                raw_cc = e.get("causal_confidence")
                if isinstance(raw_cc, int | float):
                    confidences.append(float(raw_cc))
                elif isinstance(raw_cc, str):
                    mapped = _CONFIDENCE_MAP.get(raw_cc.lower())
                    if mapped is not None:
                        confidences.append(mapped)
            if delays:
                kpis.edge_max_freshness_delay_ms = max(delays)
            if confidences:
                kpis.edge_min_causal_confidence = min(confidences)

    # --- Element diagnostics ---
    if isinstance(element_snapshot, dict):
        total_stutter = 0
        total_freeze = 0
        worst_freeze = 0.0
        worst_stutter_ratio = 0.0

        for _pname, elements in element_snapshot.items():
            if not isinstance(elements, dict):
                continue
            for _ename, info in elements.items():
                diag = ((info or {}).get("diagnostics") or {}).get("payload") or {}
                if diag:
                    total_stutter += diag.get("stutter_events", 0) or 0
                    total_freeze += diag.get("freeze_events", 0) or 0
                    worst_freeze = max(worst_freeze, diag.get("max_freeze_ms", 0) or 0)
                    worst_stutter_ratio = max(
                        worst_stutter_ratio, diag.get("stutter_ratio", 0) or 0
                    )

        kpis.total_stutter_events = total_stutter
        kpis.total_freeze_events = total_freeze
        kpis.max_freeze_ms = worst_freeze
        kpis.max_stutter_ratio = worst_stutter_ratio

    # --- Thumbnail verification ---
    if isinstance(thumbnail_snapshot, list) and thumbnail_snapshot:
        total = len(thumbnail_snapshot)
        successful = sum(1 for t in thumbnail_snapshot if t.get("ok"))
        kpis.thumbnail_success_pct = (successful / total) * 100.0

    return kpis


def _extract_distribution(pipe: dict[str, Any], kpis: SampleKpis) -> None:
    """Pull interval distribution from the first element's first src pad."""
    elements = pipe.get("elements") or {}
    for _ename, elem in elements.items():
        if not isinstance(elem, dict):
            continue
        src_pads = elem.get("src_pads") or {}
        for _pname, pad in src_pads.items():
            if not isinstance(pad, dict):
                continue
            # Full mode: distribution.interval
            dist = (pad.get("distribution") or {}).get("interval")
            if isinstance(dist, dict):
                kpis.interval_p95_ms = dist.get("p95")
                kpis.interval_p99_ms = dist.get("p99")
                kpis.interval_std_ms = dist.get("std")
                return
            # Lite mode: accumulators
            acc = pad.get("accumulators")
            if isinstance(acc, dict):
                kpis.interval_std_ms = acc.get("std_interval_ms")
                return


# ---------------------------------------------------------------------------
# Rep-level summary
# ---------------------------------------------------------------------------


@dataclass
class RepSummary:
    """Summary statistics for one repetition."""

    rep_index: int
    sample_count: int
    kpis: dict[str, dict[str, float | None]]  # kpi_name -> {mean, p50, p95, p99, max, min}


def _load_rep_samples(rep_dir: Path) -> list[SampleKpis]:
    """Load all snapshots from a rep directory and return extracted KPIs.

    Reads the manifest, iterates over each sample, loads the 4 JSON snapshot
    files, and calls :func:`extract_sample_kpis` for each.
    """
    series = rep_dir / "series"
    manifest_path = rep_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    samples_meta = manifest.get("samples", [])

    result: list[SampleKpis] = []
    for sample in samples_meta:
        idx = sample["index"]
        unix = sample["unix"]
        stamp = f"{idx:04d}_{unix}"
        t_rel = sample.get("t_rel_s", 0)

        analysis = _load_json(series / "pipeline_analysis" / f"{stamp}.json")
        health = _load_json(series / "health" / f"{stamp}.json")
        element = _load_json(series / "per_element" / f"{stamp}.json")
        thumbnail = _load_json(series / "thumbnails" / f"{stamp}.json")

        result.append(extract_sample_kpis(analysis, health, element, t_rel, thumbnail))

    return result


def summarize_rep(rep_dir: Path, rep_index: int) -> RepSummary:
    """Load all snapshots from a rep and compute per-KPI summary stats."""
    all_kpis = _load_rep_samples(rep_dir)

    # Compute summary for each numeric KPI
    kpi_summaries: dict[str, dict[str, float | None]] = {}
    numeric_fields = [f for f in KPI_BY_NAME if hasattr(SampleKpis, f)]

    for fname in numeric_fields:
        values = [getattr(k, fname) for k in all_kpis if getattr(k, fname) is not None]
        if values:
            arr = np.array(values, dtype=float)
            kpi_summaries[fname] = {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
                "p50": float(np.percentile(arr, 50)),
                "p95": float(np.percentile(arr, 95)),
                "p99": float(np.percentile(arr, 99)),
                "max": float(np.max(arr)),
                "min": float(np.min(arr)),
                "count": len(values),
            }
        else:
            kpi_summaries[fname] = {
                "mean": None,
                "std": None,
                "p50": None,
                "p95": None,
                "p99": None,
                "max": None,
                "min": None,
                "count": 0,
            }

    return RepSummary(
        rep_index=rep_index,
        sample_count=len(all_kpis),
        kpis=kpi_summaries,
    )


# ---------------------------------------------------------------------------
# Aggregate across reps
# ---------------------------------------------------------------------------


@dataclass
class AggregateKpis:
    """Aggregated KPI summary across all reps."""

    rep_count: int
    per_rep: list[RepSummary]
    aggregate: dict[str, dict[str, Any]]
    # aggregate[kpi_name] = {mean, std, min, max, per_rep_means: [...]}


def aggregate_reps(run_dir: Path) -> AggregateKpis:
    """Load all rep_NNN dirs, summarize each, then aggregate."""
    rep_dirs = sorted(run_dir.glob("rep_*"))
    if not rep_dirs:
        raise FileNotFoundError(f"No rep_* directories found in {run_dir}")

    per_rep: list[RepSummary] = []
    for i, rd in enumerate(rep_dirs):
        log.info("Summarizing rep %d: %s", i + 1, rd.name)
        per_rep.append(summarize_rep(rd, i + 1))

    # Aggregate: for each KPI, collect the per-rep means
    numeric_fields = [f for f in KPI_BY_NAME if hasattr(SampleKpis, f)]
    aggregate: dict[str, dict[str, Any]] = {}

    for fname in numeric_fields:
        rep_means = []
        for rs in per_rep:
            m = rs.kpis.get(fname, {}).get("mean")
            if m is not None:
                rep_means.append(m)

        if rep_means:
            arr = np.array(rep_means, dtype=float)
            aggregate[fname] = {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
                "per_rep_means": rep_means,
                "rep_count": len(rep_means),
            }
        else:
            aggregate[fname] = {
                "mean": None,
                "std": None,
                "min": None,
                "max": None,
                "per_rep_means": [],
                "rep_count": 0,
            }

    result = AggregateKpis(
        rep_count=len(per_rep),
        per_rep=per_rep,
        aggregate=aggregate,
    )

    # Load topology from the last rep (if available)
    topology: Any = None
    if rep_dirs:
        topo_path = rep_dirs[-1] / "topology.json"
        if topo_path.exists():
            topology = _load_json(topo_path)

    # Save aggregate.json
    save_path = run_dir / "aggregate.json"
    save_data: dict[str, Any] = {
        "rep_count": result.rep_count,
        "aggregate": result.aggregate,
        "per_rep": [
            {"rep_index": rs.rep_index, "sample_count": rs.sample_count, "kpis": rs.kpis}
            for rs in result.per_rep
        ],
    }
    if topology is not None:
        save_data["topology"] = topology
    save_path.write_text(json.dumps(save_data, indent=2, default=str))
    log.info("Aggregate KPIs saved to %s", save_path)
    return result


def load_aggregate(run_dir: Path) -> dict[str, Any]:
    """Load a previously saved aggregate.json."""
    path = run_dir / "aggregate.json"
    if not path.exists():
        raise FileNotFoundError(f"No aggregate.json in {run_dir}")
    data: dict[str, Any] = json.loads(path.read_text())
    return data


# ---------------------------------------------------------------------------
# Timeseries extraction for charts
# ---------------------------------------------------------------------------


def extract_timeseries(run_dir: Path) -> dict[int, list[SampleKpis]]:
    """Extract per-rep timeseries of SampleKpis for charting. Returns {rep_index: [...]}."""
    rep_dirs = sorted(run_dir.glob("rep_*"))
    result: dict[int, list[SampleKpis]] = {}

    for i, rd in enumerate(rep_dirs):
        if not (rd / "manifest.json").exists():
            continue
        result[i + 1] = _load_rep_samples(rd)

    return result


def _load_json(path: Path) -> Any:
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            pass
    return None
