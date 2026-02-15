"""Shared test fixtures for ab_harness tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def tmp_run_dir(tmp_path: Path) -> Path:
    """Create a minimal run directory structure with 3 reps of synthetic data."""
    rng = np.random.default_rng(42)

    for rep_idx in range(3):
        rep_dir = tmp_path / f"rep_{rep_idx:03d}"
        series = rep_dir / "series"
        for sub in ["pipeline_analysis", "health", "per_element", "thumbnails"]:
            (series / sub).mkdir(parents=True)

        samples = []
        for s_idx in range(10):
            unix_ts = 1700000000 + s_idx
            stamp = f"{s_idx:04d}_{unix_ts}"

            # Pipeline analysis snapshot
            analysis = [
                {
                    "summary": {
                        "throughput_fps": float(rng.normal(30, 1)),
                        "total_pipeline_freshness_delay_ms": float(rng.normal(50, 5)),
                        "causal_latency_health": "ok",
                    },
                    "pipeline_cpu_pct": float(rng.normal(25, 3)),
                    "system": {
                        "current_cpu_pct": float(rng.normal(40, 5)),
                        "current_load_1m": float(rng.normal(1.5, 0.3)),
                        "current_mem_used_pct": float(rng.normal(55, 3)),
                        "current_temperature_c": float(rng.normal(60, 2)),
                    },
                    "elements": {},
                    "thread_groups": [{"cpu_pct": float(rng.normal(15, 2))}],
                }
            ]

            (series / "pipeline_analysis" / f"{stamp}.json").write_text(json.dumps(analysis))
            (series / "health" / f"{stamp}.json").write_text("{}")

            # Per-element snapshot
            elements = {
                "pipeline_0": {
                    "element_0": {
                        "diagnostics": {
                            "payload": {
                                "stutter_events": int(rng.integers(0, 3)),
                                "freeze_events": int(rng.integers(0, 2)),
                                "max_freeze_ms": float(rng.uniform(0, 50)),
                                "stutter_ratio": float(rng.uniform(0, 0.1)),
                            }
                        }
                    }
                }
            }
            (series / "per_element" / f"{stamp}.json").write_text(json.dumps(elements))

            samples.append({"index": s_idx, "unix": unix_ts, "t_rel_s": float(s_idx)})

        manifest = {"samples": samples}
        (rep_dir / "manifest.json").write_text(json.dumps(manifest))

    return tmp_path
