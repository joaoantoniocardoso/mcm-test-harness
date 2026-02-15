"""Tests for the ab_harness.kpi module."""

from __future__ import annotations

from pathlib import Path

import pytest

from ab_harness.kpi import (
    KPI_BY_NAME,
    KPI_REGISTRY,
    KpiThreshold,
    SampleKpis,
    _load_json,
    aggregate_reps,
    extract_sample_kpis,
    load_aggregate,
    summarize_rep,
)

# ---------------------------------------------------------------------------
# KPI registry
# ---------------------------------------------------------------------------


class TestKpiRegistry:
    def test_kpi_registry_not_empty(self):
        assert len(KPI_REGISTRY) > 0

    def test_kpi_by_name_consistency(self):
        """Every entry in KPI_REGISTRY should be present in KPI_BY_NAME."""
        for kpi in KPI_REGISTRY:
            assert kpi.name in KPI_BY_NAME
            assert KPI_BY_NAME[kpi.name] is kpi

    def test_kpi_threshold_dataclass(self):
        kt = KpiThreshold(
            name="test_kpi",
            unit="ms",
            higher_is_better=False,
            regression_pct=10.0,
            description="A test KPI",
        )
        assert kt.name == "test_kpi"
        assert kt.unit == "ms"
        assert kt.higher_is_better is False
        assert kt.regression_pct == 10.0
        assert kt.description == "A test KPI"


# ---------------------------------------------------------------------------
# extract_sample_kpis
# ---------------------------------------------------------------------------


class TestExtractSampleKpis:
    def test_extract_sample_kpis_full(self):
        """Pass a realistic analysis_snapshot and verify pipeline/system KPIs."""
        analysis = [
            {
                "summary": {
                    "throughput_fps": 30.5,
                    "total_pipeline_freshness_delay_ms": 48.2,
                    "causal_latency_health": "ok",
                },
                "pipeline_cpu_pct": 22.1,
                "system": {
                    "current_cpu_pct": 38.0,
                    "current_load_1m": 1.2,
                    "current_mem_used_pct": 52.0,
                    "current_temperature_c": 58.0,
                },
                "elements": {},
                "thread_groups": [{"cpu_pct": 14.5}],
            }
        ]

        kpis = extract_sample_kpis(
            analysis_snapshot=analysis,
            health_snapshot={},
            element_snapshot={},
            t_rel_s=0.0,
        )

        assert isinstance(kpis, SampleKpis)
        assert kpis.throughput_fps == 30.5
        assert kpis.freshness_delay_ms == 48.2
        assert kpis.pipeline_cpu_pct == 22.1
        assert kpis.system_cpu_pct == 38.0
        assert kpis.system_load_1m == 1.2
        assert kpis.system_mem_used_pct == 52.0
        assert kpis.system_temperature_c == 58.0
        assert kpis.top_thread_cpu_pct == 14.5
        assert kpis.causal_latency_health == "ok"
        assert kpis.t_rel_s == 0.0

    def test_extract_sample_kpis_empty(self):
        """All None/empty inputs should yield None for all KPI fields."""
        kpis = extract_sample_kpis(
            analysis_snapshot=None,
            health_snapshot=None,
            element_snapshot=None,
            t_rel_s=5.0,
        )

        assert kpis.t_rel_s == 5.0
        assert kpis.throughput_fps is None
        assert kpis.freshness_delay_ms is None
        assert kpis.pipeline_cpu_pct is None
        assert kpis.system_cpu_pct is None
        assert kpis.system_load_1m is None
        assert kpis.system_mem_used_pct is None
        assert kpis.system_temperature_c is None
        assert kpis.total_stutter_events is None
        assert kpis.total_freeze_events is None
        assert kpis.max_freeze_ms is None
        assert kpis.max_stutter_ratio is None
        assert kpis.top_thread_cpu_pct is None
        assert kpis.thumbnail_success_pct is None

    def test_extract_sample_kpis_elements(self):
        """Pass element_snapshot with diagnostics, verify stutter/freeze fields."""
        element_snapshot = {
            "pipeline_0": {
                "elem_a": {
                    "diagnostics": {
                        "payload": {
                            "stutter_events": 3,
                            "freeze_events": 1,
                            "max_freeze_ms": 42.0,
                            "stutter_ratio": 0.05,
                        }
                    }
                },
                "elem_b": {
                    "diagnostics": {
                        "payload": {
                            "stutter_events": 2,
                            "freeze_events": 0,
                            "max_freeze_ms": 10.0,
                            "stutter_ratio": 0.02,
                        }
                    }
                },
            }
        }

        kpis = extract_sample_kpis(
            analysis_snapshot=None,
            health_snapshot=None,
            element_snapshot=element_snapshot,
            t_rel_s=1.0,
        )

        assert kpis.total_stutter_events == 5  # 3 + 2
        assert kpis.total_freeze_events == 1  # 1 + 0
        assert kpis.max_freeze_ms == 42.0  # max(42, 10)
        assert kpis.max_stutter_ratio == 0.05  # max(0.05, 0.02)

    def test_extract_sample_kpis_thumbnails(self):
        """Thumbnail success percentage should be computed correctly."""
        thumbnail_snapshot = [
            {"ok": True},
            {"ok": True},
            {"ok": False},
            {"ok": True},
        ]

        kpis = extract_sample_kpis(
            analysis_snapshot=None,
            health_snapshot=None,
            element_snapshot=None,
            t_rel_s=2.0,
            thumbnail_snapshot=thumbnail_snapshot,
        )

        assert kpis.thumbnail_success_pct == pytest.approx(75.0)

    def test_extract_sample_kpis_thumbnails_empty_list(self):
        """Empty thumbnail list should leave the field as None."""
        kpis = extract_sample_kpis(
            analysis_snapshot=None,
            health_snapshot=None,
            element_snapshot=None,
            t_rel_s=0.0,
            thumbnail_snapshot=[],
        )
        assert kpis.thumbnail_success_pct is None


# ---------------------------------------------------------------------------
# summarize_rep / aggregate_reps
# ---------------------------------------------------------------------------


class TestSummarizeRep:
    def test_summarize_rep(self, tmp_run_dir: Path):
        rep_dir = tmp_run_dir / "rep_000"
        summary = summarize_rep(rep_dir, rep_index=0)

        assert summary.rep_index == 0
        assert summary.sample_count == 10
        assert isinstance(summary.kpis, dict)
        # Should have entries for the KPIs our fixture generates
        assert "throughput_fps" in summary.kpis
        assert summary.kpis["throughput_fps"]["mean"] is not None
        assert summary.kpis["throughput_fps"]["count"] == 10


class TestAggregateReps:
    def test_aggregate_reps(self, tmp_run_dir: Path):
        agg = aggregate_reps(tmp_run_dir)

        assert agg.rep_count == 3
        assert len(agg.per_rep) == 3
        assert isinstance(agg.aggregate, dict)

        # Check that throughput_fps was aggregated
        assert "throughput_fps" in agg.aggregate
        tp = agg.aggregate["throughput_fps"]
        assert tp["mean"] is not None
        assert tp["rep_count"] == 3
        assert len(tp["per_rep_means"]) == 3

        # aggregate.json should have been written
        assert (tmp_run_dir / "aggregate.json").exists()

    def test_aggregate_reps_no_reps(self, tmp_path: Path):
        """Empty directory should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="No rep_"):
            aggregate_reps(tmp_path)


# ---------------------------------------------------------------------------
# load_aggregate / _load_json
# ---------------------------------------------------------------------------


class TestLoadAggregate:
    def test_load_aggregate_missing(self, tmp_path: Path):
        """Should raise FileNotFoundError when no aggregate.json exists."""
        with pytest.raises(FileNotFoundError, match=r"No aggregate\.json"):
            load_aggregate(tmp_path)

    def test_load_aggregate_roundtrip(self, tmp_run_dir: Path):
        """Aggregate, then load the result back."""
        aggregate_reps(tmp_run_dir)
        data = load_aggregate(tmp_run_dir)
        assert "aggregate" in data
        assert "rep_count" in data
        assert data["rep_count"] == 3


class TestLoadJson:
    def test_load_json_missing(self, tmp_path: Path):
        result = _load_json(tmp_path / "nonexistent.json")
        assert result is None

    def test_load_json_invalid(self, tmp_path: Path):
        bad = tmp_path / "bad.json"
        bad.write_text("not json at all {{{")
        result = _load_json(bad)
        assert result is None

    def test_load_json_valid(self, tmp_path: Path):
        good = tmp_path / "good.json"
        good.write_text('{"key": 42}')
        result = _load_json(good)
        assert result == {"key": 42}
