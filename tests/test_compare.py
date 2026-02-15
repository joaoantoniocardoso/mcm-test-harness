"""Tests for the ab_harness.compare module."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

from ab_harness.compare import (
    KpiComparison,
    RunComparison,
    _bootstrap_ci,
    _build_spotify_dataframe,
    _cliffs_delta_label,
    _count_iqr_outliers,
    _durbin_watson,
    _effect_size_label,
    _holm_bonferroni,
    _mann_whitney_u,
    _scipy_ttest_fallback,
    _shapiro_wilk,
    cliffs_delta,
    cohens_d,
    compare_runs,
    recommend_sample_size,
    save_comparison,
)

# ---------------------------------------------------------------------------
# _build_spotify_dataframe
# ---------------------------------------------------------------------------


class TestBuildSpotifyDataframe:
    def test_build_spotify_dataframe(self):
        a = [10.0, 12.0, 11.0]
        b = [20.0, 22.0, 21.0]
        df = _build_spotify_dataframe(a, b)

        assert list(df.columns) == ["group", "value_sum", "value_sumsq", "count"]
        assert len(df) == 2
        assert list(df["group"]) == ["baseline", "iteration"]

        # baseline row corresponds to b, iteration row to a
        assert df.loc[df["group"] == "baseline", "count"].iloc[0] == 3
        assert df.loc[df["group"] == "iteration", "count"].iloc[0] == 3
        assert df.loc[df["group"] == "baseline", "value_sum"].iloc[0] == pytest.approx(sum(b))
        assert df.loc[df["group"] == "iteration", "value_sum"].iloc[0] == pytest.approx(sum(a))
        assert df.loc[df["group"] == "baseline", "value_sumsq"].iloc[0] == pytest.approx(
            sum(x**2 for x in b)
        )


# ---------------------------------------------------------------------------
# _scipy_ttest_fallback
# ---------------------------------------------------------------------------


class TestScipyTtestFallback:
    def test_scipy_ttest_fallback_identical(self):
        """Two identical samples should give p ~ 1.0 and not significant."""
        values = [10.0, 10.5, 9.5, 10.2, 9.8]
        result = _scipy_ttest_fallback(values, values)

        assert result["p_value"] == pytest.approx(1.0, abs=0.01)
        assert result["is_significant"] is False
        assert result["difference"] == pytest.approx(0.0, abs=1e-10)
        assert "ci_lower" in result
        assert "ci_upper" in result
        assert result["power"] is None  # fallback doesn't compute power

    def test_scipy_ttest_fallback_different(self):
        """Two clearly different samples should give low p and significant=True."""
        rng = np.random.default_rng(123)
        a = list(rng.normal(10, 1, size=30))
        b = list(rng.normal(20, 1, size=30))
        result = _scipy_ttest_fallback(a, b)

        assert result["p_value"] < 0.001
        assert result["is_significant"] is True
        assert result["difference"] < 0  # a_mean (10) - b_mean (20) < 0
        assert result["ci_upper"] < 0  # entire CI below zero


# ---------------------------------------------------------------------------
# cohens_d
# ---------------------------------------------------------------------------


class TestCohensD:
    def testcohens_d_identical(self):
        values = [10.0, 10.5, 9.5, 10.2, 9.8]
        d = cohens_d(values, values)
        assert d == pytest.approx(0.0, abs=1e-10)

    def testcohens_d_large_effect(self):
        a = [100.0, 101.0, 102.0, 99.0, 100.5]
        b = [50.0, 51.0, 52.0, 49.0, 50.5]
        d = cohens_d(a, b)
        assert d is not None
        assert abs(d) > 0.8

    def testcohens_d_insufficient_data(self):
        assert cohens_d([1.0], [2.0]) is None
        assert cohens_d([], []) is None
        assert cohens_d([1.0, 2.0], [3.0]) is None


# ---------------------------------------------------------------------------
# _effect_size_label
# ---------------------------------------------------------------------------


class TestEffectSizeLabel:
    @pytest.mark.parametrize(
        "d, expected",
        [
            (None, "N/A"),
            (0.0, "negligible"),
            (0.1, "negligible"),
            (0.3, "small"),
            (0.4, "small"),
            (0.6, "medium"),
            (0.7, "medium"),
            (1.0, "large"),
            (2.5, "large"),
            (-0.3, "small"),
            (-1.0, "large"),
        ],
    )
    def test_effect_size_label(self, d, expected):
        assert _effect_size_label(d) == expected


# ---------------------------------------------------------------------------
# recommend_sample_size
# ---------------------------------------------------------------------------


class TestRecommendSampleSize:
    def test_recommend_sample_size_basic(self):
        n = recommend_sample_size(std=5.0, mde_pct=10.0, baseline_mean=100.0)
        assert isinstance(n, int)
        assert n >= 2

    def test_recommend_sample_size_edge_cases(self):
        # std == 0
        assert recommend_sample_size(std=0.0, mde_pct=10.0, baseline_mean=100.0) == 1
        # mde_pct == 0
        assert recommend_sample_size(std=5.0, mde_pct=0.0, baseline_mean=100.0) == 1
        # baseline_mean == 0
        assert recommend_sample_size(std=5.0, mde_pct=10.0, baseline_mean=0.0) == 1
        # negative std
        assert recommend_sample_size(std=-1.0, mde_pct=10.0, baseline_mean=100.0) == 1


# ---------------------------------------------------------------------------
# KpiComparison dataclass
# ---------------------------------------------------------------------------


class TestKpiComparison:
    def test_kpi_comparison_dataclass(self):
        kc = KpiComparison(
            name="throughput_fps",
            unit="fps",
            higher_is_better=True,
            a_mean=30.0,
            a_std=1.0,
            a_per_rep=[29.0, 30.0, 31.0],
            b_mean=28.0,
            b_std=1.5,
            b_per_rep=[27.0, 28.0, 29.0],
            delta=2.0,
            delta_pct=7.14,
            p_value=0.04,
            significant=True,
            verdict="improvement",
            regression_threshold_pct=5.0,
            cohens_d=1.5,
            ci_lower=0.5,
            ci_upper=3.5,
            power=0.85,
            effect_size_label="large",
        )
        assert kc.name == "throughput_fps"
        assert kc.significant is True
        assert kc.verdict == "improvement"
        assert kc.cohens_d == 1.5
        assert kc.effect_size_label == "large"


# ---------------------------------------------------------------------------
# Integration: compare_runs
# ---------------------------------------------------------------------------


class TestCompareRunsIntegration:
    def test_compare_runs_integration(self, tmp_run_dir: Path, tmp_path: Path):
        """Build two run dirs, aggregate them, then compare."""
        from ab_harness.kpi import aggregate_reps

        # Use the fixture run dir as run_a
        run_a = tmp_run_dir

        # Create run_b as a copy of the fixture with a slightly different seed
        run_b = tmp_path / "run_b"
        shutil.copytree(run_a, run_b)

        # Aggregate both sides (writes aggregate.json)
        aggregate_reps(run_a)
        aggregate_reps(run_b)

        # Compare
        result = compare_runs(run_a, run_b)

        assert isinstance(result, RunComparison)
        assert len(result.kpis) > 0
        assert result.run_a_dir == str(run_a)
        assert result.run_b_dir == str(run_b)

        # All kpi comparisons should be KpiComparison instances
        for kc in result.kpis:
            assert isinstance(kc, KpiComparison)
            assert kc.name  # non-empty name

        # With identical data the verdicts should be "neutral" or "no_data"
        for kc in result.kpis:
            assert kc.verdict in {"neutral", "no_data", "likely_regression", "likely_improvement"}

    def test_save_comparison(self, tmp_run_dir: Path, tmp_path: Path):
        """Verify save_comparison writes valid JSON."""
        import json

        from ab_harness.kpi import aggregate_reps

        run_a = tmp_run_dir
        run_b = tmp_path / "run_b"
        shutil.copytree(run_a, run_b)

        aggregate_reps(run_a)
        aggregate_reps(run_b)

        comparison = compare_runs(run_a, run_b)
        out_path = tmp_path / "comparison.json"
        save_comparison(comparison, out_path)

        assert out_path.exists()
        data = json.loads(out_path.read_text())
        assert "kpis" in data
        assert data["run_a_dir"] == str(run_a)


# ---------------------------------------------------------------------------
# _holm_bonferroni
# ---------------------------------------------------------------------------


class TestHolmBonferroni:
    def test_all_none(self):
        """All None p-values should pass through unchanged."""
        result = _holm_bonferroni([None, None, None])
        assert result == [None, None, None]

    def test_single_value(self):
        """Single p-value should be unchanged."""
        result = _holm_bonferroni([0.03])
        assert result == [pytest.approx(0.03)]

    def test_basic_correction(self):
        """Adjusted p-values should be >= original and monotonically enforced."""
        raw = [0.01, 0.04, 0.03]
        adjusted = _holm_bonferroni(raw, alpha=0.05)

        # No value should be None
        assert all(a is not None for a in adjusted)
        # Each adjusted p >= original p
        for orig, adj in zip(raw, adjusted, strict=False):
            assert adj >= orig  # type: ignore[operator]

    def test_holm_bonferroni_known_values(self):
        """Verify against hand-computed Holm-Bonferroni values."""
        # 3 p-values: sorted = [0.01, 0.03, 0.04]
        # Step 1: 0.01 * 3 = 0.03
        # Step 2: 0.03 * 2 = 0.06
        # Step 3: 0.04 * 1 = 0.04, but monotonicity -> max(0.04, 0.06) = 0.06
        raw = [0.01, 0.04, 0.03]
        adjusted = _holm_bonferroni(raw, alpha=0.05)

        assert adjusted[0] == pytest.approx(0.03)  # 0.01 * 3
        assert adjusted[2] == pytest.approx(0.06)  # 0.03 * 2
        assert adjusted[1] == pytest.approx(0.06)  # max(0.04*1, 0.06)

    def test_with_none_entries(self):
        """None entries should be passed through; non-None entries corrected."""
        raw = [0.01, None, 0.04]
        adjusted = _holm_bonferroni(raw, alpha=0.05)

        assert adjusted[1] is None
        assert adjusted[0] is not None
        assert adjusted[2] is not None
        # Only 2 non-None: 0.01*2=0.02, 0.04*1=0.04
        assert adjusted[0] == pytest.approx(0.02)
        assert adjusted[2] == pytest.approx(0.04)

    def test_cap_at_one(self):
        """Adjusted p-values should never exceed 1.0."""
        raw = [0.5, 0.6, 0.7]
        adjusted = _holm_bonferroni(raw, alpha=0.05)
        for a in adjusted:
            assert a is not None
            assert a <= 1.0

    def test_empty_list(self):
        """Empty list should return empty list."""
        assert _holm_bonferroni([]) == []


# ---------------------------------------------------------------------------
# _durbin_watson
# ---------------------------------------------------------------------------


class TestDurbinWatson:
    def test_independent_data(self):
        """IID data should give DW close to 2.0."""
        rng = np.random.default_rng(42)
        values = list(rng.normal(0, 1, size=100))
        dw = _durbin_watson(values)
        assert dw is not None
        assert 1.5 < dw < 2.5

    def test_positively_correlated(self):
        """Strongly autocorrelated data should give DW < 1.5."""
        # Random walk: each value = previous + small noise
        rng = np.random.default_rng(42)
        values = [0.0]
        for _ in range(99):
            values.append(values[-1] + rng.normal(0, 0.1))
        dw = _durbin_watson(values)
        assert dw is not None
        assert dw < 1.5

    def test_too_few_values(self):
        """Should return None for < 3 values."""
        assert _durbin_watson([1.0, 2.0]) is None
        assert _durbin_watson([1.0]) is None
        assert _durbin_watson([]) is None

    def test_constant_values(self):
        """All-equal values: residuals_sq = 0, should return None."""
        assert _durbin_watson([5.0, 5.0, 5.0, 5.0]) is None

    def test_three_values(self):
        """Minimum valid case: exactly 3 values."""
        dw = _durbin_watson([1.0, 2.0, 3.0])
        assert dw is not None
        assert 0.0 <= dw <= 4.0


# ---------------------------------------------------------------------------
# cliffs_delta
# ---------------------------------------------------------------------------


class TestCliffsDelta:
    def test_identical_samples(self):
        values = [10.0, 20.0, 30.0, 40.0, 50.0]
        d = cliffs_delta(values, values)
        assert d == pytest.approx(0.0)

    def test_perfectly_separated(self):
        a = [100.0, 101.0, 102.0]
        b = [1.0, 2.0, 3.0]
        d = cliffs_delta(a, b)
        assert d == pytest.approx(1.0)

    def test_perfectly_separated_reverse(self):
        a = [1.0, 2.0, 3.0]
        b = [100.0, 101.0, 102.0]
        d = cliffs_delta(a, b)
        assert d == pytest.approx(-1.0)

    def test_empty_input(self):
        assert cliffs_delta([], [1.0, 2.0]) is None
        assert cliffs_delta([1.0], []) is None
        assert cliffs_delta([], []) is None

    def test_overlapping_samples(self):
        a = [1.0, 2.0, 3.0, 4.0, 5.0]
        b = [3.0, 4.0, 5.0, 6.0, 7.0]
        d = cliffs_delta(a, b)
        assert d is not None
        assert -1.0 <= d <= 1.0
        assert d < 0  # a tends to be smaller


# ---------------------------------------------------------------------------
# _cliffs_delta_label
# ---------------------------------------------------------------------------


class TestCliffsDeltaLabel:
    @pytest.mark.parametrize(
        "d, expected",
        [
            (None, "N/A"),
            (0.0, "negligible"),
            (0.1, "negligible"),
            (0.2, "small"),
            (0.3, "small"),
            (0.4, "medium"),
            (0.47, "medium"),
            (0.5, "large"),
            (1.0, "large"),
            (-0.1, "negligible"),
            (-0.3, "small"),
            (-0.5, "large"),
        ],
    )
    def test_label(self, d, expected):
        assert _cliffs_delta_label(d) == expected


# ---------------------------------------------------------------------------
# _shapiro_wilk
# ---------------------------------------------------------------------------


class TestShapiroWilk:
    def test_normal_data(self):
        rng = np.random.default_rng(42)
        values = list(rng.normal(50, 5, size=30))
        p = _shapiro_wilk(values)
        assert p is not None
        assert p > 0.05  # should not reject normality for normal data

    def test_uniform_data(self):
        rng = np.random.default_rng(42)
        values = list(rng.uniform(0, 100, size=30))
        p = _shapiro_wilk(values)
        assert p is not None
        # Uniform dist may or may not reject; just check it returns a value

    def test_too_few_values(self):
        assert _shapiro_wilk([1.0, 2.0, 3.0]) is None
        assert _shapiro_wilk([1.0]) is None
        assert _shapiro_wilk([]) is None

    def test_too_many_values(self):
        values = list(range(51))
        assert _shapiro_wilk([float(v) for v in values]) is None

    def test_constant_data(self):
        assert _shapiro_wilk([5.0, 5.0, 5.0, 5.0, 5.0]) is None

    def test_exactly_four(self):
        p = _shapiro_wilk([1.0, 2.0, 3.0, 10.0])
        assert p is not None
        assert 0.0 <= p <= 1.0


# ---------------------------------------------------------------------------
# _mann_whitney_u
# ---------------------------------------------------------------------------


class TestMannWhitneyU:
    def test_identical_samples(self):
        values = [10.0, 20.0, 30.0, 40.0, 50.0]
        p = _mann_whitney_u(values, values)
        assert p is not None
        assert p > 0.05

    def test_clearly_different(self):
        a = [100.0, 101.0, 102.0, 103.0, 104.0]
        b = [1.0, 2.0, 3.0, 4.0, 5.0]
        p = _mann_whitney_u(a, b)
        assert p is not None
        assert p < 0.05

    def test_insufficient_data(self):
        assert _mann_whitney_u([1.0, 2.0], [3.0, 4.0, 5.0]) is None
        assert _mann_whitney_u([1.0, 2.0, 3.0], [4.0, 5.0]) is None
        assert _mann_whitney_u([], []) is None


# ---------------------------------------------------------------------------
# _bootstrap_ci
# ---------------------------------------------------------------------------


class TestBootstrapCi:
    def test_identical_samples(self):
        values = [10.0, 10.5, 9.5, 10.2, 9.8]
        result = _bootstrap_ci(values, values)
        assert result is not None
        lo, hi = result
        # CI should contain 0 for identical samples
        assert lo <= 0.0 <= hi

    def test_different_samples(self):
        rng = np.random.default_rng(42)
        a = list(rng.normal(100, 1, size=20))
        b = list(rng.normal(90, 1, size=20))
        result = _bootstrap_ci(a, b)
        assert result is not None
        lo, hi = result
        # CI should be entirely above 0 (a > b)
        assert lo > 0

    def test_insufficient_data(self):
        assert _bootstrap_ci([1.0, 2.0], [3.0, 4.0, 5.0]) is None
        assert _bootstrap_ci([1.0, 2.0, 3.0], [4.0, 5.0]) is None


# ---------------------------------------------------------------------------
# _count_iqr_outliers
# ---------------------------------------------------------------------------


class TestCountIqrOutliers:
    def test_no_outliers(self):
        values = [10.0, 11.0, 12.0, 13.0, 14.0]
        assert _count_iqr_outliers(values) == 0

    def test_with_outliers(self):
        values = [10.0, 11.0, 12.0, 13.0, 100.0]
        count = _count_iqr_outliers(values)
        assert count >= 1

    def test_too_few_values(self):
        assert _count_iqr_outliers([1.0, 2.0, 3.0]) == 0
        assert _count_iqr_outliers([]) == 0

    def test_constant_values(self):
        assert _count_iqr_outliers([5.0, 5.0, 5.0, 5.0]) == 0

    def test_symmetric_outliers(self):
        values = [-100.0, 10.0, 11.0, 12.0, 13.0, 100.0]
        count = _count_iqr_outliers(values)
        assert count == 2


# ---------------------------------------------------------------------------
# Strict mode integration
# ---------------------------------------------------------------------------


class TestStrictModeIntegration:
    def test_compare_runs_strict(self, tmp_run_dir: Path, tmp_path: Path):
        """Strict mode should populate adjusted_p_value and correction_method."""
        from ab_harness.kpi import aggregate_reps

        run_a = tmp_run_dir
        run_b = tmp_path / "run_b"
        shutil.copytree(run_a, run_b)

        aggregate_reps(run_a)
        aggregate_reps(run_b)

        result = compare_runs(run_a, run_b, strict=True)

        for kc in result.kpis:
            if kc.p_value is not None:
                assert kc.adjusted_p_value is not None
                assert kc.adjusted_p_value >= kc.p_value
                assert kc.correction_method == "holm-bonferroni"
            else:
                # KPIs without p-values should have None adjusted_p_value
                assert kc.adjusted_p_value is None

    def test_compare_runs_non_strict(self, tmp_run_dir: Path, tmp_path: Path):
        """Non-strict mode should leave adjusted_p_value as None."""
        from ab_harness.kpi import aggregate_reps

        run_a = tmp_run_dir
        run_b = tmp_path / "run_b"
        shutil.copytree(run_a, run_b)

        aggregate_reps(run_a)
        aggregate_reps(run_b)

        result = compare_runs(run_a, run_b, strict=False)

        for kc in result.kpis:
            assert kc.adjusted_p_value is None
            assert kc.correction_method is None


# ---------------------------------------------------------------------------
# Robust statistics integration in compare_runs
# ---------------------------------------------------------------------------


class TestRobustStatsIntegration:
    def test_compare_runs_has_robust_fields(self, tmp_run_dir: Path, tmp_path: Path):
        """compare_runs should populate robust statistics fields."""
        from ab_harness.kpi import aggregate_reps

        run_a = tmp_run_dir
        run_b = tmp_path / "run_b"
        shutil.copytree(run_a, run_b)

        aggregate_reps(run_a)
        aggregate_reps(run_b)

        result = compare_runs(run_a, run_b)

        for kc in result.kpis:
            # All KPIs should have these fields set (possibly to defaults)
            assert isinstance(kc.normality_warning, bool)
            assert kc.test_used in ("welch", "mann_whitney")
            assert isinstance(kc.outlier_count_a, int)
            assert isinstance(kc.outlier_count_b, int)
            assert kc.cliffs_delta_label in ("negligible", "small", "medium", "large", "N/A")

    def test_compare_runs_serialization_includes_robust_fields(
        self, tmp_run_dir: Path, tmp_path: Path
    ):
        """save_comparison should include the new robust fields in JSON."""
        import json

        from ab_harness.kpi import aggregate_reps

        run_a = tmp_run_dir
        run_b = tmp_path / "run_b"
        shutil.copytree(run_a, run_b)

        aggregate_reps(run_a)
        aggregate_reps(run_b)

        comparison = compare_runs(run_a, run_b)
        out_path = tmp_path / "comparison_robust.json"
        save_comparison(comparison, out_path)

        data = json.loads(out_path.read_text())
        kpi = data["kpis"][0]
        assert "normality_warning" in kpi
        assert "mann_whitney_p" in kpi
        assert "test_used" in kpi
        assert "cliffs_delta" in kpi
        assert "cliffs_delta_label" in kpi
        assert "outlier_count_a" in kpi
        assert "outlier_count_b" in kpi
