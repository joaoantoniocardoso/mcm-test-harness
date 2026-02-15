"""Unit tests for ab_harness.sequential_testing module."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ab_harness.sequential_testing import (
    AdaptiveState,
    InterimResult,
    MsprtAdaptiveState,
    ReplayResult,
    _robust_pooled_sigma,
    check_interim,
    compute_futility_boundary,
    compute_gst_boundaries,
    create_adaptive_state,
    create_msprt_state,
    format_replay_summary,
    replay_all_methods,
)

# ---------------------------------------------------------------------------
# Futility boundary
# ---------------------------------------------------------------------------


class TestComputeFutilityBoundary:
    """Tests for the heuristic futility stopping rule."""

    @pytest.mark.parametrize(
        "n_per_side, max_reps",
        [
            (1, 100),  # 1% budget
            (10, 100),  # 10% budget
            (29, 100),  # 29% budget (just under 30%)
        ],
        ids=["1pct", "10pct", "29pct"],
    )
    def test_early_returns_one(self, n_per_side: int, max_reps: int) -> None:
        """Before 30% budget is spent, futility should never trigger (threshold=1.0)."""
        threshold = compute_futility_boundary(
            n_per_side,
            max_reps,
            mde_frac=0.02,
        )
        assert threshold == 1.0, (
            f"Expected 1.0 for n={n_per_side}/{max_reps} "
            f"(t={n_per_side / max_reps:.2f}), got {threshold}"
        )

    def test_midpoint_between_half_and_eight_tenths(self) -> None:
        """At ~65% budget the threshold should lie between 0.5 and 0.8."""
        threshold = compute_futility_boundary(
            n_per_side=65,
            max_reps=100,
            mde_frac=0.02,
        )
        assert 0.5 <= threshold <= 0.8, (
            f"Expected threshold in [0.5, 0.8] at t=0.65, got {threshold}"
        )

    def test_end_returns_half(self) -> None:
        """At max_reps (t=1.0) the threshold should be 0.5."""
        threshold = compute_futility_boundary(
            n_per_side=100,
            max_reps=100,
            mde_frac=0.02,
        )
        assert threshold == pytest.approx(0.5, abs=1e-9), f"Expected 0.5 at t=1.0, got {threshold}"

    def test_boundary_at_thirty_pct(self) -> None:
        """At exactly 30% budget, the threshold should be 0.8."""
        threshold = compute_futility_boundary(
            n_per_side=30,
            max_reps=100,
            mde_frac=0.02,
        )
        assert threshold == pytest.approx(0.8, abs=1e-9), (
            f"Expected 0.8 at t=0.30, got {threshold}"
        )


# ---------------------------------------------------------------------------
# GST boundaries (spotify-confidence)
# ---------------------------------------------------------------------------


class TestComputeGstBoundaries:
    """Tests for Group Sequential Testing boundary computation."""

    @pytest.fixture
    def boundaries(self) -> dict[int, float]:
        return compute_gst_boundaries(max_reps=100, look_every=20)

    def test_returns_nonempty_dict(self, boundaries: dict[int, float]) -> None:
        """Should return a non-empty dict with int keys and float values."""
        assert isinstance(boundaries, dict)
        assert len(boundaries) > 0, "Boundaries dict must not be empty"
        for k, v in boundaries.items():
            assert isinstance(k, int), f"Key {k!r} should be int"
            assert isinstance(v, float), f"Value {v!r} should be float"

    def test_includes_max_reps(self, boundaries: dict[int, float]) -> None:
        """The max_reps key must always be present in boundaries."""
        assert 100 in boundaries, (
            f"max_reps=100 not in boundaries keys: {sorted(boundaries.keys())}"
        )

    def test_monotonically_increasing(self, boundaries: dict[int, float]) -> None:
        """Later look points should have larger (more lenient) p-thresholds."""
        sorted_items = sorted(boundaries.items())
        for i in range(1, len(sorted_items)):
            n_prev, p_prev = sorted_items[i - 1]
            n_curr, p_curr = sorted_items[i]
            assert p_curr >= p_prev, (
                f"Boundary not monotonic: p({n_prev})={p_prev:.6f} > p({n_curr})={p_curr:.6f}"
            )

    def test_all_thresholds_positive(self, boundaries: dict[int, float]) -> None:
        """Every p-value threshold must be positive."""
        for n, p in boundaries.items():
            assert p > 0, f"Threshold at n={n} should be positive, got {p}"

    def test_max_reps_included_when_not_multiple(self) -> None:
        """max_reps should be included even when not a multiple of look_every."""
        boundaries = compute_gst_boundaries(max_reps=50, look_every=15)
        assert 50 in boundaries, "max_reps=50 must appear even though 50 is not a multiple of 15"


# ---------------------------------------------------------------------------
# InterimResult dataclass
# ---------------------------------------------------------------------------


class TestInterimResult:
    """Tests for InterimResult.stopped property."""

    @pytest.mark.parametrize(
        "efficacy, futility, expected",
        [
            (False, False, False),
            (True, False, True),
            (False, True, True),
            (True, True, True),
        ],
        ids=["neither", "efficacy", "futility", "both"],
    )
    def test_stopped_property(self, efficacy: bool, futility: bool, expected: bool) -> None:
        """InterimResult.stopped is True iff efficacy or futility is set."""
        result = InterimResult(
            look_number=1,
            reps_per_side=10,
            information_fraction=0.5,
            kpi_name="test_kpi",
            p_value=0.03,
            efficacy_boundary=0.05,
            futility_threshold=0.8,
            stopped_for_efficacy=efficacy,
            stopped_for_futility=futility,
            delta_pct=1.0,
            cohens_d=0.2,
        )
        assert result.stopped is expected


# ---------------------------------------------------------------------------
# AdaptiveState creation
# ---------------------------------------------------------------------------


class TestCreateAdaptiveState:
    """Tests for create_adaptive_state factory function."""

    def test_defaults(self) -> None:
        """Default target_kpis=['system_cpu_pct'] and auto look_every > 0."""
        state = create_adaptive_state(max_reps=100)
        assert state.target_kpis == ["system_cpu_pct"], (
            f"Expected default KPI, got {state.target_kpis}"
        )
        assert state.look_every > 0, "Auto look_every should be positive"
        assert state.max_reps == 100
        assert state.alpha == pytest.approx(0.05)
        assert state.futility_enabled is True
        assert state.stopped is False

    def test_custom_parameters(self) -> None:
        """Custom target_kpis and look_every should be respected."""
        state = create_adaptive_state(
            max_reps=200,
            look_every=25,
            alpha=0.01,
            target_kpis=["throughput_fps", "latency_ms"],
            futility=False,
        )
        assert state.target_kpis == ["throughput_fps", "latency_ms"]
        assert state.look_every == 25
        assert state.alpha == pytest.approx(0.01)
        assert state.futility_enabled is False
        assert state.max_reps == 200

    def test_boundaries_populated(self) -> None:
        """The boundaries dict should be populated after creation."""
        state = create_adaptive_state(max_reps=50, look_every=10)
        assert len(state.boundaries) > 0
        assert 50 in state.boundaries, "max_reps should be in boundaries"


# ---------------------------------------------------------------------------
# AdaptiveState save/load
# ---------------------------------------------------------------------------


class TestAdaptiveStateSave:
    """Tests for AdaptiveState.save() persistence."""

    def test_save_creates_valid_json(self, tmp_path: Path) -> None:
        """save() should write a valid JSON file with all expected keys."""
        state = create_adaptive_state(max_reps=50, look_every=10)
        out_file = tmp_path / "adaptive_state.json"
        state.save(out_file)

        assert out_file.exists(), "JSON file was not created"

        data = json.loads(out_file.read_text())
        expected_keys = {
            "max_reps",
            "look_every",
            "alpha",
            "target_kpis",
            "boundaries",
            "futility_enabled",
            "stopped",
            "stop_reason",
            "stop_kpi",
            "stop_at_rep",
            "interim_results",
        }
        assert expected_keys <= set(data.keys()), (
            f"Missing keys: {expected_keys - set(data.keys())}"
        )
        assert data["max_reps"] == 50
        assert data["look_every"] == 10


# ---------------------------------------------------------------------------
# check_interim
# ---------------------------------------------------------------------------


class TestCheckInterim:
    """Tests for check_interim (GST live analysis)."""

    @pytest.fixture
    def state(self) -> AdaptiveState:
        return create_adaptive_state(
            max_reps=100,
            look_every=20,
            target_kpis=["cpu"],
        )

    def test_not_at_look_point(self, state: AdaptiveState) -> None:
        """Returns None when reps_per_side is not a look point."""
        result = check_interim(
            state,
            a_per_rep={"cpu": [10.0] * 15},
            b_per_rep={"cpu": [10.0] * 15},
            reps_per_side=15,  # not a multiple of 20
        )
        assert result is None, "Expected None for non-look-point rep count"

    def test_at_look_point_similar_data(self, state: AdaptiveState) -> None:
        """At a look point with identical-distribution data, should not stop."""
        rng = np.random.default_rng(123)
        n = 20
        a_data = rng.normal(10.0, 1.0, size=n).tolist()
        b_data = rng.normal(10.0, 1.0, size=n).tolist()

        result = check_interim(
            state,
            a_per_rep={"cpu": a_data},
            b_per_rep={"cpu": b_data},
            reps_per_side=20,
        )
        # With similar distributions the test should either return None
        # (no stopping) or an InterimResult with stopped=False
        if result is not None:
            assert result.stopped is False, "Similar distributions should not trigger stopping"


# ---------------------------------------------------------------------------
# MsprtAdaptiveState
# ---------------------------------------------------------------------------


class TestCreateMsprtState:
    """Tests for create_msprt_state factory."""

    def test_defaults(self) -> None:
        """Default target_kpis=['system_cpu_pct'] and min_n=8."""
        state = create_msprt_state(max_reps=50)
        assert isinstance(state, MsprtAdaptiveState)
        assert state.target_kpis == ["system_cpu_pct"]
        assert state.min_n == 8
        assert state.max_reps == 50
        assert state.alpha == pytest.approx(0.05)
        assert state.sigma == pytest.approx(1.0)
        assert state.futility_enabled is True
        assert state.stopped is False

    def test_custom_parameters(self) -> None:
        """Custom parameters are stored correctly."""
        state = create_msprt_state(
            max_reps=200,
            alpha=0.01,
            target_kpis=["latency"],
            min_n=10,
            sigma=2.5,
            futility=False,
        )
        assert state.target_kpis == ["latency"]
        assert state.min_n == 10
        assert state.alpha == pytest.approx(0.01)
        assert state.sigma == pytest.approx(2.5)
        assert state.futility_enabled is False

    def test_save_creates_valid_json(self, tmp_path: Path) -> None:
        """save() should write a valid JSON file with all expected keys."""
        state = create_msprt_state(max_reps=50)
        out_file = tmp_path / "msprt_state.json"
        state.save(out_file)

        assert out_file.exists(), "JSON file was not created"

        data = json.loads(out_file.read_text())
        expected_keys = {
            "method",
            "max_reps",
            "alpha",
            "target_kpis",
            "min_n",
            "sigma",
            "futility_enabled",
            "stopped",
            "stop_reason",
            "stop_kpi",
            "stop_at_rep",
            "interim_results",
        }
        assert expected_keys <= set(data.keys()), (
            f"Missing keys: {expected_keys - set(data.keys())}"
        )
        assert data["method"] == "msprt"
        assert data["max_reps"] == 50


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------


class TestReplayAllMethods:
    """Tests for replay_all_methods post-hoc analysis."""

    def test_returns_replay_result(self) -> None:
        """Synthetic no-effect data should produce a ReplayResult with both methods."""
        rng = np.random.default_rng(42)
        a = rng.normal(10, 1, size=20).tolist()
        b = rng.normal(10, 1, size=20).tolist()

        result = replay_all_methods(a, b, kpi_name="test_metric")

        assert isinstance(result, ReplayResult)
        assert result.kpi_name == "test_metric"
        assert result.total_reps == 20

        # GST result dict
        assert isinstance(result.gst, dict)
        assert "method" in result.gst
        assert "stopped" in result.gst
        assert "trajectory" in result.gst

        # mSPRT result dict
        assert isinstance(result.msprt, dict)
        assert "method" in result.msprt
        assert "stopped" in result.msprt


class TestFormatReplaySummary:
    """Tests for format_replay_summary human-readable output."""

    def test_contains_method_names(self) -> None:
        """Output string should mention both method names."""
        result = ReplayResult(
            kpi_name="cpu",
            total_reps=50,
            gst={"method": "GST_spotify", "stopped": False, "stop_at": None},
            msprt={
                "method": "msprt",
                "stopped": False,
                "stop_at": None,
            },
        )
        summary = format_replay_summary(result)

        assert isinstance(summary, str)
        assert "GST" in summary, "Summary should mention GST"
        assert "mSPRT" in summary, "Summary should mention mSPRT"

    def test_contains_kpi_name(self) -> None:
        """Output should include the KPI name."""
        result = ReplayResult(
            kpi_name="throughput_fps",
            total_reps=30,
            gst={"method": "GST_spotify", "stopped": False, "stop_at": None},
            msprt={
                "method": "msprt",
                "stopped": False,
                "stop_at": None,
            },
        )
        summary = format_replay_summary(result)
        assert "throughput_fps" in summary


# ---------------------------------------------------------------------------
# Robust pooled sigma (MAD-based)
# ---------------------------------------------------------------------------


class TestRobustPooledSigma:
    """Tests for _robust_pooled_sigma MAD-based variance estimator."""

    def test_normal_data(self) -> None:
        """For normal data, MAD-based sigma should approximate the true std."""
        rng = np.random.default_rng(42)
        true_sigma = 3.0
        x = rng.normal(10, true_sigma, size=200)
        y = rng.normal(10, true_sigma, size=200)

        sigma = _robust_pooled_sigma(x, y)
        assert sigma is not None
        assert abs(sigma - true_sigma) < 1.0, (
            f"MAD sigma={sigma:.2f} should be close to true sigma={true_sigma}"
        )

    def test_heavy_tailed_data(self) -> None:
        """MAD sigma should be more stable than classical for outlier-heavy data."""
        rng = np.random.default_rng(42)
        # Normal data with injected outliers
        x_clean = rng.normal(10, 1, size=50)
        y_clean = rng.normal(10, 1, size=50)

        # Inject extreme outliers
        x_dirty = x_clean.copy()
        y_dirty = y_clean.copy()
        x_dirty[0] = 1000.0
        y_dirty[0] = -500.0

        mad_sigma = _robust_pooled_sigma(x_dirty, y_dirty)
        classical_sigma = float(np.sqrt((np.var(x_dirty, ddof=1) + np.var(y_dirty, ddof=1)) / 2.0))

        assert mad_sigma is not None
        # MAD sigma should be much smaller than the classical one (outlier-inflated)
        assert mad_sigma < classical_sigma, (
            f"MAD sigma ({mad_sigma:.2f}) should be < "
            f"classical sigma ({classical_sigma:.2f}) with outliers"
        )

    def test_constant_data(self) -> None:
        """All-constant data: MAD=0, should fall back to classical (also 0 -> None)."""
        x = np.array([5.0, 5.0, 5.0, 5.0])
        y = np.array([5.0, 5.0, 5.0, 5.0])

        sigma = _robust_pooled_sigma(x, y)
        # Both MAD and classical variance are 0 -> returns None
        assert sigma is None or sigma == 0.0

    def test_one_side_constant(self) -> None:
        """One side constant, other varying: should still produce a valid sigma."""
        rng = np.random.default_rng(42)
        x = np.array([5.0, 5.0, 5.0, 5.0, 5.0])
        y = rng.normal(10, 2, size=5)

        sigma = _robust_pooled_sigma(x, y)
        assert sigma is not None
        assert sigma > 0


# ---------------------------------------------------------------------------
# Seed reproducibility (trial ordering)
# ---------------------------------------------------------------------------


class TestSeedReproducibility:
    """Tests for reproducible randomized trial ordering."""

    def test_same_seed_same_order(self) -> None:
        """The same seed should produce the same trial ordering."""
        import random

        trials = list(range(20))

        rng1 = random.Random(42)
        order1 = trials.copy()
        rng1.shuffle(order1)

        rng2 = random.Random(42)
        order2 = trials.copy()
        rng2.shuffle(order2)

        assert order1 == order2, "Same seed should produce identical ordering"

    def test_different_seed_different_order(self) -> None:
        """Different seeds should (almost certainly) produce different orderings."""
        import random

        trials = list(range(20))

        rng1 = random.Random(42)
        order1 = trials.copy()
        rng1.shuffle(order1)

        rng2 = random.Random(999)
        order2 = trials.copy()
        rng2.shuffle(order2)

        assert order1 != order2, "Different seeds should produce different orderings"
