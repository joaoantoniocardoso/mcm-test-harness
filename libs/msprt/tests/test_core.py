"""Tests for msprt.core -- validated against the R mixtureSPRT reference.

Golden values are loaded from ``golden_values.json``, which can be
regenerated independently with the companion R script
``generate_golden_values.R``.  The R script uses R's native ``pnorm``
and ``dnorm`` (not Python/scipy), providing true cross-language
validation of the mSPRT formulas.

References
----------
- R reference: https://github.com/erik-giertz/mixtureSPRT
- Paper: Johari, Pekelis & Walsh (2017), "Peeking at A/B Tests"
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from msprt import MsprtResult, compute_tau, msprt_statistic, msprt_test

# Load golden values once at module level
_GOLDEN_PATH = Path(__file__).parent / "golden_values.json"
_GOLDEN: dict = json.loads(_GOLDEN_PATH.read_text())


# ---------------------------------------------------------------------------
# compute_tau golden-value tests
# ---------------------------------------------------------------------------


class TestComputeTau:
    """Golden-value tests for compute_tau against R calcTau formula."""

    @pytest.mark.parametrize(
        "case",
        _GOLDEN["compute_tau"],
        ids=[
            f"alpha={c['alpha']}_sigma={c['sigma']}_N={c['truncation']}"
            for c in _GOLDEN["compute_tau"]
        ],
    )
    def test_golden_values(self, case: dict) -> None:
        """compute_tau must match the R calcTau reference to high precision."""
        tau = compute_tau(
            alpha=case["alpha"],
            sigma=case["sigma"],
            truncation=case["truncation"],
        )
        assert tau == pytest.approx(case["expected_tau"], rel=1e-10), (
            f"compute_tau({case['alpha']}, {case['sigma']}, {case['truncation']}) = {tau}, "
            f"expected {case['expected_tau']}"
        )

    def test_tau_positive(self) -> None:
        """tau must always be positive."""
        tau = compute_tau(alpha=0.05, sigma=1.0, truncation=100)
        assert tau > 0

    def test_tau_decreases_with_truncation(self) -> None:
        """Larger horizon should give a smaller tau (tighter prior)."""
        tau_small = compute_tau(alpha=0.05, sigma=1.0, truncation=50)
        tau_large = compute_tau(alpha=0.05, sigma=1.0, truncation=500)
        assert tau_large < tau_small

    def test_invalid_alpha(self) -> None:
        with pytest.raises(ValueError, match="alpha"):
            compute_tau(alpha=0.0, sigma=1.0, truncation=100)
        with pytest.raises(ValueError, match="alpha"):
            compute_tau(alpha=1.0, sigma=1.0, truncation=100)

    def test_invalid_sigma(self) -> None:
        with pytest.raises(ValueError, match="sigma"):
            compute_tau(alpha=0.05, sigma=0.0, truncation=100)

    def test_invalid_truncation(self) -> None:
        with pytest.raises(ValueError, match="truncation"):
            compute_tau(alpha=0.05, sigma=1.0, truncation=0)


# ---------------------------------------------------------------------------
# msprt_statistic golden-value tests
# ---------------------------------------------------------------------------


class TestMsprtStatistic:
    """Golden-value tests for Lambda trajectory computation."""

    _STAT = _GOLDEN["msprt_statistic"]
    SIGMA = _STAT["sigma"]
    ALPHA = _STAT["alpha"]
    TRUNCATION = _STAT["truncation"]

    X_NO_EFFECT = np.array(_STAT["no_effect"]["x"])
    Y_NO_EFFECT = np.array(_STAT["no_effect"]["y"])
    EXPECTED_LAMBDA_NO_EFFECT = np.array(_STAT["no_effect"]["expected_lambda"])

    X_LARGE_EFFECT = np.array(_STAT["large_effect"]["x"])
    Y_LARGE_EFFECT = np.array(_STAT["large_effect"]["y"])
    EXPECTED_LAMBDA_LARGE_EFFECT = np.array(_STAT["large_effect"]["expected_lambda"])

    @pytest.fixture
    def tau_no_effect(self) -> float:
        return compute_tau(alpha=self.ALPHA, sigma=self.SIGMA, truncation=self.TRUNCATION)

    def test_no_effect_trajectory(self, tau_no_effect: float) -> None:
        """Lambda trajectory for no-effect data must match golden values."""
        trajectory = msprt_statistic(
            self.X_NO_EFFECT,
            self.Y_NO_EFFECT,
            sigma=self.SIGMA,
            tau=tau_no_effect,
        )
        np.testing.assert_allclose(
            trajectory,
            self.EXPECTED_LAMBDA_NO_EFFECT,
            rtol=1e-10,
            err_msg="No-effect trajectory mismatch",
        )

    def test_large_effect_trajectory(self, tau_no_effect: float) -> None:
        """Lambda trajectory for large-effect data must match golden values."""
        trajectory = msprt_statistic(
            self.X_LARGE_EFFECT,
            self.Y_LARGE_EFFECT,
            sigma=self.SIGMA,
            tau=tau_no_effect,
        )
        np.testing.assert_allclose(
            trajectory,
            self.EXPECTED_LAMBDA_LARGE_EFFECT,
            rtol=1e-10,
            err_msg="Large-effect trajectory mismatch",
        )

    def test_no_effect_stays_below_threshold(self, tau_no_effect: float) -> None:
        """No-effect data should not cross the 1/alpha threshold."""
        trajectory = msprt_statistic(
            self.X_NO_EFFECT,
            self.Y_NO_EFFECT,
            sigma=self.SIGMA,
            tau=tau_no_effect,
        )
        assert np.all(trajectory <= 1.0 / self.ALPHA)

    def test_large_effect_crosses_threshold(self, tau_no_effect: float) -> None:
        """Large-effect data should cross the 1/alpha threshold."""
        trajectory = msprt_statistic(
            self.X_LARGE_EFFECT,
            self.Y_LARGE_EFFECT,
            sigma=self.SIGMA,
            tau=tau_no_effect,
        )
        assert np.any(trajectory > 1.0 / self.ALPHA)

    def test_empty_input(self) -> None:
        result = msprt_statistic(np.array([]), np.array([]), sigma=1.0, tau=0.5)
        assert len(result) == 0

    def test_mismatched_lengths(self) -> None:
        with pytest.raises(ValueError, match="same shape"):
            msprt_statistic(
                np.array([1.0, 2.0]),
                np.array([1.0]),
                sigma=1.0,
                tau=0.5,
            )


# ---------------------------------------------------------------------------
# Prefix consistency (incremental property)
# ---------------------------------------------------------------------------


class TestPrefixConsistency:
    """Verify that computing on a prefix gives the same values."""

    def test_prefix_matches_full(self) -> None:
        """msprt_statistic(x[:k], y[:k]) must equal msprt_statistic(x, y)[:k]."""
        rng = np.random.default_rng(99)
        x = rng.normal(0, 1, size=50)
        y = rng.normal(0, 1, size=50)
        sigma = 1.0
        tau = compute_tau(alpha=0.05, sigma=sigma, truncation=50)

        full_trajectory = msprt_statistic(x, y, sigma=sigma, tau=tau)

        for k in [5, 10, 25, 40]:
            prefix_trajectory = msprt_statistic(x[:k], y[:k], sigma=sigma, tau=tau)
            np.testing.assert_allclose(
                prefix_trajectory,
                full_trajectory[:k],
                rtol=1e-12,
                err_msg=f"Prefix k={k} mismatch",
            )


# ---------------------------------------------------------------------------
# msprt_test high-level API
# ---------------------------------------------------------------------------


class TestMsprtTest:
    """Tests for the high-level msprt_test API."""

    def test_no_effect_does_not_reject(self) -> None:
        """H0 data should not be rejected."""
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, size=100)
        y = rng.normal(0, 1, size=100)
        result = msprt_test(x, y, sigma=1.0, alpha=0.05)

        assert isinstance(result, MsprtResult)
        assert result.n_obs == 100
        assert result.rejected is False
        assert result.n_rejection is None

    def test_large_effect_rejects(self) -> None:
        """Clear effect should be detected."""
        x = np.array([3.1, 2.8, 3.3, 2.9, 3.0, 3.2, 2.7, 3.4, 3.1, 2.8])
        y = np.array([0.9, 1.0, 0.8, 1.1, 0.7, 1.0, 0.9, 0.8, 1.1, 0.7])
        result = msprt_test(x, y, sigma=1.0, alpha=0.05)

        assert result.rejected is True
        assert result.n_rejection == 5  # golden: rejects at observation 5

    def test_min_n_respected(self) -> None:
        """Rejection should not happen before min_n."""
        x = np.array([3.1, 2.8, 3.3, 2.9, 3.0, 3.2, 2.7, 3.4, 3.1, 2.8])
        y = np.array([0.9, 1.0, 0.8, 1.1, 0.7, 1.0, 0.9, 0.8, 1.1, 0.7])
        result = msprt_test(x, y, sigma=1.0, alpha=0.05, min_n=8)

        assert result.rejected is True
        assert result.n_rejection is not None
        assert result.n_rejection >= 8

    def test_truncation_parameter(self) -> None:
        """Custom truncation should affect tau (and therefore trajectory)."""
        x = np.array([1.5, 1.3, 1.4, 1.6, 1.2])
        y = np.array([1.0, 1.0, 1.0, 1.0, 1.0])

        result_short = msprt_test(x, y, sigma=1.0, truncation=5)
        result_long = msprt_test(x, y, sigma=1.0, truncation=1000)

        # Different tau means different trajectories
        assert not np.allclose(result_short.trajectory, result_long.trajectory)

    def test_result_fields(self) -> None:
        """All MsprtResult fields must be populated."""
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([1.0, 2.0, 3.0])
        result = msprt_test(x, y, sigma=1.0)

        assert result.n_obs == 3
        assert len(result.trajectory) == 3
        assert result.alpha == 0.05


# ---------------------------------------------------------------------------
# Type I error Monte Carlo validation
# ---------------------------------------------------------------------------


class TestTypeIErrorRate:
    """Monte Carlo validation that mSPRT controls the false positive rate."""

    @pytest.mark.slow
    @pytest.mark.parametrize("sigma", [1.0, 2.0])
    def test_type_i_error_bounded(self, sigma: float) -> None:
        """Under H0, the rejection rate must be at or below alpha (+margin).

        We use 5000 replications and allow a small margin (0.015) above
        alpha to account for finite-sample Monte Carlo noise.
        """
        n_reps = 5000
        alpha = 0.05
        n_obs = 200
        rng = np.random.default_rng(42)

        rejections = 0
        for _ in range(n_reps):
            x = rng.normal(0, sigma, size=n_obs)
            y = rng.normal(0, sigma, size=n_obs)
            result = msprt_test(x, y, sigma=sigma, alpha=alpha)
            if result.rejected:
                rejections += 1

        rate = rejections / n_reps
        assert rate <= alpha + 0.015, (
            f"Type I error rate {rate:.4f} exceeds alpha={alpha} + 0.015 margin "
            f"(sigma={sigma}, {n_reps} reps, {rejections} rejections)"
        )


# ---------------------------------------------------------------------------
# Type II error (power) Monte Carlo validation
# ---------------------------------------------------------------------------


class TestPower:
    """Monte Carlo validation that mSPRT has reasonable power under H1."""

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "delta, sigma, n_obs, min_power",
        [
            (0.5, 1.0, 200, 0.70),
            (1.0, 1.0, 100, 0.85),
        ],
        ids=["moderate-effect", "large-effect"],
    )
    def test_power_exceeds_minimum(
        self,
        delta: float,
        sigma: float,
        n_obs: int,
        min_power: float,
    ) -> None:
        """Under H1 with known effect, rejection rate must exceed min_power.

        Simulates x ~ N(delta, sigma) vs y ~ N(0, sigma) and checks that
        the mSPRT rejects H0 at least ``min_power`` fraction of the time.

        We use 2000 replications to keep runtime manageable while still
        giving a tight enough estimate (SE ~ 0.01 for power ~ 0.80).
        """
        n_reps = 2000
        alpha = 0.05
        rng = np.random.default_rng(123)

        rejections = 0
        for _ in range(n_reps):
            x = rng.normal(delta, sigma, size=n_obs)
            y = rng.normal(0, sigma, size=n_obs)
            result = msprt_test(x, y, sigma=sigma, alpha=alpha)
            if result.rejected:
                rejections += 1

        power = rejections / n_reps
        assert power >= min_power, (
            f"Power {power:.4f} is below minimum {min_power} "
            f"(delta={delta}, sigma={sigma}, n={n_obs}, "
            f"{n_reps} reps, {rejections} rejections)"
        )
