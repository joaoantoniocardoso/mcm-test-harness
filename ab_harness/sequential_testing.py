"""Sequential (adaptive) stopping for A/B tests.

Implements two production-grade sequential testing methods:

1. **Group Sequential Testing (GST)** via ``spotify-confidence``
   - Default for overnight / batch runs (highest power when max N known)
   - Pre-planned interim analyses with Lan-DeMets alpha-spending
   - Exact numerical boundary computation (not approximate)
   - Spotify's internal A/B platform library, Apache-2.0

2. **Mixture Sequential Probability Ratio Test (mSPRT)** via ``msprt``
   - Default for quick interactive ``iterate`` runs
   - Always-valid: checks after every completed A/B pair with zero
     statistical penalty, no look schedule needed
   - Based on Johari, Pekelis & Walsh (2017/2022)
   - In-repo implementation under ``libs/msprt/``

Both methods are available via :func:`replay_all_methods` for
retrospective comparison.  GST is the default for overnight runs
via :func:`check_interim` (checks at pre-planned look points).
mSPRT is the default for interactive runs via
:func:`check_interim_msprt` (checks every trial).

References
----------
- Spotify's method comparison (100k MC sims):
  https://engineering.atspotify.com/2023/03/choosing-sequential-testing-framework-comparisons-and-discussions
- GST is systematically more powerful than mSPRT/GAVI when max
  sample size is known (our case: ``--repetitions``).
- Johari R., Pekelis L. & Walsh D. (2022),
  "Always Valid Inference: Continuous Monitoring of A/B Tests",
  Operations Research 70(3):1806-1821.
"""

from __future__ import annotations

__all__ = [
    "AdaptiveState",
    "InterimResult",
    "MsprtAdaptiveState",
    "ReplayResult",
    "_robust_pooled_sigma",
    "check_interim",
    "check_interim_msprt",
    "compute_futility_boundary",
    "compute_gst_boundaries",
    "create_adaptive_state",
    "create_msprt_state",
    "format_replay_summary",
    "replay_all_methods",
    "serialize_replay_results",
]

import json
import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats as sp_stats

from ab_harness.compare import cohens_d

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Group Sequential Testing -- via spotify-confidence
# ---------------------------------------------------------------------------


def compute_gst_boundaries(
    max_reps: int,
    look_every: int,
    alpha: float = 0.05,
    rho: float = 3.0,
) -> dict[int, float]:
    """Compute GST efficacy boundaries using spotify-confidence.

    Uses the Lan-DeMets alpha-spending approach implemented in
    ``spotify_confidence.analysis.frequentist.sequential_bound_solver``.

    Parameters
    ----------
    max_reps : int
        Maximum number of reps per side.
    look_every : int
        Perform an interim analysis every *look_every* reps per side.
    alpha : float
        Overall Type I error rate (default 0.05).
    rho : float
        Alpha-spending function exponent (power family).
        ``rho=3`` gives cubic spending (closest to O'Brien-Fleming).
        ``rho=2`` gives quadratic spending.

    Returns
    -------
    dict[int, float]
        Mapping ``{reps_per_side: p_value_threshold}`` for each look.
        At look *k*, reject H0 if ``p_value < boundaries[reps_per_side]``.
    """
    from spotify_confidence.analysis.frequentist.sequential_bound_solver import (
        EMPTY_STATE,
    )
    from spotify_confidence.analysis.frequentist.sequential_bound_solver import (
        bounds as spotify_bounds,
    )

    look_points = list(range(look_every, max_reps + 1, look_every))
    # Ensure the final look is included
    if not look_points or look_points[-1] != max_reps:
        look_points.append(max_reps)

    # Information fractions (must be monotonically increasing in (0, 1])
    t = np.array([n / max_reps for n in look_points], dtype=float)

    # Compute z-score boundaries via spotify-confidence
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # suppress pandas FutureWarning from lib
        result = spotify_bounds(
            t,
            alpha=alpha,
            rho=rho,
            ztrun=8.0,
            sides=2,
            state=EMPTY_STATE,
        )
    z_bounds = result.bounds

    # Convert z-score boundaries to two-sided p-value thresholds
    boundaries: dict[int, float] = {}
    for i, n in enumerate(look_points):
        p_threshold = float(2.0 * (1.0 - sp_stats.norm.cdf(abs(z_bounds[i]))))
        boundaries[n] = p_threshold
        log.debug(
            "GST boundary (spotify): look %d/%d at n=%d (t=%.2f): z=%.4f -> p_threshold=%.6f",
            i + 1,
            len(look_points),
            n,
            t[i],
            z_bounds[i],
            p_threshold,
        )

    return boundaries


def compute_futility_boundary(
    n_per_side: int,
    max_reps: int,
    mde_frac: float,
    alpha: float = 0.05,
    power: float = 0.80,
) -> float:
    """Heuristic futility stopping rule (not a formal beta-spending boundary).

    Returns a p-value threshold above which we recommend stopping for
    futility (stop if ``p > threshold``).

    Note
    ----
    This is a **pragmatic guard**, not a validated statistical boundary
    with known operating characteristics.  Unlike the efficacy boundaries
    from spotify-confidence (which provide exact Type I error control),
    this uses a simple linear interpolation that is deliberately
    conservative: it only activates after 30% of the budget is spent
    and requires increasingly strong evidence of futility as the
    experiment progresses.  It has no formal power guarantee.

    The rule: at information fraction t >= 0.3, the threshold linearly
    decreases from 0.8 to 0.5 at t = 1.0.  Before t = 0.3 the
    threshold is 1.0 (never triggers).
    """
    # Information fraction
    t = n_per_side / max_reps
    if t < 0.3:
        # Too early for futility assessment
        return 1.0  # never trigger
    # Linear interpolation: at t=0.3 threshold=0.8, at t=1.0 threshold=0.5
    threshold = 0.8 - (t - 0.3) * (0.3 / 0.7)
    return max(0.5, threshold)


@dataclass
class InterimResult:
    """Result of a single interim analysis."""

    look_number: int
    reps_per_side: int
    information_fraction: float
    kpi_name: str
    p_value: float
    efficacy_boundary: float
    futility_threshold: float
    stopped_for_efficacy: bool
    stopped_for_futility: bool
    delta_pct: float | None
    cohens_d: float | None

    @property
    def stopped(self) -> bool:
        return self.stopped_for_efficacy or self.stopped_for_futility

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict for JSON persistence."""
        return {
            "look_number": self.look_number,
            "reps_per_side": self.reps_per_side,
            "information_fraction": self.information_fraction,
            "kpi_name": self.kpi_name,
            "p_value": self.p_value,
            "efficacy_boundary": self.efficacy_boundary,
            "futility_threshold": self.futility_threshold,
            "stopped_for_efficacy": self.stopped_for_efficacy,
            "stopped_for_futility": self.stopped_for_futility,
            "delta_pct": self.delta_pct,
            "cohens_d": self.cohens_d,
        }


@dataclass
class AdaptiveState:
    """Tracks state across interim analyses during a live experiment."""

    max_reps: int
    look_every: int
    alpha: float
    target_kpis: list[str]
    boundaries: dict[int, float]
    futility_enabled: bool
    interim_results: list[InterimResult] = field(default_factory=list)
    stopped: bool = False
    stop_reason: str = ""
    stop_kpi: str = ""
    stop_at_rep: int = 0

    def save(self, path: Path) -> None:
        """Persist adaptive state to JSON for audit trail."""
        data = {
            "max_reps": self.max_reps,
            "look_every": self.look_every,
            "alpha": self.alpha,
            "target_kpis": self.target_kpis,
            "boundaries": {str(k): v for k, v in self.boundaries.items()},
            "futility_enabled": self.futility_enabled,
            "stopped": self.stopped,
            "stop_reason": self.stop_reason,
            "stop_kpi": self.stop_kpi,
            "stop_at_rep": self.stop_at_rep,
            "interim_results": [r.to_dict() for r in self.interim_results],
        }
        path.write_text(json.dumps(data, indent=2))


def create_adaptive_state(
    *,
    max_reps: int,
    look_every: int = 0,
    alpha: float = 0.05,
    target_kpis: list[str] | None = None,
    futility: bool = True,
) -> AdaptiveState:
    """Create the adaptive state for a new experiment.

    Parameters
    ----------
    max_reps : int
        Maximum number of reps per side (the ``--repetitions`` value).
    look_every : int
        Check every N completed reps per side. If 0, auto-selects a
        sensible default based on max_reps.
    alpha : float
        Overall significance level.
    target_kpis : list[str] | None
        KPI names to monitor for stopping. If None, uses the primary
        KPI ``system_cpu_pct``.
    futility : bool
        Enable futility stopping (stop early when effect is clearly null).
    """
    if look_every <= 0:
        # Auto-select: aim for ~6-10 looks
        look_every = max(5, max_reps // 8)

    if target_kpis is None:
        target_kpis = ["system_cpu_pct"]

    boundaries = compute_gst_boundaries(max_reps, look_every, alpha)

    log.info(
        "Adaptive testing (spotify-confidence GST): max_reps=%d, look_every=%d, "
        "%d planned looks, alpha=%.3f",
        max_reps,
        look_every,
        len(boundaries),
        alpha,
    )
    for n, a in sorted(boundaries.items()):
        log.info("  Look at n=%d: p_threshold=%.6f", n, a)

    return AdaptiveState(
        max_reps=max_reps,
        look_every=look_every,
        alpha=alpha,
        target_kpis=target_kpis,
        boundaries=boundaries,
        futility_enabled=futility,
    )


def check_interim(
    state: AdaptiveState,
    a_per_rep: dict[str, list[float]],
    b_per_rep: dict[str, list[float]],
    reps_per_side: int,
) -> InterimResult | None:
    """Run an interim analysis if we're at a look point.

    Parameters
    ----------
    state : AdaptiveState
        Current adaptive state.
    a_per_rep : dict[str, list[float]]
        Per-rep means for iteration (A), keyed by KPI name.
    b_per_rep : dict[str, list[float]]
        Per-rep means for baseline (B), keyed by KPI name.
    reps_per_side : int
        Number of completed reps per side so far.

    Returns
    -------
    InterimResult | None
        The interim result if this was a look point, else None.
        Check ``result.stopped`` to decide whether to halt collection.
    """
    if reps_per_side not in state.boundaries:
        return None

    if state.stopped:
        return None  # already decided to stop

    boundary = state.boundaries[reps_per_side]
    info_frac = reps_per_side / state.max_reps
    look_num = sorted(state.boundaries.keys()).index(reps_per_side) + 1

    log.info(
        "=== INTERIM ANALYSIS %d: n=%d per side (%.0f%% of max) ===",
        look_num,
        reps_per_side,
        info_frac * 100,
    )

    for kpi_name in state.target_kpis:
        a_vals = a_per_rep.get(kpi_name, [])
        b_vals = b_per_rep.get(kpi_name, [])

        if len(a_vals) < 2 or len(b_vals) < 2:
            log.info(
                "  %s: insufficient data (A=%d, B=%d), skipping",
                kpi_name,
                len(a_vals),
                len(b_vals),
            )
            continue

        # Welch's t-test (scipy -- validated)
        _t_stat, p_val = sp_stats.ttest_ind(a_vals, b_vals, equal_var=False)
        p_value = float(p_val)

        # Effect size
        mean_a, mean_b = np.mean(a_vals), np.mean(b_vals)
        delta_pct = ((mean_a - mean_b) / abs(mean_b) * 100) if mean_b != 0 else None
        effect_d = cohens_d(a_vals, b_vals)

        # Futility check
        futility_thresh = 1.0
        stopped_futility = False
        if state.futility_enabled:
            futility_thresh = compute_futility_boundary(
                reps_per_side,
                state.max_reps,
                mde_frac=0.02,
            )
            stopped_futility = p_value > futility_thresh

        # Efficacy check (p-value vs spotify-confidence boundary)
        stopped_efficacy = p_value < boundary

        result = InterimResult(
            look_number=look_num,
            reps_per_side=reps_per_side,
            information_fraction=info_frac,
            kpi_name=kpi_name,
            p_value=p_value,
            efficacy_boundary=boundary,
            futility_threshold=futility_thresh,
            stopped_for_efficacy=stopped_efficacy,
            stopped_for_futility=stopped_futility,
            delta_pct=float(delta_pct) if delta_pct is not None else None,
            cohens_d=effect_d,
        )
        state.interim_results.append(result)

        log.info(
            "  %s: p=%.4f (boundary=%.6f), delta=%.2f%%, d=%.3f",
            kpi_name,
            p_value,
            boundary,
            delta_pct if delta_pct is not None else 0,
            effect_d if effect_d is not None else 0,
        )

        if stopped_efficacy:
            log.info(
                "  >>> EARLY STOP (efficacy): %s is significant at look %d (p=%.4f < %.6f)",
                kpi_name,
                look_num,
                p_value,
                boundary,
            )
            state.stopped = True
            state.stop_reason = "efficacy"
            state.stop_kpi = kpi_name
            state.stop_at_rep = reps_per_side
            return result

        if stopped_futility:
            log.info(
                "  >>> EARLY STOP (futility): %s shows no trend at look %d "
                "(p=%.4f > %.4f futility threshold)",
                kpi_name,
                look_num,
                p_value,
                futility_thresh,
            )
            state.stopped = True
            state.stop_reason = "futility"
            state.stop_kpi = kpi_name
            state.stop_at_rep = reps_per_side
            return result

    log.info("  No stopping criterion met. Continuing collection...")
    return None


# ---------------------------------------------------------------------------
# 2. Mixture Sequential Probability Ratio Test (mSPRT) -- via libs/msprt
# ---------------------------------------------------------------------------


@dataclass
class MsprtAdaptiveState:
    """Tracks state for live adaptive stopping using mSPRT.

    Unlike :class:`AdaptiveState` (GST), this checks **every trial**
    with zero statistical penalty, thanks to the always-valid mSPRT.
    """

    max_reps: int
    alpha: float
    target_kpis: list[str]
    min_n: int
    sigma: float
    futility_enabled: bool
    # Per-KPI accumulated observation lists
    _a_vals: dict[str, list[float]] = field(default_factory=dict, repr=False)
    _b_vals: dict[str, list[float]] = field(default_factory=dict, repr=False)
    _pairs_fed: int = 0
    interim_results: list[InterimResult] = field(default_factory=list)
    stopped: bool = False
    stop_reason: str = ""
    stop_kpi: str = ""
    stop_at_rep: int = 0

    def save(self, path: Path) -> None:
        """Persist adaptive state to JSON for audit trail."""
        data = {
            "method": "msprt",
            "max_reps": self.max_reps,
            "alpha": self.alpha,
            "target_kpis": self.target_kpis,
            "min_n": self.min_n,
            "sigma": self.sigma,
            "futility_enabled": self.futility_enabled,
            "pairs_fed": self._pairs_fed,
            "stopped": self.stopped,
            "stop_reason": self.stop_reason,
            "stop_kpi": self.stop_kpi,
            "stop_at_rep": self.stop_at_rep,
            "interim_results": [r.to_dict() for r in self.interim_results],
        }
        path.write_text(json.dumps(data, indent=2))


def _robust_pooled_sigma(x: np.ndarray, y: np.ndarray) -> float | None:
    """Estimate pooled sigma using the Median Absolute Deviation (MAD).

    More robust than sample variance for heavy-tailed distributions
    (e.g., max_freeze_ms).  Falls back to the classical pooled standard
    deviation when MAD returns zero (constant data).

    The conversion factor 1.4826 makes the MAD consistent with the
    standard deviation for normal distributions.
    """
    mad_x = float(np.median(np.abs(x - np.median(x)))) * 1.4826
    mad_y = float(np.median(np.abs(y - np.median(y)))) * 1.4826

    if mad_x > 0 or mad_y > 0:
        return float(np.sqrt((mad_x**2 + mad_y**2) / 2.0))

    # Fallback: classical pooled standard deviation
    pooled_std = float(np.sqrt((np.var(x, ddof=1) + np.var(y, ddof=1)) / 2.0))
    return pooled_std if pooled_std > 0 else None


def create_msprt_state(
    *,
    max_reps: int,
    alpha: float = 0.05,
    target_kpis: list[str] | None = None,
    min_n: int = 8,
    sigma: float = 1.0,
    futility: bool = True,
) -> MsprtAdaptiveState:
    """Create adaptive state using mSPRT (always-valid sequential test).

    Parameters
    ----------
    max_reps : int
        Maximum number of reps per side (the ``--repetitions`` value).
    alpha : float
        Overall significance level.
    target_kpis : list[str] | None
        KPI names to monitor for stopping. If None, uses ``system_cpu_pct``.
    min_n : int
        Minimum completed pairs before checking begins (default 8).
        Increased from 5 to improve early variance-estimate stability,
        especially for heavy-tailed KPIs like ``max_freeze_ms``.
    sigma : float
        Estimated population standard deviation (default 1.0).
        Used for the mSPRT mixing variance computation.
    futility : bool
        Enable futility stopping.
    """
    if target_kpis is None:
        target_kpis = ["system_cpu_pct"]

    a_vals: dict[str, list[float]] = {kpi: [] for kpi in target_kpis}
    b_vals: dict[str, list[float]] = {kpi: [] for kpi in target_kpis}

    log.info(
        "Adaptive testing (mSPRT): max_reps=%d, "
        "min_n=%d, sigma=%.3f, checks every trial, alpha=%.3f, kpis=%s",
        max_reps,
        min_n,
        sigma,
        alpha,
        target_kpis,
    )

    state = MsprtAdaptiveState(
        max_reps=max_reps,
        alpha=alpha,
        target_kpis=target_kpis,
        min_n=min_n,
        sigma=sigma,
        futility_enabled=futility,
    )
    state._a_vals = a_vals
    state._b_vals = b_vals
    return state


def check_interim_msprt(
    state: MsprtAdaptiveState,
    kpi_means_a: dict[str, float],
    kpi_means_b: dict[str, float],
    reps_per_side: int,
) -> InterimResult | None:
    """Feed one new A/B pair into the mSPRT and check for stopping.

    Unlike :func:`check_interim` (GST), this is called after **every**
    completed pair.  The mSPRT statistic is recomputed on accumulated data.

    Parameters
    ----------
    state : MsprtAdaptiveState
        Current adaptive state (mutated in place).
    kpi_means_a : dict[str, float]
        KPI means for the newly completed iteration (A) rep.
    kpi_means_b : dict[str, float]
        KPI means for the newly completed baseline (B) rep.
    reps_per_side : int
        Number of completed reps per side (including this one).

    Returns
    -------
    InterimResult | None
        The interim result if a stopping criterion was met, else None.
    """
    from msprt import compute_tau, msprt_statistic

    if state.stopped:
        return None

    info_frac = reps_per_side / state.max_reps

    # Accumulate new observations
    for kpi_name in state.target_kpis:
        a_val = kpi_means_a.get(kpi_name)
        b_val = kpi_means_b.get(kpi_name)
        if a_val is not None:
            state._a_vals.setdefault(kpi_name, []).append(float(a_val))
        if b_val is not None:
            state._b_vals.setdefault(kpi_name, []).append(float(b_val))

    state._pairs_fed = reps_per_side

    # Don't check before min_n pairs
    if reps_per_side < state.min_n:
        return None

    look_num = reps_per_side  # every pair is a look

    log.info(
        "=== INTERIM ANALYSIS (mSPRT) n=%d per side (%.0f%% of max) ===",
        reps_per_side,
        info_frac * 100,
    )

    threshold = 1.0 / state.alpha

    for kpi_name in state.target_kpis:
        a_list = state._a_vals.get(kpi_name, [])
        b_list = state._b_vals.get(kpi_name, [])

        n_pairs = min(len(a_list), len(b_list))
        if n_pairs < 2:
            log.info("  %s: insufficient data (n=%d), skipping", kpi_name, n_pairs)
            continue

        x = np.array(a_list[:n_pairs])
        y = np.array(b_list[:n_pairs])

        # Estimate sigma from pooled data using robust MAD estimator
        sigma = state.sigma
        if n_pairs >= state.min_n:
            robust_sigma = _robust_pooled_sigma(x, y)
            if robust_sigma is not None and robust_sigma > 0:
                sigma = robust_sigma

        tau = compute_tau(alpha=state.alpha, sigma=sigma, truncation=state.max_reps)
        trajectory = msprt_statistic(x, y, sigma=sigma, tau=tau)
        lambda_n = float(trajectory[-1])

        # Efficacy: Lambda exceeds 1/alpha
        stopped_efficacy = lambda_n > threshold

        # Effect size estimates
        mean_a, mean_b = float(np.mean(x)), float(np.mean(y))
        delta_pct = ((mean_a - mean_b) / abs(mean_b) * 100) if mean_b != 0 else None
        effect_d = cohens_d(x.tolist(), y.tolist())

        # Approximate p-value from the likelihood ratio (for display/futility)
        # Under H0, -2*log(Lambda) ~ chi-sq(1) asymptotically, but for
        # the mSPRT we use 1/Lambda as a conservative p-value bound.
        p_value = min(1.0, 1.0 / lambda_n) if lambda_n > 0 else 1.0

        # Futility check
        futility_thresh = 1.0
        stopped_futility = False
        if state.futility_enabled:
            futility_thresh = compute_futility_boundary(
                reps_per_side,
                state.max_reps,
                mde_frac=0.02,
            )
            stopped_futility = p_value > futility_thresh

        result = InterimResult(
            look_number=look_num,
            reps_per_side=reps_per_side,
            information_fraction=info_frac,
            kpi_name=kpi_name,
            p_value=p_value,
            efficacy_boundary=threshold,
            futility_threshold=futility_thresh,
            stopped_for_efficacy=stopped_efficacy,
            stopped_for_futility=stopped_futility,
            delta_pct=delta_pct,
            cohens_d=effect_d,
        )
        state.interim_results.append(result)

        log.info(
            "  %s: Lambda=%.4f (threshold=%.1f), p~%.4f, delta=%.2f%%, d=%.3f",
            kpi_name,
            lambda_n,
            threshold,
            p_value,
            delta_pct if delta_pct is not None else 0,
            effect_d if effect_d is not None else 0,
        )

        if stopped_efficacy:
            log.info(
                "  >>> EARLY STOP (efficacy): %s Lambda=%.4f > %.1f at n=%d",
                kpi_name,
                lambda_n,
                threshold,
                reps_per_side,
            )
            state.stopped = True
            state.stop_reason = "efficacy"
            state.stop_kpi = kpi_name
            state.stop_at_rep = reps_per_side
            return result

        if stopped_futility:
            log.info(
                "  >>> EARLY STOP (futility): %s shows no trend at n=%d "
                "(p~%.4f > %.4f futility threshold)",
                kpi_name,
                reps_per_side,
                p_value,
                futility_thresh,
            )
            state.stopped = True
            state.stop_reason = "futility"
            state.stop_kpi = kpi_name
            state.stop_at_rep = reps_per_side
            return result

    log.info("  No stopping criterion met. Continuing collection...")
    return None


def _replay_msprt(
    a_per_rep: list[float],
    b_per_rep: list[float],
    alpha: float = 0.05,
    min_n: int = 10,
) -> dict[str, Any]:
    """Replay data through the mSPRT.

    Parameters
    ----------
    a_per_rep, b_per_rep : list[float]
        Ordered per-rep means for iteration (A) and baseline (B).
    alpha : float
        Significance level.
    min_n : int
        Minimum observation pairs before testing begins (default 10).
        Uses robust MAD-based sigma estimation (see :func:`_robust_pooled_sigma`).

    Returns
    -------
    dict with method info, stopped flag, stop_at, and trajectory.
    """
    from msprt import msprt_test

    max_n = min(len(a_per_rep), len(b_per_rep))
    x = np.array(a_per_rep[:max_n])
    y = np.array(b_per_rep[:max_n])

    # Estimate sigma from pooled data using robust MAD estimator
    robust_sigma = _robust_pooled_sigma(x, y)
    sigma = robust_sigma if robust_sigma is not None and robust_sigma > 0 else 1.0

    result = msprt_test(x, y, sigma=sigma, alpha=alpha, min_n=min_n)

    # Build trajectory dicts for each pair from min_n onward
    trajectory: list[dict[str, Any]] = []
    threshold = 1.0 / alpha
    for i in range(max(0, min_n - 1), max_n):
        trajectory.append(
            {
                "n": i + 1,
                "lambda": float(result.trajectory[i]),
                "rejected": bool(result.trajectory[i] > threshold),
            }
        )

    return {
        "method": "msprt",
        "stopped": result.rejected,
        "stop_at": result.n_rejection,
        "min_n": min_n,
        "max_n_checked": max_n,
        "trajectory": trajectory,
    }


# ---------------------------------------------------------------------------
# 3. Unified replay: run all methods on collected data
# ---------------------------------------------------------------------------


@dataclass
class ReplayResult:
    """Result of replaying collected data through sequential methods."""

    kpi_name: str
    total_reps: int
    gst: dict[str, Any]
    msprt: dict[str, Any]


def replay_all_methods(
    a_per_rep: list[float],
    b_per_rep: list[float],
    kpi_name: str,
    max_reps: int | None = None,
    look_every: int | None = None,
    alpha: float = 0.05,
) -> ReplayResult:
    """Replay collected data through both sequential testing methods.

    This is the key post-hoc analysis tool: given already-collected
    per-rep data, simulate when each method *would have* stopped.

    Methods:
    1. GST with Lan-DeMets alpha-spending (spotify-confidence)
    2. Mixture Sequential Probability Ratio Test (mSPRT)

    Parameters
    ----------
    a_per_rep, b_per_rep : list[float]
        Per-rep means for iteration (A) and baseline (B), in collection order.
    kpi_name : str
        Name of the KPI being analyzed.
    max_reps : int | None
        Maximum reps (for GST boundary computation). Defaults to len of data.
    look_every : int | None
        GST look interval. Auto-selected if None.
    alpha : float
        Significance level.

    Returns
    -------
    ReplayResult with per-method stopping info and trajectories.
    """
    n_total = min(len(a_per_rep), len(b_per_rep))
    if max_reps is None:
        max_reps = n_total
    if look_every is None:
        look_every = max(5, max_reps // 8)

    # --- GST replay (spotify-confidence) ---
    boundaries = compute_gst_boundaries(max_reps, look_every, alpha)
    gst_result: dict[str, Any] = {
        "method": "GST_spotify",
        "stopped": False,
        "stop_at": None,
        "boundaries": {str(k): v for k, v in boundaries.items()},
        "trajectory": [],
    }

    for n_look, boundary in sorted(boundaries.items()):
        if n_look > n_total:
            break
        a_sub = a_per_rep[:n_look]
        b_sub = b_per_rep[:n_look]
        if len(a_sub) < 2 or len(b_sub) < 2:
            continue
        _, p_val = sp_stats.ttest_ind(a_sub, b_sub, equal_var=False)
        p_value = float(p_val)

        gst_result["trajectory"].append(
            {
                "n": n_look,
                "p_value": p_value,
                "boundary": boundary,
                "rejected": p_value < boundary,
            }
        )

        if p_value < boundary and not gst_result["stopped"]:
            gst_result["stopped"] = True
            gst_result["stop_at"] = n_look

    # --- mSPRT replay ---
    msprt_result = _replay_msprt(a_per_rep, b_per_rep, alpha=alpha)

    return ReplayResult(
        kpi_name=kpi_name,
        total_reps=n_total,
        gst=gst_result,
        msprt=msprt_result,
    )


def format_replay_summary(result: ReplayResult) -> str:
    """Human-readable summary of a replay result."""
    lines = [
        f"\n{'=' * 70}",
        f"Sequential Testing Replay: {result.kpi_name} ({result.total_reps} reps/side)",
        f"{'=' * 70}",
        "",
        f"{'Method':<30} {'Stopped?':<10} {'At rep':<10} {'Savings':<10}",
        f"{'-' * 60}",
    ]

    for name, method_result in [
        ("GST (spotify-confidence)", result.gst),
        ("mSPRT", result.msprt),
    ]:
        stopped = method_result.get("stopped", False)
        stop_at = method_result.get("stop_at")
        if stopped and stop_at:
            savings = f"{(1 - stop_at / result.total_reps) * 100:.0f}%"
            lines.append(f"{name:<30} {'Yes':<10} {stop_at:<10} {savings:<10}")
        else:
            lines.append(f"{name:<30} {'No':<10} {'-':<10} {'0%':<10}")

    lines.append("")
    return "\n".join(lines)


def serialize_replay_results(results: list[ReplayResult]) -> list[dict[str, Any]]:
    """Serialize a list of ReplayResults to dicts suitable for JSON output.

    Strips trajectory data (which can be large) from each method's result.
    """
    return [
        {
            "kpi_name": r.kpi_name,
            "total_reps": r.total_reps,
            "gst": {k: v for k, v in r.gst.items() if k != "trajectory"},
            "msprt": {k: v for k, v in r.msprt.items() if k != "trajectory"},
        }
        for r in results
    ]
