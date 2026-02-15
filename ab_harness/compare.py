"""Compare two runs: compute KPI deltas, Welch's t-test, pass/fail verdicts.

Uses ``spotify-confidence.StudentsTTest`` for validated p-values,
confidence intervals, and power estimates.
"""

from __future__ import annotations

__all__ = [
    "KpiComparison",
    "RunComparison",
    "TopologyDiff",
    "_bootstrap_ci",
    "_cliffs_delta_label",
    "_count_iqr_outliers",
    "_durbin_watson",
    "_holm_bonferroni",
    "_mann_whitney_u",
    "_shapiro_wilk",
    "cliffs_delta",
    "cohens_d",
    "compare_runs",
    "compare_topology",
    "recommend_sample_size",
    "save_comparison",
]

import json
import logging
import math
import warnings
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from scipy import stats as scipy_stats

from ab_harness.kpi import KPI_REGISTRY, load_aggregate

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------


def _build_spotify_dataframe(
    a_per_rep: list[float],
    b_per_rep: list[float],
) -> pd.DataFrame:
    """Convert per-rep means into the sufficient-statistics format
    that ``spotify-confidence.StudentsTTest`` expects.

    Each row is one group (baseline / iteration) with:
    - ``value_sum``: sum of per-rep means
    - ``value_sumsq``: sum of *squares* of per-rep means
    - ``count``: number of reps
    """
    return pd.DataFrame(
        {
            "group": ["baseline", "iteration"],
            "value_sum": [sum(b_per_rep), sum(a_per_rep)],
            "value_sumsq": [
                sum(x**2 for x in b_per_rep),
                sum(x**2 for x in a_per_rep),
            ],
            "count": [len(b_per_rep), len(a_per_rep)],
        }
    )


def _spotify_ttest(
    a_per_rep: list[float],
    b_per_rep: list[float],
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Run Welch's t-test via spotify-confidence.

    Returns a dict with ``p_value``, ``ci_lower``, ``ci_upper``,
    ``difference``, ``power``, and ``is_significant``.
    Falls back to a minimal scipy-only result if the library call fails.
    """
    from spotify_confidence import StudentsTTest

    try:
        df = _build_spotify_dataframe(a_per_rep, b_per_rep)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            test = StudentsTTest(
                df,
                numerator_column="value_sum",
                numerator_sum_squares_column="value_sumsq",
                denominator_column="count",
                categorical_group_columns="group",
            )
            result = test.difference(
                level_1="baseline",
                level_2="iteration",
                verbose=True,
            )

        row = result.iloc[0]
        return {
            "p_value": float(row["p-value"]),
            "ci_lower": float(row["ci_lower"]),
            "ci_upper": float(row["ci_upper"]),
            "difference": float(row["difference"]),
            "is_significant": bool(row["is_significant"]),
            "power": float(row["power"]) if pd.notna(row.get("power")) else None,
        }
    except Exception as exc:
        log.warning("spotify-confidence StudentsTTest failed, falling back to scipy: %s", exc)
        return _scipy_ttest_fallback(a_per_rep, b_per_rep, alpha)


def _scipy_ttest_fallback(
    a_per_rep: list[float],
    b_per_rep: list[float],
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Minimal fallback using scipy if spotify-confidence fails."""
    import numpy as np

    _, p_val = scipy_stats.ttest_ind(a_per_rep, b_per_rep, equal_var=False)
    p_value = float(p_val)

    mean_a, mean_b = np.mean(a_per_rep), np.mean(b_per_rep)
    diff = float(mean_a - mean_b)

    na, nb = len(a_per_rep), len(b_per_rep)
    var_a = np.var(a_per_rep, ddof=1)
    var_b = np.var(b_per_rep, ddof=1)
    se = float(np.sqrt(var_a / na + var_b / nb))

    # Welch-Satterthwaite df
    num = (var_a / na + var_b / nb) ** 2
    denom = (var_a / na) ** 2 / (na - 1) + (var_b / nb) ** 2 / (nb - 1)
    df = float(num / denom) if denom > 0 else 1.0
    t_crit = float(scipy_stats.t.ppf(1.0 - alpha / 2, df))

    return {
        "p_value": p_value,
        "ci_lower": diff - t_crit * se,
        "ci_upper": diff + t_crit * se,
        "difference": diff,
        "is_significant": p_value < alpha,
        "power": None,
    }


def cohens_d(a: list[float], b: list[float]) -> float | None:
    """Compute Cohen's d (pooled standard deviation)."""
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return None
    import numpy as np

    arr_a, arr_b = np.array(a), np.array(b)
    var_a = float(np.var(arr_a, ddof=1))
    var_b = float(np.var(arr_b, ddof=1))
    pooled_var = ((na - 1) * var_a + (nb - 1) * var_b) / (na + nb - 2)
    if pooled_var <= 0:
        return None
    return float((np.mean(arr_a) - np.mean(arr_b)) / math.sqrt(pooled_var))


def _effect_size_label(d: float | None) -> str:
    """Classify Cohen's d magnitude."""
    if d is None:
        return "N/A"
    ad = abs(d)
    if ad < 0.2:
        return "negligible"
    if ad < 0.5:
        return "small"
    if ad < 0.8:
        return "medium"
    return "large"


def _holm_bonferroni(
    p_values: Sequence[float | None],
    alpha: float = 0.05,
) -> list[float | None]:
    """Apply Holm-Bonferroni step-down correction to a list of p-values.

    Returns adjusted p-values (each >= the original).  ``None`` entries
    are passed through unchanged.
    """
    n = len(p_values)
    # Build list of (original_index, p_value) for non-None entries
    indexed = [(i, p) for i, p in enumerate(p_values) if p is not None]
    if not indexed:
        return list(p_values)

    # Sort by p-value ascending
    indexed.sort(key=lambda t: t[1])

    adjusted: dict[int, float] = {}
    max_adj = 0.0
    for rank, (orig_idx, p) in enumerate(indexed):
        k = len(indexed) - rank  # remaining hypotheses
        adj_p = min(p * k, 1.0)
        # Enforce monotonicity: adjusted p must be >= previous adjusted p
        adj_p = max(adj_p, max_adj)
        max_adj = adj_p
        adjusted[orig_idx] = adj_p

    result: list[float | None] = []
    for i in range(n):
        if p_values[i] is None:
            result.append(None)
        else:
            result.append(adjusted[i])
    return result


def _durbin_watson(values: list[float]) -> float | None:
    """Compute the Durbin-Watson statistic for serial autocorrelation.

    Operates on residuals from the mean (i.e., demeaned values).
    Returns a value in [0, 4].  DW ~ 2.0 indicates no autocorrelation;
    DW < 1.5 or DW > 2.5 suggests moderate serial correlation.
    Returns ``None`` if fewer than 3 observations or zero variance.
    """
    n = len(values)
    if n < 3:
        return None
    mean = sum(values) / n
    residuals = [v - mean for v in values]
    residuals_sq = sum(r**2 for r in residuals)
    if residuals_sq == 0:
        return None
    diffs_sq = sum((residuals[i] - residuals[i - 1]) ** 2 for i in range(1, n))
    return diffs_sq / residuals_sq


def cliffs_delta(a: list[float], b: list[float]) -> float | None:
    """Compute Cliff's delta: P(a > b) - P(a < b).

    A non-parametric effect size in [-1, +1] that does not assume
    normality.  More robust than Cohen's d for heavy-tailed or
    skewed distributions.

    Classification (Romano et al. 2006):
      |d| < 0.147  -> negligible
      |d| < 0.33   -> small
      |d| < 0.474  -> medium
      |d| >= 0.474 -> large
    """
    if not a or not b:
        return None
    count = sum((ai > bj) - (ai < bj) for ai in a for bj in b)
    return count / (len(a) * len(b))


def _cliffs_delta_label(d: float | None) -> str:
    """Classify Cliff's delta magnitude (Romano et al. 2006)."""
    if d is None:
        return "N/A"
    ad = abs(d)
    if ad < 0.147:
        return "negligible"
    if ad < 0.33:
        return "small"
    if ad < 0.474:
        return "medium"
    return "large"


def _shapiro_wilk(values: list[float]) -> float | None:
    """Return the Shapiro-Wilk p-value, or None if not applicable.

    Requires 4 <= n <= 50 observations.  Returns None for constant
    data or insufficient samples.
    """
    n = len(values)
    if n < 4 or n > 50:
        return None
    if max(values) - min(values) < 1e-12:
        return None  # constant data
    try:
        _stat, p = scipy_stats.shapiro(values)
        return float(p)
    except Exception:
        return None


def _mann_whitney_u(
    a: list[float],
    b: list[float],
) -> float | None:
    """Two-sided Mann-Whitney U test p-value.

    Distribution-free alternative to Welch's t-test.  Tests whether
    one group tends to produce larger values than the other.
    Requires at least 3 observations per side.
    """
    if len(a) < 3 or len(b) < 3:
        return None
    try:
        _stat, p = scipy_stats.mannwhitneyu(a, b, alternative="two-sided")
        return float(p)
    except Exception:
        return None


def _bootstrap_ci(
    a: list[float],
    b: list[float],
    alpha: float = 0.05,
) -> tuple[float, float] | None:
    """BCa bootstrap CI for the difference in means (a - b).

    Falls back to percentile method if BCa fails.
    Requires scipy >= 1.9 for scipy.stats.bootstrap.
    """
    import numpy as np

    if len(a) < 3 or len(b) < 3:
        return None
    try:
        arr_a = np.array(a)
        arr_b = np.array(b)
        result = scipy_stats.bootstrap(
            (arr_a, arr_b),
            statistic=lambda x, y, axis: np.mean(x, axis=axis) - np.mean(y, axis=axis),
            n_resamples=9999,
            confidence_level=1 - alpha,
            method="BCa",
            random_state=np.random.default_rng(42),
        )
        return (float(result.confidence_interval.low), float(result.confidence_interval.high))
    except Exception:
        try:
            result = scipy_stats.bootstrap(
                (arr_a, arr_b),
                statistic=lambda x, y, axis: np.mean(x, axis=axis) - np.mean(y, axis=axis),
                n_resamples=9999,
                confidence_level=1 - alpha,
                method="percentile",
                random_state=np.random.default_rng(42),
            )
            return (float(result.confidence_interval.low), float(result.confidence_interval.high))
        except Exception:
            return None


def _count_iqr_outliers(values: list[float]) -> int:
    """Count observations outside 1.5 * IQR (Tukey fences).

    Returns 0 for fewer than 4 observations.
    """
    if len(values) < 4:
        return 0
    import numpy as np

    arr = np.array(values)
    q1, q3 = float(np.percentile(arr, 25)), float(np.percentile(arr, 75))
    iqr = q3 - q1
    if iqr < 1e-12:
        return 0
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return int(np.sum((arr < lower) | (arr > upper)))


def recommend_sample_size(
    std: float,
    mde_pct: float,
    baseline_mean: float,
    alpha: float = 0.05,
    target_power: float = 0.80,
    robust: bool = False,
    per_rep_means: list[float] | None = None,
) -> int:
    """Compute the number of reps per side needed to detect a given MDE.

    Uses the formula: n = 2 * ((z_alpha + z_beta) / (mde / sigma))^2

    When ``robust=True`` and ``per_rep_means`` is provided, uses the
    MAD-scaled estimator instead of the sample standard deviation for
    sigma, which is more resistant to outliers.
    """
    if robust and per_rep_means and len(per_rep_means) >= 3:
        import numpy as np

        arr = np.array(per_rep_means)
        mad = float(np.median(np.abs(arr - np.median(arr)))) * 1.4826
        if mad > 0:
            std = mad
    if std <= 0 or mde_pct <= 0 or baseline_mean == 0:
        return 1
    mde_abs = abs(baseline_mean) * mde_pct / 100.0
    z_alpha = float(scipy_stats.norm.ppf(1.0 - alpha / 2))
    z_beta = float(scipy_stats.norm.ppf(target_power))
    n = 2.0 * ((z_alpha + z_beta) / (mde_abs / std)) ** 2
    return max(2, math.ceil(n))


@dataclass
class KpiComparison:
    """Comparison result for a single KPI between two runs."""

    name: str
    unit: str
    higher_is_better: bool

    # Run A (current / iteration)
    a_mean: float | None
    a_std: float | None
    a_per_rep: list[float]

    # Run B (baseline)
    b_mean: float | None
    b_std: float | None
    b_per_rep: list[float]

    # Deltas
    delta: float | None  # a_mean - b_mean
    delta_pct: float | None  # percentage change
    p_value: float | None  # Welch's t-test p-value
    significant: bool  # p < 0.05
    verdict: str  # "improvement", "regression", "neutral", "no_data"
    regression_threshold_pct: float

    # Enhanced statistics
    cohens_d: float | None = None  # effect size
    ci_lower: float | None = None  # 95% CI lower bound on delta
    ci_upper: float | None = None  # 95% CI upper bound on delta
    power: float | None = None  # post-hoc statistical power
    effect_size_label: str = "N/A"  # "negligible"/"small"/"medium"/"large"

    # Multiple comparisons (--strict mode, Holm-Bonferroni)
    adjusted_p_value: float | None = None  # Holm-Bonferroni adjusted p-value
    correction_method: str | None = None  # e.g. "holm-bonferroni"

    # Serial correlation diagnostics (Durbin-Watson)
    durbin_watson_a: float | None = None  # DW statistic for iteration per-rep means
    durbin_watson_b: float | None = None  # DW statistic for baseline per-rep means

    # Robust statistics diagnostics
    normality_warning: bool = False  # True if either side fails Shapiro-Wilk
    shapiro_p_a: float | None = None  # Shapiro-Wilk p for iteration
    shapiro_p_b: float | None = None  # Shapiro-Wilk p for baseline

    # Non-parametric test
    mann_whitney_p: float | None = None  # Mann-Whitney U p-value
    test_used: str = "welch"  # "welch" or "mann_whitney" (which drove verdict)

    # Non-parametric effect size
    cliffs_delta: float | None = None
    cliffs_delta_label: str = "N/A"

    # Bootstrap CI (computed when normality is violated)
    bootstrap_ci_lower: float | None = None
    bootstrap_ci_upper: float | None = None

    # Outlier diagnostics
    outlier_count_a: int = 0  # IQR outliers in iteration per-rep means
    outlier_count_b: int = 0  # IQR outliers in baseline per-rep means

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict for JSON persistence."""
        return asdict(self)


@dataclass
class TopologyDiff:
    """Result of comparing topologies between two runs."""

    changed: bool
    pipelines_added: list[str]
    pipelines_removed: list[str]
    elements_changed_thread: list[dict[str, Any]]
    edges_added: list[dict[str, Any]]
    edges_removed: list[dict[str, Any]]
    invalidated_kpis: list[str]
    details: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# KPIs that become unreliable when thread assignments change
_THREAD_SENSITIVE_KPIS = {
    "pipeline_cpu_pct",
    "top_thread_cpu_pct",
}

# KPIs that become unreliable when pipeline structure (elements/edges) changes
_STRUCTURE_SENSITIVE_KPIS = {
    "throughput_fps",
    "freshness_delay_ms",
    "edge_max_freshness_delay_ms",
    "edge_min_causal_confidence",
    "total_stutter_events",
    "total_freeze_events",
    "max_freeze_ms",
    "max_stutter_ratio",
    "interval_p95_ms",
    "interval_p99_ms",
    "interval_std_ms",
}


def compare_topology(
    agg_a: dict[str, Any],
    agg_b: dict[str, Any],
) -> TopologyDiff:
    """Compare topologies from two aggregate dicts.

    Returns a :class:`TopologyDiff` describing what changed and which KPIs
    are invalidated.
    """
    topo_a = agg_a.get("topology") or {}
    topo_b = agg_b.get("topology") or {}

    pipes_a = {
        p["pipeline_name"]: p for p in (topo_a.get("pipelines") or []) if isinstance(p, dict)
    }
    pipes_b = {
        p["pipeline_name"]: p for p in (topo_b.get("pipelines") or []) if isinstance(p, dict)
    }

    names_a = set(pipes_a.keys())
    names_b = set(pipes_b.keys())

    pipelines_added = sorted(names_a - names_b)
    pipelines_removed = sorted(names_b - names_a)

    elements_changed_thread: list[dict[str, Any]] = []
    edges_added: list[dict[str, Any]] = []
    edges_removed: list[dict[str, Any]] = []

    # Compare elements in shared pipelines
    for pname in sorted(names_a & names_b):
        elems_a = pipes_a[pname].get("elements") or {}
        elems_b = pipes_b[pname].get("elements") or {}

        for ename in set(elems_a.keys()) | set(elems_b.keys()):
            ea = elems_a.get(ename, {})
            eb = elems_b.get(ename, {})
            tid_a = ea.get("thread_id") if isinstance(ea, dict) else None
            tid_b = eb.get("thread_id") if isinstance(eb, dict) else None
            if tid_a != tid_b:
                elements_changed_thread.append(
                    {
                        "pipeline": pname,
                        "element": ename,
                        "a_thread_id": tid_a,
                        "b_thread_id": tid_b,
                    }
                )

        # Compare edges
        def _edge_key(e: dict) -> tuple:
            return (e.get("from_element", ""), e.get("to_element", ""))

        edges_a_list = pipes_a[pname].get("edges") or []
        edges_b_list = pipes_b[pname].get("edges") or []
        keys_a = {_edge_key(e) for e in edges_a_list if isinstance(e, dict)}
        keys_b = {_edge_key(e) for e in edges_b_list if isinstance(e, dict)}

        for key in keys_a - keys_b:
            edges_added.append({"pipeline": pname, "from": key[0], "to": key[1]})
        for key in keys_b - keys_a:
            edges_removed.append({"pipeline": pname, "from": key[0], "to": key[1]})

    # Determine which KPIs are invalidated
    invalidated: set[str] = set()

    if pipelines_added or pipelines_removed:
        # Pipeline set changed -- all structure-sensitive KPIs are invalid
        invalidated |= _STRUCTURE_SENSITIVE_KPIS | _THREAD_SENSITIVE_KPIS

    if elements_changed_thread:
        invalidated |= _THREAD_SENSITIVE_KPIS

    if edges_added or edges_removed:
        invalidated |= _STRUCTURE_SENSITIVE_KPIS

    changed = bool(
        pipelines_added
        or pipelines_removed
        or elements_changed_thread
        or edges_added
        or edges_removed
    )

    # Build a human-readable summary
    details_parts: list[str] = []
    if pipelines_added:
        details_parts.append(f"Pipelines added: {', '.join(pipelines_added)}")
    if pipelines_removed:
        details_parts.append(f"Pipelines removed: {', '.join(pipelines_removed)}")
    if elements_changed_thread:
        for ec in elements_changed_thread:
            details_parts.append(
                f"Element '{ec['element']}' in '{ec['pipeline']}' moved "
                f"from thread {ec['b_thread_id']} to {ec['a_thread_id']}"
            )
    if edges_added:
        details_parts.append(f"{len(edges_added)} edge(s) added")
    if edges_removed:
        details_parts.append(f"{len(edges_removed)} edge(s) removed")

    return TopologyDiff(
        changed=changed,
        pipelines_added=pipelines_added,
        pipelines_removed=pipelines_removed,
        elements_changed_thread=elements_changed_thread,
        edges_added=edges_added,
        edges_removed=edges_removed,
        invalidated_kpis=sorted(invalidated),
        details="; ".join(details_parts) if details_parts else "No topology changes",
    )


@dataclass
class RunComparison:
    """Full comparison between two runs."""

    run_a_dir: str
    run_b_dir: str
    kpis: list[KpiComparison]
    regressions: list[KpiComparison]
    improvements: list[KpiComparison]
    topology_diff: TopologyDiff | None = None


def compare_runs(
    run_a_dir: Path,
    run_b_dir: Path,
    significance_level: float = 0.05,
    strict: bool = False,
) -> RunComparison:
    """
    Compare run A (iteration) against run B (baseline).

    Uses ``spotify-confidence.StudentsTTest`` for Welch's t-test,
    p-values, CIs, and power.  Falls back to scipy if the library
    call fails for any KPI.

    Parameters
    ----------
    strict : bool
        When ``True``, apply Holm-Bonferroni step-down correction
        across all KPIs to control the family-wise error rate (FWER)
        at ``significance_level``.  Adjusted p-values are stored in
        :attr:`KpiComparison.adjusted_p_value` and significance is
        determined from the adjusted value instead of the raw p-value.

    Returns a RunComparison with per-KPI results.
    """
    agg_a = load_aggregate(run_a_dir)
    agg_b = load_aggregate(run_b_dir)

    a_data = agg_a.get("aggregate", {})
    b_data = agg_b.get("aggregate", {})

    # Topology comparison
    topo_diff = compare_topology(agg_a, agg_b)
    if topo_diff.changed:
        log.warning("Topology changed between runs: %s", topo_diff.details)
        if topo_diff.invalidated_kpis:
            log.warning("Invalidated KPIs: %s", ", ".join(topo_diff.invalidated_kpis))

    comparisons: list[KpiComparison] = []

    for kpi_def in KPI_REGISTRY:
        name = kpi_def.name
        a_kpi = a_data.get(name, {})
        b_kpi = b_data.get(name, {})

        a_mean = a_kpi.get("mean")
        a_std = a_kpi.get("std")
        a_per_rep = a_kpi.get("per_rep_means", [])

        b_mean = b_kpi.get("mean")
        b_std = b_kpi.get("std")
        b_per_rep = b_kpi.get("per_rep_means", [])

        # Compute delta
        delta: float | None = None
        delta_pct: float | None = None
        p_value: float | None = None
        significant = False
        ci_lo: float | None = None
        ci_hi: float | None = None
        pwr: float | None = None
        verdict = "no_data"

        if a_mean is not None and b_mean is not None:
            delta = a_mean - b_mean
            assert delta is not None

            if b_mean != 0:
                delta_pct = (delta / abs(b_mean)) * 100
            elif a_mean != 0:
                delta_pct = float("inf") if delta > 0 else float("-inf")
            else:
                delta_pct = 0.0

            # Statistical test via spotify-confidence (needs >=2 samples/side)
            if len(a_per_rep) >= 2 and len(b_per_rep) >= 2:
                try:
                    ttest = _spotify_ttest(a_per_rep, b_per_rep, alpha=significance_level)
                    p_value = ttest["p_value"]
                    ci_lo = ttest["ci_lower"]
                    ci_hi = ttest["ci_upper"]
                    pwr = ttest["power"]
                    significant = p_value is not None and p_value < significance_level
                except Exception:
                    pass

            # Verdict -- check topology invalidation first
            if name in (topo_diff.invalidated_kpis if topo_diff.changed else []):
                verdict = "topology_changed"
            elif delta_pct is not None and delta is not None:
                if math.isfinite(delta_pct):
                    is_better = (delta > 0 and kpi_def.higher_is_better) or (
                        delta < 0 and not kpi_def.higher_is_better
                    )
                    is_worse = not is_better and delta != 0
                    threshold_exceeded = abs(delta_pct) > kpi_def.regression_pct

                    if is_worse and threshold_exceeded and significant:
                        verdict = "regression"
                    elif is_better and threshold_exceeded and significant:
                        verdict = "improvement"
                    elif is_worse and threshold_exceeded:
                        verdict = "likely_regression"
                    elif is_better and threshold_exceeded:
                        verdict = "likely_improvement"
                    else:
                        verdict = "neutral"
                else:
                    # Non-finite delta_pct: baseline mean was zero.
                    # Any from-zero change implicitly exceeds any percentage
                    # threshold, so treat threshold_exceeded as True.
                    is_better = (delta > 0 and kpi_def.higher_is_better) or (
                        delta < 0 and not kpi_def.higher_is_better
                    )
                    is_worse = not is_better and delta != 0

                    if is_worse and significant:
                        verdict = "regression"
                    elif is_better and significant:
                        verdict = "improvement"
                    elif is_worse:
                        verdict = "likely_regression"
                    elif is_better:
                        verdict = "likely_improvement"
                    else:
                        verdict = "neutral"

        # Cohen's d and effect-size label (lightweight, keep inline)
        d = cohens_d(a_per_rep, b_per_rep)
        es_label = _effect_size_label(d)

        # --- Robust statistics ---
        # Cliff's delta
        cd = cliffs_delta(a_per_rep, b_per_rep)
        cd_label = _cliffs_delta_label(cd)

        # Normality diagnostics (Shapiro-Wilk)
        shap_a = _shapiro_wilk(a_per_rep) if len(a_per_rep) >= 4 else None
        shap_b = _shapiro_wilk(b_per_rep) if len(b_per_rep) >= 4 else None
        normality_warn = False
        if (shap_a is not None and shap_a < significance_level) or (
            shap_b is not None and shap_b < significance_level
        ):
            normality_warn = True

        # Mann-Whitney U (non-parametric alternative)
        mw_p = _mann_whitney_u(a_per_rep, b_per_rep)

        # If normality is violated, use Mann-Whitney for verdict
        test_used = "welch"
        if normality_warn and mw_p is not None:
            test_used = "mann_whitney"
            p_value = mw_p
            significant = p_value < significance_level

            # Re-derive verdict to stay consistent with updated significance.
            # Must handle all cases to avoid a stale Welch-based verdict.
            if delta is not None and delta_pct is not None:
                if name in (topo_diff.invalidated_kpis if topo_diff.changed else []):
                    verdict = "topology_changed"
                elif math.isfinite(delta_pct):
                    is_better = (delta > 0 and kpi_def.higher_is_better) or (
                        delta < 0 and not kpi_def.higher_is_better
                    )
                    is_worse = not is_better and delta != 0
                    threshold_exceeded = abs(delta_pct) > kpi_def.regression_pct

                    if is_worse and threshold_exceeded and significant:
                        verdict = "regression"
                    elif is_better and threshold_exceeded and significant:
                        verdict = "improvement"
                    elif is_worse and threshold_exceeded:
                        verdict = "likely_regression"
                    elif is_better and threshold_exceeded:
                        verdict = "likely_improvement"
                    else:
                        verdict = "neutral"
                else:
                    # Non-finite delta_pct: baseline mean was zero.
                    # Any from-zero change implicitly exceeds any percentage
                    # threshold, so treat threshold_exceeded as True.
                    is_better = (delta > 0 and kpi_def.higher_is_better) or (
                        delta < 0 and not kpi_def.higher_is_better
                    )
                    is_worse = not is_better and delta != 0

                    if is_worse and significant:
                        verdict = "regression"
                    elif is_better and significant:
                        verdict = "improvement"
                    elif is_worse:
                        verdict = "likely_regression"
                    elif is_better:
                        verdict = "likely_improvement"
                    else:
                        verdict = "neutral"

        # Bootstrap CI (only when normality is violated)
        boot_ci: tuple[float, float] | None = None
        if normality_warn:
            boot_ci = _bootstrap_ci(a_per_rep, b_per_rep, alpha=significance_level)

        # IQR outlier counts
        out_a = _count_iqr_outliers(a_per_rep)
        out_b = _count_iqr_outliers(b_per_rep)

        comparisons.append(
            KpiComparison(
                name=name,
                unit=kpi_def.unit,
                higher_is_better=kpi_def.higher_is_better,
                a_mean=a_mean,
                a_std=a_std,
                a_per_rep=a_per_rep,
                b_mean=b_mean,
                b_std=b_std,
                b_per_rep=b_per_rep,
                delta=delta,
                delta_pct=delta_pct,
                p_value=p_value,
                significant=significant,
                verdict=verdict,
                regression_threshold_pct=kpi_def.regression_pct,
                cohens_d=d,
                ci_lower=ci_lo,
                ci_upper=ci_hi,
                power=pwr,
                effect_size_label=es_label,
                normality_warning=normality_warn,
                shapiro_p_a=shap_a,
                shapiro_p_b=shap_b,
                mann_whitney_p=mw_p,
                test_used=test_used,
                cliffs_delta=cd,
                cliffs_delta_label=cd_label,
                bootstrap_ci_lower=boot_ci[0] if boot_ci else None,
                bootstrap_ci_upper=boot_ci[1] if boot_ci else None,
                outlier_count_a=out_a,
                outlier_count_b=out_b,
            )
        )

    # --- Holm-Bonferroni correction (--strict mode) ---
    if strict:
        raw_p_values = [c.p_value for c in comparisons]
        adjusted = _holm_bonferroni(raw_p_values, alpha=significance_level)
        for comp, adj_p in zip(comparisons, adjusted, strict=False):
            comp.adjusted_p_value = adj_p
            comp.correction_method = "holm-bonferroni"
            if adj_p is not None:
                comp.significant = adj_p < significance_level
                # Re-derive verdict with the corrected significance
                if comp.verdict in ("regression", "likely_regression"):
                    comp.verdict = "regression" if comp.significant else "likely_regression"
                elif comp.verdict in ("improvement", "likely_improvement"):
                    comp.verdict = "improvement" if comp.significant else "likely_improvement"
        log.info(
            "Holm-Bonferroni correction applied (strict mode, %d KPIs, alpha=%.3f).",
            len(comparisons),
            significance_level,
        )

    # --- Durbin-Watson serial correlation diagnostic ---
    for comp in comparisons:
        if len(comp.a_per_rep) >= 20:
            comp.durbin_watson_a = _durbin_watson(comp.a_per_rep)
            if comp.durbin_watson_a is not None and (
                comp.durbin_watson_a < 1.5 or comp.durbin_watson_a > 2.5
            ):
                log.warning(
                    "Serial correlation detected in iteration (A) per-rep means for %s: "
                    "Durbin-Watson=%.3f (expected ~2.0). "
                    "T-test independence assumption may be violated.",
                    comp.name,
                    comp.durbin_watson_a,
                )
        if len(comp.b_per_rep) >= 20:
            comp.durbin_watson_b = _durbin_watson(comp.b_per_rep)
            if comp.durbin_watson_b is not None and (
                comp.durbin_watson_b < 1.5 or comp.durbin_watson_b > 2.5
            ):
                log.warning(
                    "Serial correlation detected in baseline (B) per-rep means for %s: "
                    "Durbin-Watson=%.3f (expected ~2.0). "
                    "T-test independence assumption may be violated.",
                    comp.name,
                    comp.durbin_watson_b,
                )

    # --- Normality & outlier diagnostics ---
    for comp in comparisons:
        if comp.normality_warning:
            log.warning(
                "Normality violated for %s (Shapiro p: A=%.3f, B=%.3f) -- "
                "using Mann-Whitney U (p=%.4f) instead of Welch t-test.",
                comp.name,
                comp.shapiro_p_a if comp.shapiro_p_a is not None else -1,
                comp.shapiro_p_b if comp.shapiro_p_b is not None else -1,
                comp.mann_whitney_p if comp.mann_whitney_p is not None else -1,
            )
        if comp.outlier_count_a > 0 or comp.outlier_count_b > 0:
            log.warning(
                "IQR outliers in %s: A=%d, B=%d per-rep means.",
                comp.name,
                comp.outlier_count_a,
                comp.outlier_count_b,
            )

    regressions = [c for c in comparisons if "regression" in c.verdict]
    improvements = [c for c in comparisons if "improvement" in c.verdict]

    result = RunComparison(
        run_a_dir=str(run_a_dir),
        run_b_dir=str(run_b_dir),
        kpis=comparisons,
        regressions=regressions,
        improvements=improvements,
        topology_diff=topo_diff,
    )

    # Log summary
    log.info(
        "Comparison (spotify-confidence): %d KPIs, %d regressions, %d improvements%s",
        len(comparisons),
        len(regressions),
        len(improvements),
        " [strict/Holm-Bonferroni]" if strict else "",
    )
    for r in regressions:
        log.warning(
            "  REGRESSION: %s: %.2f -> %.2f (%+.1f%%, p=%.3f%s)",
            r.name,
            r.b_mean or 0,
            r.a_mean or 0,
            r.delta_pct or 0,
            r.p_value or 1,
            f", adj_p={r.adjusted_p_value:.3f}" if r.adjusted_p_value is not None else "",
        )

    return result


def save_comparison(comparison: RunComparison, path: Path) -> None:
    """Serialize comparison to JSON."""
    data: dict[str, Any] = {
        "run_a_dir": comparison.run_a_dir,
        "run_b_dir": comparison.run_b_dir,
        "kpis": [c.to_dict() for c in comparison.kpis],
        "regression_count": len(comparison.regressions),
        "improvement_count": len(comparison.improvements),
    }
    if comparison.topology_diff is not None:
        data["topology_diff"] = comparison.topology_diff.to_dict()
    path.write_text(json.dumps(data, indent=2, default=str))
