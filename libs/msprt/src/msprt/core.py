"""Mixture Sequential Probability Ratio Test (mSPRT).

A Python implementation of the mSPRT for two-sample sequential hypothesis
testing, following Johari, Pekelis & Walsh (2017) "Peeking at A/B Tests:
Why it matters, and what to do about it."

The reference R/C++ implementation is erik-giertz/mixtureSPRT on GitHub.

References
----------
- Johari R., Koomen P., Pekelis L. & Walsh D. (2017),
  "Peeking at A/B Tests", ACM KDD.
- Johari R., Pekelis L. & Walsh D. (2022),
  "Always Valid Inference: Continuous Monitoring of A/B Tests",
  Operations Research 70(3):1806-1821.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.stats import norm


def compute_tau(alpha: float, sigma: float, truncation: int) -> float:
    """Compute the optimal mixing standard deviation for the mSPRT.

    This is the square root of the mixture variance tau^2.  It controls
    the prior width over the effect size under the alternative hypothesis.

    Direct translation of the R ``calcTau`` from mixtureSPRT::

        b <- (2*log(alpha^(-1)))/(truncation*sigma^2)^(1/2)
        tau^2 <- sigma^2 * (pnorm(-b) / ((1/b)*dnorm(b) - pnorm(-b)))

    Parameters
    ----------
    alpha : float
        Significance level (e.g. 0.05).
    sigma : float
        Known (or estimated) population standard deviation.
    truncation : int
        Maximum number of observation *pairs* (horizon).

    Returns
    -------
    float
        tau -- the mixing standard deviation (sqrt of mixture variance).

    Raises
    ------
    ValueError
        If inputs are out of valid range.
    """
    if not (0 < alpha < 1):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}")
    if truncation < 1:
        raise ValueError(f"truncation must be >= 1, got {truncation}")

    b = (2.0 * math.log(1.0 / alpha)) / math.sqrt(truncation * sigma**2)
    numerator = norm.cdf(-b)
    denominator = (1.0 / b) * norm.pdf(b) - norm.cdf(-b)

    if denominator <= 0:
        raise ValueError(
            f"Degenerate mixing variance (denominator={denominator:.6e}). "
            f"Check alpha={alpha}, sigma={sigma}, truncation={truncation}."
        )

    tau_sq = sigma**2 * (numerator / denominator)
    return math.sqrt(tau_sq)


def msprt_statistic(
    x: np.ndarray,
    y: np.ndarray,
    sigma: float,
    tau: float,
    theta: float = 0.0,
) -> np.ndarray:
    """Compute the mSPRT likelihood ratio trajectory for normal data.

    For each n = 1, 2, ..., N, computes::

        Lambda_n = sqrt(2*sigma^2 / (2*sigma^2 + n*tau^2))
                 * exp(n^2 * tau^2 * (mean(x[0:n]) - mean(y[0:n]) - theta)^2
                       / (4 * sigma^2 * (2*sigma^2 + n*tau^2)))

    Parameters
    ----------
    x, y : np.ndarray
        Observation vectors of equal length (treatment and control).
    sigma : float
        Known (or estimated) population standard deviation.
    tau : float
        Mixing standard deviation (from :func:`compute_tau`).
    theta : float
        Hypothesised difference under H0 (default 0).

    Returns
    -------
    np.ndarray
        Array of length N with the likelihood ratio at each observation.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError(f"x and y must have the same shape, got {x.shape} vs {y.shape}")

    n = len(x)
    if n == 0:
        return np.array([], dtype=np.float64)

    # Cumulative means via cumulative sums
    cum_sum_x = np.cumsum(x)
    cum_sum_y = np.cumsum(y)
    ns = np.arange(1, n + 1, dtype=np.float64)
    cum_mean_x = cum_sum_x / ns
    cum_mean_y = cum_sum_y / ns

    double_var = 2.0 * sigma**2
    tau_sq = tau**2

    # Vectorised Lambda computation
    denom = double_var + ns * tau_sq
    root_part = np.sqrt(double_var / denom)
    diff = cum_mean_x - cum_mean_y - theta
    exp_part = np.exp(ns**2 * tau_sq * diff**2 / (2.0 * double_var * denom))

    result: np.ndarray[Any, Any] = root_part * exp_part
    return result


@dataclass(frozen=True)
class MsprtResult:
    """Result of an mSPRT sequential test.

    Attributes
    ----------
    n_obs : int
        Total number of observation pairs processed.
    trajectory : np.ndarray
        The likelihood ratio Lambda_n at each observation n=1..n_obs.
    n_rejection : int | None
        The first observation index (1-based) at which H0 was rejected,
        or None if H0 was never rejected.
    rejected : bool
        Whether H0 was rejected at any point.
    alpha : float
        The significance level used.
    """

    n_obs: int
    trajectory: np.ndarray
    n_rejection: int | None
    rejected: bool
    alpha: float


def msprt_test(
    x: np.ndarray,
    y: np.ndarray,
    sigma: float,
    alpha: float = 0.05,
    truncation: int | None = None,
    theta: float = 0.0,
    min_n: int = 1,
) -> MsprtResult:
    """Run a complete mSPRT test on two observation vectors.

    High-level API that computes the mixing parameter, runs the
    sequential test, and returns a structured result.

    Parameters
    ----------
    x, y : array-like
        Treatment and control observations (equal length).
    sigma : float
        Known (or estimated) population standard deviation.
    alpha : float
        Significance level (default 0.05).
    truncation : int | None
        Horizon for tau computation.  Defaults to ``len(x)``.
    theta : float
        Hypothesised difference under H0 (default 0).
    min_n : int
        Minimum observations before rejection is considered (default 1).

    Returns
    -------
    MsprtResult
        Structured result with trajectory, rejection point, etc.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = len(x)

    if truncation is None:
        truncation = n

    tau = compute_tau(alpha=alpha, sigma=sigma, truncation=truncation)
    trajectory = msprt_statistic(x, y, sigma=sigma, tau=tau, theta=theta)

    threshold = 1.0 / alpha

    # Find first rejection at or after min_n
    n_rejection: int | None = None
    for i in range(max(0, min_n - 1), n):
        if trajectory[i] > threshold:
            n_rejection = i + 1  # 1-based
            break

    return MsprtResult(
        n_obs=n,
        trajectory=trajectory,
        n_rejection=n_rejection,
        rejected=n_rejection is not None,
        alpha=alpha,
    )
