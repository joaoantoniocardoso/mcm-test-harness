#!/usr/bin/env Rscript
#
# Generate golden values for the Python mSPRT library by evaluating the
# formulas from the R mixtureSPRT reference implementation.
#
# Reference: https://github.com/erik-giertz/mixtureSPRT
# Paper:     Johari, Pekelis & Walsh (2017), "Peeking at A/B Tests"
#
# Usage:
#   Rscript generate_golden_values.R
#
# This writes golden_values.json in the same directory.  The Python test
# suite (test_core.py) loads this file and cross-validates against it.
# If you change any test inputs, re-run this script to update the JSON.
#
# The formulas below are a direct transcription from the R reference.
# They use R's pnorm/dnorm, which are independently implemented from
# Python's scipy.stats.norm.{cdf,pdf}, making this a true cross-language
# validation.

library(jsonlite)

# ---------------------------------------------------------------------------
# calcTau -- from mixtureSPRT::calcTau
# ---------------------------------------------------------------------------
calcTau <- function(alpha, sigma, truncation) {
  b <- (2 * log(alpha^(-1))) / (truncation * sigma^2)^(1/2)
  tau_sq <- sigma^2 * (pnorm(-b) / ((1/b) * dnorm(b) - pnorm(-b)))
  return(sqrt(tau_sq))
}

# ---------------------------------------------------------------------------
# mSPRT Lambda -- from mixtureSPRT
# ---------------------------------------------------------------------------
msprt_lambda <- function(x, y, sigma, tau, theta = 0.0) {
  n <- length(x)
  stopifnot(length(y) == n)
  if (n == 0) return(numeric(0))

  cum_mean_x <- cumsum(x) / seq_len(n)
  cum_mean_y <- cumsum(y) / seq_len(n)
  ns <- seq_len(n)

  double_var <- 2.0 * sigma^2
  tau_sq <- tau^2

  denom <- double_var + ns * tau_sq
  root_part <- sqrt(double_var / denom)
  diff <- cum_mean_x - cum_mean_y - theta
  exp_part <- exp(ns^2 * tau_sq * diff^2 / (2.0 * double_var * denom))

  return(root_part * exp_part)
}

# ---------------------------------------------------------------------------
# Generate all golden values
# ---------------------------------------------------------------------------

# 1. compute_tau cases
tau_cases <- list(
  list(alpha = 0.05, sigma = 1.0, truncation = 10000L),
  list(alpha = 0.05, sigma = 1.0, truncation = 200L),
  list(alpha = 0.01, sigma = 2.0, truncation = 500L)
)

tau_results <- lapply(tau_cases, function(c) {
  tau <- calcTau(c$alpha, c$sigma, c$truncation)
  cat(sprintf("calcTau(alpha=%.2f, sigma=%.1f, N=%d) = %.15e\n",
              c$alpha, c$sigma, c$truncation, tau))
  list(
    alpha = c$alpha,
    sigma = c$sigma,
    truncation = c$truncation,
    expected_tau = tau
  )
})

# 2. msprt_statistic cases
sigma <- 1.0
alpha <- 0.05
truncation <- 10L
tau <- calcTau(alpha, sigma, truncation)
cat(sprintf("\ntau for statistic tests = %.15e\n\n", tau))

x_no <- c(1.2, 0.8, 1.5, 0.9, 1.1, 1.3, 0.7, 1.4, 1.0, 1.6)
y_no <- c(0.9, 1.0, 0.8, 1.1, 0.7, 1.0, 0.9, 0.8, 1.1, 0.7)
x_lg <- c(3.1, 2.8, 3.3, 2.9, 3.0, 3.2, 2.7, 3.4, 3.1, 2.8)
y_lg <- c(0.9, 1.0, 0.8, 1.1, 0.7, 1.0, 0.9, 0.8, 1.1, 0.7)

lambda_no <- msprt_lambda(x_no, y_no, sigma, tau)
lambda_lg <- msprt_lambda(x_lg, y_lg, sigma, tau)

cat("No-effect Lambda trajectory:\n")
cat(sprintf("  %.15e\n", lambda_no))
cat("\nLarge-effect Lambda trajectory:\n")
cat(sprintf("  %.15e\n", lambda_lg))

# 3. Assemble and write JSON
golden <- list(
  `_comment` = "Golden values for mSPRT cross-validation. Regenerate with: Rscript generate_golden_values.R",
  compute_tau = tau_results,
  msprt_statistic = list(
    sigma = sigma,
    alpha = alpha,
    truncation = truncation,
    tau = tau,
    no_effect = list(
      x = x_no,
      y = y_no,
      expected_lambda = lambda_no
    ),
    large_effect = list(
      x = x_lg,
      y = y_lg,
      expected_lambda = lambda_lg
    )
  )
)

out_path <- file.path(dirname(sys.frame(1)$ofile %||% "."), "golden_values.json")
write_json(golden, out_path, pretty = TRUE, digits = 15, auto_unbox = TRUE)
cat(sprintf("\nGolden values written to: %s\n", out_path))
