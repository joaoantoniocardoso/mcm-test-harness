# Statistical Methodology: A Guide for Programmers

This document explains the statistical methods used by the A/B test harness in
plain language. You don't need a statistics background -- just curiosity about
why we do things the way we do.

## Table of Contents

- [The Problem We're Solving](#the-problem-we-are-solving)
- [Why We Need Statistics at All](#why-we-need-statistics-at-all)
- [Core Concepts](#core-concepts)
  - [Repetitions and Averages](#repetitions-and-averages)
  - [Standard Deviation: Measuring Noise](#standard-deviation-measuring-noise)
  - [Hypothesis Testing: Guilty Until Proven Innocent](#hypothesis-testing-guilty-until-proven-innocent)
  - [P-values: "How Surprised Should I Be?"](#p-values-how-surprised-should-i-be)
  - [Confidence Intervals: "Where Does the Truth Likely Live?"](#confidence-intervals-where-does-the-truth-likely-live)
  - [Effect Size: "Is This Difference Meaningful?"](#effect-size-is-this-difference-meaningful)
  - [Statistical Power: "Can I Even Detect This?"](#statistical-power-can-i-even-detect-this)
- [How a Test Runs](#how-a-test-runs)
  - [Step 1: Collect Baseline Data](#step-1-collect-baseline-data)
  - [Step 2: Collect Iteration Data](#step-2-collect-iteration-data)
  - [Step 3: Compare With a t-test](#step-3-compare-with-a-t-test)
  - [Step 4: Render a Verdict](#step-4-render-a-verdict)
- [Interleaved Collection: Fighting Time-based Noise](#interleaved-collection-fighting-time-based-noise)
- [Adaptive Stopping: Don't Wait If You Already Know](#adaptive-stopping-dont-wait-if-you-already-know)
  - [The Multiple Peeking Problem](#the-multiple-peeking-problem)
  - [Method 1: Group Sequential Testing (GST)](#method-1-group-sequential-testing-gst)
  - [Method 2: Mixture Sequential Probability Ratio Test (mSPRT)](#method-2-mixture-sequential-probability-ratio-test-msprt)
  - [Futility Stopping: Knowing When to Give Up](#futility-stopping-knowing-when-to-give-up)
  - [Which Method Should I Use?](#which-method-should-i-use)
- [Reducing Noise: Isolation and Hardening](#reducing-noise-isolation-and-hardening)
- [Sample Size: "How Many Reps Do I Need?"](#sample-size-how-many-reps-do-i-need)
- [Reading the Report](#reading-the-report)
  - [The Verdict System](#the-verdict-system)
  - [Key Numbers to Look At](#key-numbers-to-look-at)
- [Common Pitfalls](#common-pitfalls)
- [Glossary](#glossary)
- [Further Reading](#further-reading)

---

## The Problem We're Solving

You changed some code in MCM (Mavlink Camera Manager). You want to know: **did
my change make things better, worse, or has it no measurable impact?**

This sounds simple. Run the old code, measure CPU usage. Run the new code,
measure CPU usage. Compare the numbers. Done, right?

Not quite. The problem is **noise**. Every time you measure something on a real
system, the numbers wiggle. CPU usage might be 47.2% one second and 48.1% the
next, even with the exact same code running. Temperature fluctuates. The OS
scheduler makes different decisions. Network traffic varies. A background
process wakes up.

If baseline CPU is 47.5% and iteration CPU is 48.3%, is that a real regression
or just noise? Statistics gives us the tools to answer that question with
controlled confidence.

## Why We Need Statistics at All

Imagine you flip a coin 10 times and get 7 heads. Is the coin unfair? Maybe.
But even a perfectly fair coin gives 7+ heads about 17% of the time. You'd want
more flips before concluding anything.

Performance measurements work the same way. A single run tells you almost
nothing, because you can't separate the real signal (your code change) from the
noise (everything else going on in the system). Statistics is the discipline of
separating signal from noise, and quantifying how confident we should be in
what we've found.

The harness uses statistics to answer three questions:

1. **Is there a difference?** (hypothesis testing, p-values)
2. **How big is the difference?** (effect size, confidence intervals)
3. **Should we trust this result?** (power, sample size)

## Core Concepts

### Repetitions and Averages

The harness doesn't measure once -- it measures multiple **repetitions**
("reps"). Each rep is an independent measurement window (e.g., 60 seconds of
data collection). Running 10 reps gives us 10 independent measurements of
each KPI.

We then look at the **mean** (average) of those reps. The mean smooths out
per-rep noise and gives a better estimate of the "true" value.

> **Analogy**: If you want to know how long your commute takes, you wouldn't
> time it once. You'd time it on 10 different days and average.

### Standard Deviation: Measuring Noise

The **standard deviation** (std) tells you how spread out your measurements
are. A small std means your measurements are consistent and tightly clustered;
a large std means they bounce around a lot.

- CPU at 48.0% ± 0.5% std → very stable, easy to detect small changes
- CPU at 48.0% ± 5.0% std → noisy, you'd need a big change to notice it

Standard deviation is the single most important number for planning experiments.
High noise means you need more reps to see through it.

### Hypothesis Testing: Guilty Until Proven Innocent

When comparing baseline vs. iteration, we start with a deliberate assumption:

> **Null hypothesis (H0)**: "There is no real difference. Any variation we see
> is just noise."

This is the innocent-until-proven-guilty principle. We require **evidence** to
claim a difference exists. The statistical test then asks: *"If there were truly
no difference, how unlikely would it be to see data this extreme?"*

If the answer is "very unlikely" (less than 5% chance), we **reject** the null
hypothesis and say the difference is **statistically significant**.

Why 5%? It's a convention (called the **significance level**, or alpha = 0.05).
It means we accept a 5% chance of a **false alarm** -- claiming a difference
exists when it doesn't. You could use 1% for stricter control, or 10% for more
sensitivity. The harness defaults to 5%.

### P-values: "How Surprised Should I Be?"

The **p-value** is the probability of seeing a difference *at least this large*
if the null hypothesis were true (i.e., if your code change had zero effect).

- **p = 0.80** → "An 80% chance noise alone explains this." Not significant at all.
- **p = 0.15** → "A 15% chance." Suggestive, but not conclusive.
- **p = 0.03** → "Only a 3% chance this is noise." Below 0.05, so we call it
  significant.
- **p = 0.001** → "A 0.1% chance." Very strong evidence of a real difference.

**What p-values are NOT:**
- The p-value is *not* the probability that your change had no effect. (This is
  a common and subtle misunderstanding.)
- A p-value of 0.03 does not mean "97% sure the change is real." It means
  "if the change did nothing, we'd see data this extreme only 3% of the time."

**The specific test we use: Welch's t-test.** This is a standard statistical
test for comparing two group means when the groups might have different amounts
of variability (which is common -- your code change might make things more or
less stable). It's the default choice for A/B testing with continuous
measurements and moderate sample sizes.

### Confidence Intervals: "Where Does the Truth Likely Live?"

Instead of just asking "is there a difference?", we can ask "how big is the
difference, and what's the range of plausible values?"

A **95% confidence interval** (CI) gives a range that, over many experiments,
would contain the true difference 95% of the time.

Example: *"CPU increased by 2.1%, 95% CI [0.5%, 3.7%]"*

This tells you:
- Our best estimate is +2.1% CPU
- We're fairly confident the true change is somewhere between +0.5% and +3.7%
- Since the entire interval is above zero, we're confident the change is real
  (the interval doesn't include "no difference")

**Key insight**: If the CI includes zero, the difference is not statistically
significant. If it doesn't include zero, it is significant. CIs and p-values
always agree at the same confidence level.

CIs are more informative than p-values because they tell you about the
*magnitude* and *direction* of the effect, not just whether it exists.

### Effect Size: "Is This Difference Meaningful?"

A difference can be **statistically significant** but **practically tiny**.
With enough reps, even a 0.01% CPU difference becomes statistically detectable
-- but you probably don't care about it.

We use **Cohen's d** to measure effect size in a standardized way. It answers:
"How large is the difference relative to the natural variability?"

| Cohen's d | Label      | Meaning                                       |
|-----------|------------|-----------------------------------------------|
| < 0.2     | negligible | Difference is smaller than the normal noise    |
| 0.2 - 0.5 | small     | Detectable but modest                          |
| 0.5 - 0.8 | medium    | Clearly visible in the data                    |
| > 0.8     | large      | Obvious, dominant effect                       |

The formula is:

```
Cohen's d = (mean_A - mean_B) / pooled_standard_deviation
```

It divides the raw difference by the typical spread. A difference of 2% CPU
means very different things depending on whether your measurements normally
fluctuate by 0.5% (d ≈ 4.0, huge) or by 10% (d ≈ 0.2, negligible).

### Statistical Power: "Can I Even Detect This?"

**Power** is the probability that your test will detect a real difference *if
one exists*. It answers: "If my code change truly affects CPU by X%, what's the
chance my experiment will catch it?"

- **Power = 0.80** (80%) is the standard target. It means: if the effect is
  real, you have an 80% chance of detecting it and a 20% chance of missing it.
- **Power = 0.50** means it's a coin flip -- you're as likely to miss the
  effect as to find it.

Power depends on three things:
1. **Effect size** -- larger effects are easier to detect
2. **Noise level** (standard deviation) -- less noisy data makes detection
   easier
3. **Sample size** (number of reps) -- more reps give more power

This is why the harness has a `recommend` command: it uses your baseline noise
to calculate how many reps you need to achieve 80% power for detecting a given
minimum effect size.

## How a Test Runs

### Step 1: Collect Baseline Data

Deploy the *current* code (before your change). Run N repetitions, each M
seconds long. For each rep, collect 1 Hz snapshots of all KPIs (CPU, FPS,
latency, etc.).

Each rep produces a single mean value per KPI. With 10 reps, you get 10
numbers per KPI.

### Step 2: Collect Iteration Data

Apply your code change. Deploy the new build. Run the same N reps of M
seconds. Same KPIs, same conditions.

### Step 3: Compare With a t-test

For each KPI, we now have two groups of numbers:
- **Baseline**: [47.2, 48.1, 47.8, 47.5, ...]
- **Iteration**: [49.1, 50.3, 48.9, 49.7, ...]

We run **Welch's t-test** to determine whether these groups are statistically
different. The test accounts for the possibility that the two groups might have
different amounts of variability.

The test produces:
- A **p-value** (how surprising is this difference if there's no real effect?)
- A **confidence interval** on the difference (what's the plausible range of
  the true change?)
- A **power** estimate (how likely were we to catch this if it was real?)

### Step 4: Render a Verdict

For each KPI, the harness combines three pieces of information to decide:

1. **Delta percentage**: How much did the metric change?
   `delta_pct = (mean_iteration - mean_baseline) / |mean_baseline| × 100`

2. **Threshold**: Does the change exceed the KPI's regression threshold?
   Each KPI has a threshold (e.g., CPU +10%, FPS -5%) that defines what change
   is "worth caring about."

3. **Significance**: Is p < 0.05?

The verdict logic:

| Change direction | Exceeds threshold? | Significant? | Verdict              |
|------------------|--------------------|--------------|----------------------|
| Worse            | Yes                | Yes          | **regression**       |
| Worse            | Yes                | No           | likely_regression    |
| Better           | Yes                | Yes          | **improvement**      |
| Better           | Yes                | No           | likely_improvement   |
| Either           | No                 | Either       | neutral              |

"Worse" and "better" depend on the KPI. For throughput (FPS), higher is better.
For CPU usage, lower is better.

## Interleaved Collection: Fighting Time-based Noise

Running all baseline reps first and then all iteration reps has a problem: if
the temperature slowly rises over time (or the Pi's thermal throttling kicks
in, or a background process starts), you can't tell whether the difference is
from your code or from the changing environment.

The harness solves this with **interleaved collection**. Instead of:

```
B B B B B B B B B B   then   A A A A A A A A A A
```

It interleaves them:

```
B A A B B A A B B A A B B A A B B A A B
```

This is called **ABBA ordering** (for small N ≤ 10) or **randomized balanced**
ordering (for larger N). The idea is simple: if something external changes
during the experiment, both groups are equally affected.

The interleaving requires swapping binaries between reps. The harness caches
both binaries and uses `docker cp` to swap them quickly (seconds, not minutes).

## Adaptive Stopping: Don't Wait If You Already Know

Sometimes the answer is obvious early. If your code change doubles CPU usage,
you don't need 300 reps to know it's a regression -- 10 reps would suffice. But
if you always run the full 300, you waste hours.

**Adaptive stopping** (also called sequential testing) lets you peek at the
data as it comes in and stop early if the answer is already clear. This can
save enormous amounts of time for overnight tests.

### The Multiple Peeking Problem

Here's the catch: you can't just run a regular t-test after every rep and stop
as soon as p < 0.05. If you check 50 times, random noise will eventually look
significant -- even when there's no real effect. This is called the **multiple
comparisons problem** or "alpha inflation."

Think of it like this: if you flip a fair coin, the chance of getting 5 heads in
a row is only about 3%. But if you flip a coin 100 times and look at *every*
window of 5 consecutive flips, you'll almost certainly find one streak of 5
heads somewhere. You didn't discover an unfair coin; you just looked at too many
windows.

Repeatedly checking your experiment has the same effect: your 5% false alarm
rate silently creeps up to 15%, 25%, or even higher, depending on how often
you peek.

The harness offers two methods that solve this mathematically, each in a
different way.

### Method 1: Group Sequential Testing (GST)

**Library**: spotify-confidence (Spotify's production A/B platform, Apache-2.0)

**Idea**: Plan your peeks in advance. At each pre-scheduled check, use a
*stricter* p-value threshold to compensate for having peeked before. The more
you've peeked, the stricter the requirement.

**How it works**:

1. Before the experiment starts, compute a schedule of "look points" (e.g.,
   check at rep 5, 10, 15, 20, ...).
2. At each look point, compute a p-value threshold (called a "boundary"). The
   early boundaries are very strict (hard to reject), and later boundaries
   relax toward the usual 0.05. This is called **alpha-spending** -- you're
   "spending" your 5% error budget gradually across checks.
3. At each look, if p-value < boundary → stop (efficacy). Otherwise, continue.

**The spending function**: The harness uses a **power family with exponent 3**
(cubic spending), which behaves similarly to O'Brien-Fleming boundaries. Early
looks have extremely strict thresholds (nearly impossible to stop), while later
looks are close to the standard 0.05. This means you sacrifice very little
power compared to a fixed-sample test.

Example boundary schedule for 100 max reps, checking every 10:

| Reps per side | Boundary (p-threshold) | Meaning                          |
|---------------|------------------------|----------------------------------|
| 10            | 0.000005               | Need overwhelming evidence early |
| 20            | 0.000370               | Still very strict                |
| 30            | 0.003100               | Getting more reasonable          |
| 50            | 0.016000               | Moderately strict                |
| 80            | 0.036000               | Close to standard                |
| 100           | 0.048000               | Nearly the full alpha            |

The magic: even though you've peeked many times, the overall false alarm rate
stays at exactly 5%. The math guarantees it.

**Tradeoff**: GST only checks at scheduled points. If you schedule checks every
10 reps, you might have a definitive result at rep 12 but won't stop until rep
20.

### Method 2: Mixture Sequential Probability Ratio Test (mSPRT)

**Library**: In-repo implementation under `libs/msprt/` (MIT licensed, based on
Johari, Pekelis & Walsh 2017/2022)

**Idea**: Use a likelihood-ratio test with a Gaussian mixture prior that allows
you to check after *every single pair* with zero penalty. No schedule needed,
no alpha inflation. Ever.

**How it works**:

1. Before starting, compute a mixing parameter (tau) that encodes how large an
   effect you expect to detect, calibrated to your significance level and
   maximum sample size.
2. After each completed A/B pair, compute the likelihood ratio Lambda_n. This
   compares "how likely is the observed data under the alternative (there is a
   difference)" vs "how likely under the null (no difference)."
3. If Lambda_n exceeds 1/alpha (e.g., 20 for alpha=0.05), we stop: the
   evidence for a real difference is overwhelming.

The key insight: unlike a fixed-sample t-test (which is only valid at a
predetermined sample size), the mSPRT's rejection threshold is *always valid*.
You can check after every single observation pair without inflating the false
positive rate. This comes at the cost of needing slightly more data than a
fixed-sample test to reach the same conclusion.

**Technical detail**: The implementation follows the R `mixtureSPRT` package by
Erik Stenberg (referenced by the original authors). The mixing variance tau^2
is computed from the significance level, population standard deviation, and
maximum sample size. The statistic Lambda_n at each observation n is a product
of a shrinkage term and an exponential of the squared cumulative mean
difference.

**Tradeoff**: Slightly less powerful than GST (needs a few more reps on
average), but maximally flexible: you can check every single rep.

### Futility Stopping: Knowing When to Give Up

Both methods also support **futility stopping**. If, partway through, the data
strongly suggests there's *no* meaningful difference, there's no point in
continuing.

The harness uses a pragmatic heuristic (not a formal statistical boundary):

- Before 30% of the budget is used: never stop for futility (too early to tell)
- After 30%: if the p-value is very high (much larger than 0.05), the data
  says "there's no signal here, and gathering more data is unlikely to change
  that"
- The threshold linearly tightens from 0.8 to 0.5 as the experiment progresses

This is intentionally conservative -- it's better to continue a pointless
experiment for a while than to miss a real effect.

### Which Method Should I Use?

| Criterion                     | GST (spotify-confidence) | mSPRT                   |
|-------------------------------|--------------------------|-------------------------|
| Checks every trial?           | No, only at scheduled points | Yes, every pair      |
| Statistical power             | Slightly higher          | Slightly lower          |
| Flexibility                   | Must plan schedule ahead | Fully flexible          |
| When to stop if clear result  | At next scheduled look   | Immediately             |
| Mathematical rigor            | Very high (exact bounds) | Very high (anytime-valid) |
| Best for                      | Overnight / batch runs with known budget | Quick interactive `iterate` runs |

**Defaults**:
- **Overnight / batch** (`overnight`, `batch` commands): GST (higher power when
  the maximum sample size is known ahead of time -- our case with `--repetitions`).
- **Interactive iterate** (`iterate` command): mSPRT (check every pair, stop
  immediately when significant -- most practical for quick exploratory testing).

## Reducing Noise: Isolation and Hardening

Statistical methods can only work well when the noise is **random** (not
systematic). The harness takes aggressive steps to reduce and randomize noise
on the Raspberry Pi:

1. **Isolated container**: Stop all other containers (BlueOS services,
   extensions). MCM runs alone, so nothing competes for CPU/memory.

2. **CPU governor pinning**: Lock all CPU cores to maximum frequency
   (`performance` governor). This eliminates frequency-scaling jitter where
   the CPU changes speed mid-experiment.

3. **Memory clearing**: Flush page cache, dentries/inodes, and swap before
   each experiment. Every run starts from the same memory state.

4. **Thermal warmup**: Stress all CPU cores until temperature stabilizes
   (< 0.5 C change over 10 seconds), then wait 5 seconds. This eliminates the
   "cold start" effect where the first few reps run cooler (and faster) than
   later ones.

5. **Interleaved execution**: As described above, interleaving baseline and
   iteration reps ensures that any remaining time-varying noise affects both
   groups equally.

These steps dramatically reduce the standard deviation of measurements, which
means you need fewer reps to detect the same effect size.

## Sample Size: "How Many Reps Do I Need?"

This is the most practical question. The answer depends on:

1. **How noisy is the metric?** (standard deviation from your baseline)
2. **How small an effect do you want to detect?** (the Minimum Detectable
   Effect, or MDE)
3. **How much risk of missing it can you accept?** (power, default 80%)

The formula used:

```
n = 2 × ((z_α + z_β) / (MDE / σ))²
```

Where:
- `z_α` ≈ 1.96 (for 5% significance, two-sided)
- `z_β` ≈ 0.84 (for 80% power)
- `MDE` = the minimum absolute change you want to detect
- `σ` = standard deviation from your baseline data

**Quick rules of thumb** (assuming 80% power, 5% significance):

| Coefficient of Variation (CV = std/mean) | Reps to detect 5% change | Reps to detect 10% change |
|-----------------------------------------|--------------------------|---------------------------|
| 1% (very stable)                         | 2                        | 2                         |
| 5% (moderate)                            | 16                       | 4                         |
| 10% (noisy)                              | 64                       | 16                        |
| 20% (very noisy)                         | 252                      | 64                        |

Use the `recommend` command to get exact numbers based on your actual baseline:

```bash
python -m ab_harness recommend --experiment my_test --min-effect 2 5 10
```

## Reading the Report

The HTML report contains a lot of information. Here's what to focus on.

### The Verdict System

| Icon   | Verdict            | Meaning                                        |
|--------|--------------------|------------------------------------------------|
| Red    | regression         | Statistically significant worsening beyond threshold |
| Orange | likely_regression  | Worsening beyond threshold, but not yet significant (might need more reps) |
| Green  | improvement        | Statistically significant improvement beyond threshold |
| Blue   | likely_improvement | Improvement beyond threshold, but not yet significant |
| Gray   | neutral            | Change is within the noise threshold             |

### Key Numbers to Look At

For each KPI in the comparison table:

- **Delta %**: How much the metric changed (positive = increased). The sign
  depends on the KPI -- for CPU, an increase is bad; for FPS, an increase is
  good.

- **p-value**: The smaller, the more confident we are that the change is real.
  Below 0.05 = significant (highlighted).

- **95% CI**: The range the true change likely falls in. If it doesn't include
  zero, the difference is significant.

- **Cohen's d**: How practically large the effect is, relative to noise.
  "negligible" means don't worry about it even if p is small.

- **Power**: If the test found "no difference," was it actually capable of
  detecting one? Power > 0.80 means the non-result is trustworthy. Power <
  0.50 means "we probably can't see small effects -- run more reps."

## Common Pitfalls

**"The p-value is 0.06, so there's no effect."**
No. A p-value of 0.06 means the evidence is suggestive but doesn't meet the
conventional 5% threshold. It doesn't mean the effect is zero. Look at the
confidence interval and effect size -- they tell a richer story.

**"We ran 3 reps and got p = 0.04, so this is definitely a regression."**
Be cautious. With very few reps, the t-test has low power and the estimates are
unreliable. A barely-significant result with 3 reps can easily flip with more
data. Treat it as a flag to investigate, not a final verdict.

**"The delta is 0.1% CPU but p = 0.001, so this is a serious regression."**
Statistical significance doesn't mean practical significance. A 0.1% CPU
change is likely irrelevant even if it's "real." This is why the harness checks
both significance AND the threshold before flagging a regression.

**"I ran baseline on Monday and iteration on Friday."**
This is dangerous. Ambient temperature, network conditions, and the Pi's state
can change between days. Use interleaved collection (`--adaptive`) or at least
minimize the gap between baseline and iteration.

**"I checked the data after every rep and stopped when p < 0.05."**
This invalidates the p-value due to the multiple peeking problem. Use
`--adaptive` which applies proper sequential testing methods. Never peek and
stop without the math to support it.

## Multiple Comparisons Across KPIs

The harness tests 20 KPIs simultaneously, each at alpha = 0.05. Without any
correction, the family-wise error rate (FWER) -- the probability of *at least
one* false positive across all KPIs -- is approximately:

```
FWER = 1 - (1 - 0.05)^20 ≈ 64%
```

This means that even when there is no real effect on any KPI, you have a ~64%
chance of seeing at least one "statistically significant" result by chance
alone.

### Why we don't correct by default

The harness deliberately does **not** apply a multiple comparisons correction in
its default mode. This is a conscious design choice for two reasons:

1. **Regression thresholds act as a practical significance filter.** A KPI is
   only flagged as a regression if the change *both* exceeds its per-KPI
   threshold (e.g., CPU +10%, FPS -5%) *and* is statistically significant. This
   dual requirement dramatically reduces false alarms. A spurious p < 0.05 on
   a 0.1% CPU change will still be classified as "neutral."

2. **We prefer sensitivity over FWER control for interactive testing.** When
   engineers iterate quickly (3-10 reps), false negatives (missing a real
   regression) are more costly than false positives (which are caught by
   threshold filtering). Bonferroni-style corrections reduce power, increasing
   the chance of missing real effects.

### Strict mode: `--strict`

For experiments with many reps (where even tiny effects become significant), the
harness offers a `--strict` flag that applies **Holm-Bonferroni step-down
correction** across all KPIs:

```bash
python -m ab_harness iteration --experiment my_test --strict
python -m ab_harness compare --run-a path/a --run-b path/b --strict
```

Holm-Bonferroni controls the FWER at the nominal alpha (0.05) while being less
conservative than the classic Bonferroni correction. It works by:

1. Sorting all KPI p-values from smallest to largest.
2. Comparing the smallest p-value against alpha / N (where N = number of KPIs).
3. If it rejects, comparing the next against alpha / (N-1), and so on.
4. Stopping at the first non-rejection.

In strict mode, both the raw and adjusted p-values are reported. The
`adjusted_p_value` field in the comparison results reflects the corrected value.

### When to use `--strict`

- Overnight / batch runs with many reps (> 30 per side)
- Regulatory or release-gate comparisons where false positives are costly
- Any time you want formal FWER control

## Serial Correlation Diagnostic

The t-test assumes that per-rep measurements are independent. In interleaved
mode with ABBA ordering, reps are temporally adjacent and share the same system
state. The 5-second warmup per rep partially resets the state, but subtle
serial correlation may emerge in very long experiments (hundreds of reps).

The harness automatically checks for serial correlation using the **Durbin-Watson
statistic** whenever a side has 20 or more reps. The DW statistic ranges from
0 to 4:

- **DW ≈ 2.0**: No serial correlation (ideal).
- **DW < 1.5**: Positive autocorrelation (adjacent reps are correlated).
- **DW > 2.5**: Negative autocorrelation (adjacent reps alternate).

When moderate autocorrelation is detected (DW < 1.5 or DW > 2.5), the harness
logs a warning but does **not** change the verdict. The warning is diagnostic
only -- it flags that the independence assumption may be mildly violated, which
could inflate the t-test's confidence.

## Robust Variance Estimation (mSPRT)

The mSPRT's sigma (standard deviation) parameter controls the mixing variance
used for sequential testing. By default, sigma is re-estimated from the
collected data once enough pairs are available.

For KPIs with heavy-tailed distributions (e.g., `max_freeze_ms`, event counts),
the classical sample variance can be unstable early in the experiment, leading
to unreliable sigma estimates.

The harness uses a **MAD-based robust estimator** instead:

```
sigma_robust = 1.4826 × median(|x - median(x)|)
```

The constant 1.4826 makes the Median Absolute Deviation (MAD) consistent with
the standard deviation for normal distributions, while being far more resistant
to outliers. The pooled estimate combines both sides:

```
sigma_pooled = sqrt((sigma_robust_A² + sigma_robust_B²) / 2)
```

If MAD returns zero (e.g., all-equal data), the estimator falls back to the
classical pooled standard deviation.

The minimum number of pairs before testing begins (`min_n`) defaults to 8
(increased from 5) to further stabilize early variance estimates.

## Robust Comparison Statistics

The t-test and Cohen's d assume that per-rep means are approximately
normally distributed.  For most KPIs (CPU, FPS, latency) this holds well
thanks to the Central Limit Theorem operating on the within-rep 1 Hz
samples.  However, some KPIs have distributions that resist CLT
normalization at typical sample sizes:

- **`system_load_1m`**: High coefficient of variation (~55%)
- **`max_freeze_ms`**, **`total_stutter_events`**, **`total_freeze_events`**:
  Zero-inflated and heavy-tailed
- **`max_stutter_ratio`**: Bounded and often near zero

For these KPIs, the harness automatically detects normality violations and
switches to distribution-free methods.

### Normality Diagnostic: Shapiro-Wilk

Before comparing each KPI, the harness runs the **Shapiro-Wilk test** on
each side's per-rep means (when 4 <= N <= 50).  If either side rejects
normality at the experiment's alpha level (default 0.05), a
`normality_warning` flag is set and the comparison pipeline switches to
non-parametric methods for that KPI.

### Non-parametric Test: Mann-Whitney U

When normality is violated, the **Mann-Whitney U test**
(`scipy.stats.mannwhitneyu`, two-sided) replaces Welch's t-test for
verdict determination.  This test makes no distributional assumptions --
it tests whether one group's values tend to be larger than the other's.
The p-value from Mann-Whitney drives the verdict logic (significance check
and threshold comparison).

For well-behaved KPIs (normality not rejected), Welch's t-test remains
the default because it has higher statistical power under normality.

The `test_used` field in each KPI comparison indicates which test drove
the verdict: `"welch"` or `"mann_whitney"`.

### Non-parametric Effect Size: Cliff's Delta

Alongside Cohen's d, the harness computes **Cliff's delta** for every KPI.
Cliff's delta measures the probability that a randomly chosen observation
from group A exceeds a randomly chosen observation from group B, minus
the reverse probability:

```
Cliff's delta = P(a > b) - P(a < b)
```

The value ranges from -1 to +1.  Classification (Romano et al. 2006):

| |Cliff's delta| | Label      |
|-----------------|------------|
| < 0.147         | negligible |
| 0.147 - 0.33    | small      |
| 0.33 - 0.474    | medium     |
| >= 0.474        | large      |

When `normality_warning` is set, the report displays Cliff's delta
instead of Cohen's d for the effect size label, since it does not assume
normality.

### Bootstrap Confidence Intervals

When normality is violated, the parametric CI from the t-test may be
inaccurate.  The harness computes a **BCa (bias-corrected and
accelerated) bootstrap CI** using 9,999 resamples.  This provides a
distribution-free estimate of the plausible range for the true difference
in means.  Bootstrap CIs are only computed for KPIs with a normality
warning, to avoid unnecessary computation.

### Outlier Diagnostics: IQR Fences

For each KPI, the harness counts per-rep means that fall outside
**Tukey's fences** (1.5 x IQR below Q1 or above Q3).  These are
reported as `outlier_count_a` and `outlier_count_b` in the comparison
output.

Outliers are **never removed** -- they are flagged for human review.
A high outlier count suggests that the measurement process has an
instability (e.g., thermal throttling spike, transient OS load) that
may warrant investigation or re-running affected reps.

## Reproducible Trial Ordering

When interleaved execution uses randomized balanced ordering (N > 10 reps),
the trial sequence is randomized using Python's `random.Random`. By default,
a random seed is generated automatically and logged for post-hoc reproducibility.

To make an experiment exactly reproducible, pass `--seed`:

```bash
python -m ab_harness iteration --experiment my_test --repetitions 50 --seed 12345
```

The seed (whether user-provided or auto-generated) is saved in:
- The experiment metadata (`metadata.json` field `randomization_seed`)
- The trial sequence file (`trial_sequence.json` field `seed`)

This allows re-running an experiment with the exact same trial ordering to
verify results or investigate anomalies.

## Glossary

| Term | Definition |
|------|-----------|
| **Alpha (α)** | The false alarm rate. Probability of claiming a difference exists when it doesn't. Default: 0.05 (5%). |
| **Baseline (B)** | The control group. The current code, before your change. |
| **Cohen's d** | Standardized effect size. The difference in means divided by the pooled standard deviation. |
| **Confidence Interval (CI)** | A range of values that likely contains the true parameter. A 95% CI means that if you repeated the experiment many times, 95% of the resulting intervals would contain the truth. |
| **Effect Size** | How large a difference is, in practical terms. Cohen's d is one measure. |
| **Futility** | Stopping an experiment early because the data suggests there is no meaningful effect. |
| **Hypothesis Testing** | A framework for deciding whether observed data is consistent with "no effect" (null hypothesis) or evidence of a real change. |
| **Information Fraction** | In sequential testing, how much of your planned data you've collected so far. t = current_reps / max_reps. |
| **Interleaved** | Alternating between baseline and iteration reps to control for time-varying noise. |
| **Iteration (A)** | The treatment group. Your modified code. |
| **KPI** | Key Performance Indicator. A metric you're tracking (e.g., CPU %, FPS, latency). |
| **MDE** | Minimum Detectable Effect. The smallest change you want your experiment to be able to detect. |
| **Null Hypothesis (H0)** | The assumption that there is no real difference between baseline and iteration. |
| **P-value** | The probability of observing data at least as extreme as what you got, assuming the null hypothesis is true. |
| **Power** | The probability of detecting a real effect if it exists. Target: 0.80 (80%). |
| **Regression** | A statistically significant worsening of a KPI beyond its threshold. |
| **Rep (Repetition)** | One independent measurement window (e.g., 60 seconds of data collection). |
| **Sequential Testing** | Methods that allow checking results during data collection while controlling the false alarm rate. |
| **Standard Deviation (std)** | A measure of how spread out measurements are. Low std = consistent; high std = noisy. |
| **Welch's t-test** | A statistical test comparing two group means that doesn't assume equal variances. The standard choice for A/B tests. |

## Further Reading

- **Spotify's engineering blog on sequential testing**:
  [Choosing Sequential Testing Framework](https://engineering.atspotify.com/2023/03/choosing-sequential-testing-framework-comparisons-and-discussions/)
  -- The blog post that motivated our GST implementation.

- **Johari, Pekelis & Walsh (2017), "Peeking at A/B Tests: Why it matters,
  and what to do about it"** -- The paper behind our mSPRT implementation.
  See also Johari et al. (2022), "Always Valid Inference", Operations Research.

- **Evan Miller, "How Not To Run an A/B Test"**:
  [evanmiller.org/how-not-to-run-an-ab-test.html](https://www.evanmiller.org/how-not-to-run-an-ab-test.html)
  -- A classic, accessible explanation of the multiple peeking problem.

- **Wikipedia: Welch's t-test**:
  [en.wikipedia.org/wiki/Welch%27s_t-test](https://en.wikipedia.org/wiki/Welch%27s_t-test)
  -- Reference on the main test used by the harness.
