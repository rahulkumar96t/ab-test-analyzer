# Experiment report: checkout-redesign (treatment vs control)

## Decision

Verdict: SHIP

- conversion improved: +7.05% (p = 0.00039)

## Sample

| arm | units | share | expected share |
|---|---|---|---|
| control | 60,185 | 50.15% | 50.00% |
| treatment | 59,815 | 49.85% | 50.00% |

Sample-ratio check: chi-square 1.141, p 0.2855 (threshold 0.001) -> OK

## Primary metric

| metric | control | treatment | difference [CI] | relative [CI] | p | P(better) | exp. loss if shipped | result |
|---|---|---|---|---|---|---|---|---|
| conversion | 8.00% | 8.57% | +0.56 pp [+0.25 pp, +0.88 pp] | +7.05% [+3.02%, +11.08%] | 0.0004 | 100.0% | 7.73e-06 pp | better |

95% confidence intervals, two-sided alpha 0.05. Minimum detectable effect at this sample size (80% power): 5.57% relative.

## Secondary metrics (holm correction)

| metric | control | treatment | difference [CI] | relative [CI] | p | adj. p | P(better) | exp. loss if shipped | result |
|---|---|---|---|---|---|---|---|---|---|
| revenue_per_user (CUPED) | 4.856 | 5.163 | +0.3072 [+0.01929, +0.5951] | +6.33% [+0.20%, +12.45%] | 0.0365 | 0.0730 | 98.2% | 0.000973 | no detectable change |
| revenue_per_session | 1.214 | 1.264 | +0.04942 [-0.02124, +0.1201] | +4.07% [-1.88%, +10.01%] | 0.1705 | 0.1705 | 91.5% | 0.00141 | no detectable change |
| sessions_per_user (CUPED) | 4.006 | 4.079 | +0.07293 [+0.0509, +0.09496] | +1.82% [+1.27%, +2.38%] | <0.0001 | <0.0001 | 100.0% | 7.2e-14 | better |

## Guardrails

| metric | control | treatment | degradation | worst case | margin | status |
|---|---|---|---|---|---|---|
| page_load_ms (lower is better) | 903.4 | 905.6 | +0.23% | +0.58% | 2.0% | PASS |
| bounce_rate (lower is better) | 30.17% | 30.19% | +0.08% | +1.53% | 3.0% | PASS |

Worst case = one-sided 95% upper bound of the relative degradation. PASS: below the margin; FAIL: significant degradation that may exceed it; INCONCLUSIVE: neither can be shown.

## Sequential monitoring (mSPRT)

14 cumulative looks on 'day'; always-valid p 0.0029; first below 0.05 at look 6; confidence sequence for the difference [+0.20 pp, +1.05 pp] (mixture tau 0.00389).

Always-valid p by look: 1.0000, 0.8345, 0.3428, 0.1701, 0.0961, 0.0294, 0.0286, 0.0091, 0.0029, 0.0029, 0.0029, 0.0029, 0.0029, 0.0029

## Method details

- conversion: two-proportion z-test; Bayesian: beta-binomial, Beta(1, 1) prior
- revenue_per_user: Welch t-test on CUPED-adjusted values; Bayesian: normal model, flat prior
- revenue_per_user: CUPED on pre_revenue (theta 0.06998, correlation 0.162) reduced the variance of the estimate by 2.6%; CI width 0.5835 -> 0.5758, raw p 0.0339
- revenue_per_user: Mann-Whitney p 0.0004, P(treatment > control) 0.5028 (tests ordering, not means)
- revenue_per_user: bootstrap (2000 resamples, percentile) of the raw difference +0.3157 [+0.02202, +0.6125]
- revenue_per_session: delta-method z-test (ratio of sums); Bayesian: normal model, flat prior
- revenue_per_session: bootstrap (2000 resamples, percentile) of the raw difference +0.04942 [-0.02143, +0.1212]
- sessions_per_user: Welch t-test on CUPED-adjusted values; Bayesian: normal model, flat prior
- sessions_per_user: CUPED on pre_sessions (theta 0.2409, correlation 0.889) reduced the variance of the estimate by 79.0%; CI width 0.09624 -> 0.04406, raw p 0.0001
