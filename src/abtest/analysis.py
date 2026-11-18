"""Run a full experiment analysis from a spec and user-level data, and decide.

Order of operations, and why:

1. **SRM first.** If the arm sizes do not match the configured split, the assignment or logging
   is broken and no metric can be trusted, so the verdict is ``invalid`` whatever else happens.
2. **Primary metric** at the full alpha (it is the pre-registered decision metric).
3. **Secondary metrics** corrected for multiple testing as one family (Holm by default).
4. **Guardrails** as non-inferiority checks against a relative margin.
5. **Decision** from the above, plus the minimum detectable effect so that "no significant
   change" is reported together with how large a change the test could have seen.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Literal

import numpy as np
import pandas as pd
from scipy import stats

from abtest.bayes import BayesResult, beta_binomial, normal_posterior
from abtest.bootstrap import BootstrapResult, bootstrap_difference, mean_statistic, ratio_statistic
from abtest.cuped import CupedResult, cuped_test
from abtest.frequentist import (
    Comparison,
    RankComparison,
    mann_whitney,
    two_proportion_ztest,
    welch_ttest,
    z_quantile,
)
from abtest.multiple import adjust_pvalues
from abtest.ratio import delta_ratio_test
from abtest.sequential import SequentialSummary, msprt_monitor
from abtest.spec import ExperimentSpec, GuardrailSpec, MetricSpec
from abtest.srm import SRMResult, srm_check

Verdict = Literal["ship", "do not ship", "inconclusive", "invalid"]
GuardrailStatus = Literal["pass", "fail", "inconclusive"]
Outcome = Literal["better", "worse", "no detectable change"]


@dataclass(frozen=True)
class MetricResult:
    spec: MetricSpec
    comparison: Comparison
    p_adjusted: float
    bayes: BayesResult
    cuped: CupedResult | None = None
    rank: RankComparison | None = None
    bootstrap: BootstrapResult | None = None

    def outcome(self, alpha: float) -> Outcome:
        if self.p_adjusted >= alpha:
            return "no detectable change"
        up = self.comparison.effect > 0
        return "better" if up == (self.spec.direction == "increase") else "worse"


@dataclass(frozen=True)
class GuardrailResult:
    spec: GuardrailSpec
    comparison: Comparison
    degradation: float
    degradation_lower: float
    degradation_upper: float
    status: GuardrailStatus


@dataclass(frozen=True)
class Decision:
    verdict: Verdict
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class ExperimentReport:
    spec: ExperimentSpec
    n_control: int
    n_treatment: int
    srm: SRMResult
    primary: MetricResult
    secondary: tuple[MetricResult, ...]
    guardrails: tuple[GuardrailResult, ...]
    sequential: SequentialSummary | None
    primary_mde_relative: float
    decision: Decision


# ------------------------------------------------------------------------------- metrics


def compare_metric(
    metric: MetricSpec, control: pd.DataFrame, treatment: pd.DataFrame, alpha: float
) -> Comparison:
    """The plain (unadjusted) comparison appropriate for the metric kind."""
    if metric.kind == "binary":
        col = metric.column
        return two_proportion_ztest(
            int(control[col].sum()), len(control), int(treatment[col].sum()), len(treatment), alpha
        )
    if metric.kind == "continuous":
        return welch_ttest(control[metric.column], treatment[metric.column], alpha)
    return delta_ratio_test(
        control[metric.numerator],
        control[metric.denominator],
        treatment[metric.numerator],
        treatment[metric.denominator],
        alpha,
    )


def analyze_metric(
    metric: MetricSpec, control: pd.DataFrame, treatment: pd.DataFrame, spec: ExperimentSpec
) -> MetricResult:
    alpha = spec.alpha
    cuped = None
    if metric.covariate:
        cuped = cuped_test(
            control[metric.column],
            control[metric.covariate],
            treatment[metric.column],
            treatment[metric.covariate],
            alpha,
        )
        comparison = cuped.adjusted
    else:
        comparison = compare_metric(metric, control, treatment, alpha)

    rank = None
    if metric.robust and metric.kind == "continuous":
        rank = mann_whitney(control[metric.column], treatment[metric.column])

    boot = None
    if metric.bootstrap:
        if metric.kind == "ratio":
            cols, stat = [metric.numerator, metric.denominator], ratio_statistic
        else:
            cols, stat = [metric.column], mean_statistic
        boot = bootstrap_difference(
            [control[c].to_numpy() for c in cols],
            [treatment[c].to_numpy() for c in cols],
            stat,
            n_resamples=spec.bootstrap_resamples,
            alpha=alpha,
            seed=spec.seed,
        )

    if metric.kind == "binary":
        col = metric.column
        bayes = beta_binomial(
            int(control[col].sum()),
            len(control),
            int(treatment[col].sum()),
            len(treatment),
            credible=1 - alpha,
            seed=spec.seed,
        )
    else:
        bayes = normal_posterior(comparison.effect, comparison.se, credible=1 - alpha)
    if metric.direction == "decrease":
        bayes = bayes.lower_is_better()

    return MetricResult(
        spec=metric,
        comparison=comparison,
        p_adjusted=comparison.p_value,
        bayes=bayes,
        cuped=cuped,
        rank=rank,
        bootstrap=boot,
    )


def check_guardrail(
    guardrail: GuardrailSpec, control: pd.DataFrame, treatment: pd.DataFrame, alpha: float
) -> GuardrailResult:
    """Non-inferiority check on the relative change, one-sided at ``alpha``.

    pass         - the upper confidence bound of the degradation is below the margin;
    fail         - a degradation is statistically significant and could exceed the margin;
    inconclusive - neither: the test cannot rule out a degradation beyond the margin.
    """
    comparison = compare_metric(guardrail.metric, control, treatment, alpha)
    sign = 1.0 if guardrail.metric.direction == "decrease" else -1.0
    degradation = sign * comparison.relative_effect
    half = float(stats.norm.ppf(1 - alpha)) * comparison.relative_se
    lower, upper = degradation - half, degradation + half
    status: GuardrailStatus
    if upper < guardrail.margin:
        status = "pass"
    elif lower > 0:
        status = "fail"
    else:
        status = "inconclusive"
    return GuardrailResult(guardrail, comparison, degradation, lower, upper, status)


def sequential_primary(data: pd.DataFrame, spec: ExperimentSpec) -> SequentialSummary | None:
    """mSPRT over cumulative looks (one per distinct value of ``look_column``)."""
    if not spec.look_column:
        return None
    effects: list[float] = []
    variances: list[float] = []
    tau = math.nan
    for value in np.sort(data[spec.look_column].unique()):
        upto = data[data[spec.look_column] <= value]
        c, t = _split(upto, spec)
        if min(len(c), len(t)) < 2:
            continue
        comp = compare_metric(spec.primary, c, t, spec.alpha)
        if not comp.se > 0:
            continue
        if math.isnan(tau):
            tau = spec.msprt_tau_relative * abs(comp.control_mean) or comp.se
        effects.append(comp.effect)
        variances.append(comp.se**2)
    if not effects:
        return None
    return msprt_monitor(effects, variances, tau, spec.alpha)


# ------------------------------------------------------------------------------ decision


def _fmt_rel(x: float) -> str:
    return f"{x:+.2%}"


def decide(
    srm: SRMResult,
    primary: MetricResult,
    guardrails: tuple[GuardrailResult, ...],
    alpha: float,
    mde_relative: float,
) -> Decision:
    name = primary.spec.name
    if srm.mismatch:
        shares = ", ".join(f"{s:.2%}" for s in srm.observed_shares)
        return Decision(
            "invalid",
            (
                f"sample ratio mismatch (chi-square p = {srm.p_value:.2g} < {srm.threshold:g}); "
                f"observed shares {shares}. Find the cause before reading any metric.",
            ),
        )
    reasons: list[str] = []
    failed = [g for g in guardrails if g.status == "fail"]
    unclear = [g for g in guardrails if g.status == "inconclusive"]
    for g in failed:
        reasons.append(
            f"guardrail {g.spec.metric.name} degraded by {_fmt_rel(g.degradation)} "
            f"(margin {g.spec.margin:.1%})"
        )
    outcome = primary.outcome(alpha)
    rel = primary.comparison.relative_effect
    if failed:
        return Decision("do not ship", tuple(reasons))
    if outcome == "worse":
        return Decision("do not ship", (f"{name} got worse: {_fmt_rel(rel)}",))
    if outcome == "better":
        reasons.append(f"{name} improved: {_fmt_rel(rel)} (p = {primary.p_adjusted:.2g})")
        if unclear:
            names = ", ".join(g.spec.metric.name for g in unclear)
            reasons.append(
                f"guardrail(s) {names} cannot rule out a degradation beyond the margin; "
                "extend the test or accept that risk explicitly"
            )
            return Decision("inconclusive", tuple(reasons))
        return Decision("ship", tuple(reasons))
    return Decision(
        "inconclusive",
        (
            f"no detectable change in {name} ({_fmt_rel(rel)}, p = {primary.p_adjusted:.2g}); "
            f"this sample could detect about {mde_relative:.2%} with the configured power",
        ),
    )


def _split(data: pd.DataFrame, spec: ExperimentSpec) -> tuple[pd.DataFrame, pd.DataFrame]:
    arm = data[spec.variant_column]
    return data[arm == spec.control], data[arm == spec.treatment]


def analyze(data: pd.DataFrame, spec: ExperimentSpec) -> ExperimentReport:
    """Analyse validated user-level data (see :func:`abtest.spec.validate_data`)."""
    control, treatment = _split(data, spec)
    if min(len(control), len(treatment)) < 2:
        raise ValueError("each arm needs at least two units")
    srm = srm_check([len(control), len(treatment)], spec.expected_split, spec.srm_threshold)

    primary = analyze_metric(spec.primary, control, treatment, spec)
    secondary = [analyze_metric(m, control, treatment, spec) for m in spec.secondary]
    if secondary:
        adjusted = adjust_pvalues([m.comparison.p_value for m in secondary], spec.correction)
        secondary = [
            replace(m, p_adjusted=float(p)) for m, p in zip(secondary, adjusted, strict=True)
        ]
    guardrails = tuple(
        check_guardrail(g, control, treatment, spec.guardrail_alpha) for g in spec.guardrails
    )
    z_power = float(stats.norm.ppf(spec.power))
    comp = primary.comparison
    mde = (z_quantile(spec.alpha) + z_power) * comp.se / abs(comp.control_mean)
    return ExperimentReport(
        spec=spec,
        n_control=len(control),
        n_treatment=len(treatment),
        srm=srm,
        primary=primary,
        secondary=tuple(secondary),
        guardrails=guardrails,
        sequential=sequential_primary(data, spec),
        primary_mde_relative=mde,
        decision=decide(srm, primary, guardrails, spec.alpha, mde),
    )
