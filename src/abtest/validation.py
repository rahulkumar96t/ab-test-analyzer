"""Monte-Carlo validation: do the tests hold their nominal alpha and reach the planned power?

Each check simulates many experiments with a known truth and counts how often a procedure
rejects (or how often an interval covers the truth). Rates come with a 95 % Wilson interval
for the Monte-Carlo error, so "4.9 % [4.3, 5.6]" means the nominal 5 % is consistent with the
simulation. All randomness flows from one seed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from abtest.bayes import prob_beta_greater
from abtest.bootstrap import bootstrap_difference
from abtest.cuped import cuped_test
from abtest.frequentist import mann_whitney_pvalues, welch_pvalues, welch_ttest, ztest_pvalues
from abtest.power import power_means, sample_size_means, sample_size_proportions
from abtest.ratio import delta_ratio_pvalues
from abtest.sequential import group_sequential_design, msprt_log_lambda
from abtest.simulate import ExperimentConfig, simulate_experiment
from abtest.srm import srm_check

Quantity = Literal["false-positive rate", "power", "coverage", "detection rate"]
FPR: Quantity = "false-positive rate"


@dataclass(frozen=True)
class Rate:
    hits: int
    trials: int

    @property
    def value(self) -> float:
        return self.hits / self.trials

    def wilson(self, z: float = 1.959964) -> tuple[float, float]:
        n, p = self.trials, self.value
        centre = (p + z * z / (2 * n)) / (1 + z * z / n)
        half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
        return max(0.0, centre - half), min(1.0, centre + half)


def rate(flags: NDArray[np.bool_]) -> Rate:
    return Rate(int(np.count_nonzero(flags)), int(flags.size))


@dataclass(frozen=True)
class Check:
    procedure: str
    scenario: str
    quantity: Quantity
    target: float | None
    measured: Rate

    @property
    def consistent(self) -> bool | None:
        """Is the target inside the Monte-Carlo interval? None when there is no target."""
        if self.target is None:
            return None
        lo, hi = self.measured.wilson()
        return lo <= self.target <= hi


@dataclass
class ValidationReport:
    seed: int
    sims: int
    checks: list[Check] = field(default_factory=list)
    extras: dict[str, float] = field(default_factory=dict)

    def add(
        self,
        procedure: str,
        scenario: str,
        quantity: Quantity,
        target: float | None,
        hits: ArrayLike,
    ) -> None:
        flags = np.concatenate([np.ravel(h) for h in hits]) if isinstance(hits, list) else hits
        self.checks.append(Check(procedure, scenario, quantity, target, rate(np.asarray(flags))))


def _chunks(total: int, size: int) -> list[int]:
    return [min(size, total - start) for start in range(0, total, size)]


# --------------------------------------------------------------------------- conversion


def check_ztest(sims: int, rng: np.random.Generator, report: ValidationReport) -> None:
    baseline, lift = 0.10, 0.05
    n = sample_size_proportions(baseline, lift, relative=True).control
    report.extras["ztest_n_per_arm"] = n
    x_c = rng.binomial(n, baseline, sims)
    p_aa = ztest_pvalues(x_c, n, rng.binomial(n, baseline, sims), n)
    p_ab = ztest_pvalues(x_c, n, rng.binomial(n, baseline * (1 + lift), sims), n)
    scen = f"conversion 10 %, n = {n:,}/arm"
    report.add("two-proportion z-test", f"{scen}, A/A", FPR, 0.05, p_aa < 0.05)
    report.add("two-proportion z-test", f"{scen}, +5 % relative", "power", 0.80, p_ab < 0.05)


def check_bayes(sims: int, rng: np.random.Generator, report: ValidationReport) -> None:
    n, p = 5_000, 0.10
    x_c, x_t = rng.binomial(n, p, sims), rng.binomial(n, p, sims)
    prob = [
        prob_beta_greater(1 + t, 1 + n - t, 1 + c, 1 + n - c) for c, t in zip(x_c, x_t, strict=True)
    ]
    scen = f"conversion 10 %, n = {n:,}/arm, A/A (one-sided rule)"
    report.add("beta-binomial, ship if P(better) > 95 %", scen, FPR, 0.05, np.array(prob) > 0.95)


def check_srm(sims: int, rng: np.random.Generator, report: ValidationReport) -> None:
    users = 100_000
    fair = rng.binomial(users, 0.5, sims)
    skewed = rng.binomial(users, 0.505, sims)
    p_fair = np.array([srm_check([users - t, t]).p_value for t in fair])
    p_skew = np.array([srm_check([users - t, t]).p_value for t in skewed])
    name = "SRM chi-square (threshold 0.001)"
    report.add(name, "100,000 users, 50/50", FPR, 0.001, p_fair < 0.001)
    report.add(name, "100,000 users, 50.5/49.5 split", "detection rate", None, p_skew < 0.001)


# ------------------------------------------------------------------------------ revenue


def _revenue(
    rng: np.random.Generator, shape: tuple[int, int], conversion: float, value_mean: float
) -> NDArray[np.float64]:
    """Zero-inflated log-normal revenue per user (sigma 1)."""
    out = np.zeros(shape)
    bought = rng.random(shape) < conversion
    out[bought] = value_mean * rng.lognormal(-0.5, 1.0, int(bought.sum()))
    return out


def _revenue_sd(conversion: float, value_mean: float) -> float:
    """SD of zero-inflated log-normal revenue: E[Y^2] = p * m^2 * e^(sigma^2) with sigma 1."""
    return math.sqrt(conversion * value_mean**2 * math.e - (conversion * value_mean) ** 2)


def check_revenue(sims: int, rng: np.random.Generator, report: ValidationReport) -> None:
    """Sample size planned with the control SD for +15 %; targets are the calculator's power
    using the true treatment SD as well, which is what the simulation should reproduce."""
    conv, value, lift = 0.10, 50.0, 0.15
    sd, effect = _revenue_sd(conv, value), lift * conv * value
    n = sample_size_means(effect, sd).control
    report.extras["revenue_n_per_arm"] = n
    scenarios = {
        "A/A": (conv, value),
        "+15 % via order value": (conv, value * (1 + lift)),
        "+15 % via conversion": (conv * (1 + lift), value),
    }
    for label, (t_conv, t_value) in scenarios.items():
        welch, mw = [], []
        for size in _chunks(sims, 100):
            c = _revenue(rng, (size, n), conv, value)
            t = _revenue(rng, (size, n), t_conv, t_value)
            welch.append(welch_pvalues(c, t))
            mw.append(mann_whitney_pvalues(c, t))
        aa = label == "A/A"
        quantity: Quantity = "false-positive rate" if aa else "power"
        target = 0.05
        if not aa:
            target = power_means(effect, sd, n, sd_treatment=_revenue_sd(t_conv, t_value))
            report.extras[f"revenue_predicted_power_{label.split()[-1]}"] = target
        scen = f"zero-inflated revenue, n = {n:,}/arm (80 % with control SD), {label}"
        report.add("Welch t-test", scen, quantity, target, [p < 0.05 for p in welch])
        report.add("Mann-Whitney U", scen, quantity, 0.05 if aa else None, [p < 0.05 for p in mw])


def check_bootstrap(sims: int, rng: np.random.Generator, report: ValidationReport) -> None:
    n, resamples = 1_000, 1_000
    boot_cover, welch_cover = [], []
    for _ in range(sims):
        c = _revenue(rng, (1, n), 0.10, 50.0)[0]
        t = _revenue(rng, (1, n), 0.10, 50.0)[0]
        seed = int(rng.integers(2**31))
        b = bootstrap_difference([c], [t], n_resamples=resamples, seed=seed)
        boot_cover.append(b.ci_low <= 0.0 <= b.ci_high)
        lo, hi = welch_ttest(c, t).ci
        welch_cover.append(lo <= 0.0 <= hi)
    scen = f"zero-inflated revenue, n = {n:,}/arm, true difference 0"
    report.add(
        f"bootstrap percentile CI ({resamples} resamples)",
        scen,
        "coverage",
        0.95,
        np.array(boot_cover),
    )
    report.add("Welch t CI", scen, "coverage", 0.95, np.array(welch_cover))


# -------------------------------------------------------------------------------- CUPED


def check_cuped(sims: int, rng: np.random.Generator, report: ValidationReport) -> None:
    """CUPED on the simulator's sessions metric (strong covariate) and revenue (weak)."""
    users, lift = 20_000, 0.02
    true_effect = 3.0 * lift  # sessions_mean - 1 = 3 extra sessions, engagement has mean 1
    results: dict[str, list[tuple[float, float, float, float, float]]] = {"aa": [], "ab": []}
    revenue_vr = []
    for label, sessions_lift in (("aa", 0.0), ("ab", lift)):
        for _ in range(sims):
            cfg = ExperimentConfig(
                n_users=users, sessions_lift=sessions_lift, seed=int(rng.integers(2**31))
            )
            df = simulate_experiment(cfg)
            c, t = df[df.variant == "control"], df[df.variant == "treatment"]
            r = cuped_test(c.sessions, c.pre_sessions, t.sessions, t.pre_sessions)
            row = (r.raw.p_value, r.adjusted.p_value, r.raw.effect, r.adjusted.effect)
            results[label].append((*row, r.correlation**2))
            if label == "aa":
                rv = cuped_test(c.revenue, c.pre_revenue, t.revenue, t.pre_revenue)
                revenue_vr.append(rv.variance_reduction)
    aa, ab = np.array(results["aa"]), np.array(results["ab"])
    scen = f"simulator sessions/user, {users:,} users"
    for col, name in ((0, "Welch t-test, raw"), (1, "Welch t-test, CUPED")):
        report.add(name, f"{scen}, A/A", FPR, 0.05, aa[:, col] < 0.05)
    for col, name in ((0, "Welch t-test, raw"), (1, "Welch t-test, CUPED")):
        report.add(name, f"{scen}, +{lift:.0%} sessions", "power", None, ab[:, col] < 0.05)
    report.extras.update(
        {
            "cuped_true_effect": true_effect,
            "cuped_mean_effect_raw": float(ab[:, 2].mean()),
            "cuped_mean_effect_adjusted": float(ab[:, 3].mean()),
            "cuped_sd_effect_raw": float(ab[:, 2].std(ddof=1)),
            "cuped_sd_effect_adjusted": float(ab[:, 3].std(ddof=1)),
            "cuped_empirical_variance_reduction": float(
                1 - ab[:, 3].var(ddof=1) / ab[:, 2].var(ddof=1)
            ),
            "cuped_mean_rho_squared": float(ab[:, 4].mean()),
            "cuped_revenue_mean_variance_reduction": float(np.mean(revenue_vr)),
        }
    )


# ------------------------------------------------------------------------- ratio metric


def check_ratio(sims: int, rng: np.random.Generator, report: ValidationReport) -> None:
    """Click-through rate per page view, users randomised, CTR varies by user."""
    users = 5_000

    def arm(size: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        views = 1 + rng.poisson(4.0 * rng.lognormal(-0.5, 1.0, (size, users)))
        ctr = rng.beta(0.25, 4.75, (size, users))  # mean 5 %, strongly user-specific
        return rng.binomial(views, ctr).astype(float), views.astype(float)

    delta, naive = [], []
    for size in _chunks(sims, 200):
        clicks_c, views_c = arm(size)
        clicks_t, views_t = arm(size)
        delta.append(delta_ratio_pvalues(clicks_c, views_c, clicks_t, views_t))
        naive.append(
            ztest_pvalues(clicks_c.sum(-1), views_c.sum(-1), clicks_t.sum(-1), views_t.sum(-1))
        )
    scen = f"clicks / page views, {users:,} users/arm, A/A"
    report.add("delta method (user-level)", scen, FPR, 0.05, [p < 0.05 for p in delta])
    report.add("naive page-view-level z-test", scen, FPR, 0.05, [p < 0.05 for p in naive])


# ------------------------------------------------------------------------------ peeking


@dataclass(frozen=True)
class PeekingRow:
    looks: int
    lift: float
    naive: Rate
    fixed: Rate
    msprt: Rate
    gst: Rate
    msprt_mean_fraction: float
    gst_mean_fraction: float


def simulate_peeking(
    sims: int,
    looks: int,
    n_per_arm: int = 20_000,
    baseline: float = 0.10,
    lift: float = 0.0,
    alpha: float = 0.05,
    tau_relative: float = 0.10,
    seed: int = 0,
) -> PeekingRow:
    """Conversion experiment analysed after every 1/looks of the planned sample.

    * naive  - stop at the first look with a fixed-horizon p < alpha;
    * fixed  - only the final look (the reference that holds alpha);
    * msprt  - stop when the mSPRT likelihood ratio reaches 1/alpha;
    * gst    - O'Brien-Fleming-type Lan-DeMets boundaries for ``looks`` equal looks.
    """
    if n_per_arm % looks:
        raise ValueError("n_per_arm must be divisible by looks")
    rng = np.random.default_rng(seed)
    step = n_per_arm // looks
    x_c = rng.binomial(step, baseline, (sims, looks)).cumsum(axis=1)
    x_t = rng.binomial(step, baseline * (1 + lift), (sims, looks)).cumsum(axis=1)
    n = step * np.arange(1, looks + 1)
    pooled = (x_c + x_t) / (2 * n)
    var = pooled * (1 - pooled) * 2 / n
    diff = (x_t - x_c) / n
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(var > 0, diff / np.sqrt(np.where(var > 0, var, 1.0)), 0.0)
        log_lr = np.where(
            var > 0,
            msprt_log_lambda(diff, np.where(var > 0, var, 1.0), tau_relative * baseline),
            -np.inf,
        )
    crit = 1.959963984540054
    design = group_sequential_design(n / n_per_arm, alpha, "obf")
    msprt_hit = log_lr >= math.log(1 / alpha)
    gst_hit = np.abs(z) >= np.asarray(design.boundaries)

    def mean_fraction(hit: NDArray[np.bool_]) -> float:
        first = np.where(hit.any(axis=1), hit.argmax(axis=1) + 1, looks)
        return float(first.mean() / looks)

    return PeekingRow(
        looks=looks,
        lift=lift,
        naive=rate((np.abs(z) >= crit).any(axis=1)),
        fixed=rate(np.abs(z[:, -1]) >= crit),
        msprt=rate(msprt_hit.any(axis=1)),
        gst=rate(gst_hit.any(axis=1)),
        msprt_mean_fraction=mean_fraction(msprt_hit),
        gst_mean_fraction=mean_fraction(gst_hit),
    )


# ---------------------------------------------------------------------------------- run

CHECKS = {
    "ztest": (check_ztest, 10),
    "srm": (check_srm, 10),
    "bayes": (check_bayes, 1),
    "revenue": (check_revenue, 1),
    "ratio": (check_ratio, 1),
    "cuped": (check_cuped, 1),
    "bootstrap": (check_bootstrap, 0.25),
}


def run_validation(
    sims: int = 2000, seed: int = 2026, only: list[str] | None = None
) -> ValidationReport:
    """Run the Monte-Carlo checks.

    Cheap count-based checks use ``10 * sims`` experiments, the bootstrap ``sims / 4`` (each one
    is 1,000 resamples); every row reports its own number of simulated experiments.
    """
    report = ValidationReport(seed=seed, sims=sims)
    children = np.random.default_rng(seed).spawn(len(CHECKS))
    for (name, (func, factor)), child in zip(CHECKS.items(), children, strict=True):
        if only and name not in only:
            continue
        func(max(20, int(sims * factor)), child, report)
    return report
