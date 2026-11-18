"""Experiment specification (TOML) and CSV loading.

The spec says which column holds the variant, which arm is the control, which metrics to
analyse and how. Data is one row per randomisation unit (user); see ``examples/``.
"""

from __future__ import annotations

import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import pandas as pd

from abtest.multiple import CORRECTIONS, Correction

MetricKind = Literal["binary", "continuous", "ratio"]
Direction = Literal["increase", "decrease"]


class SpecError(ValueError):
    """The spec or the data does not describe a valid experiment."""


@dataclass(frozen=True)
class MetricSpec:
    name: str
    kind: MetricKind
    column: str | None = None
    numerator: str | None = None
    denominator: str | None = None
    covariate: str | None = None
    direction: Direction = "increase"
    robust: bool = False
    bootstrap: bool = False

    @property
    def columns(self) -> list[str]:
        cols = [self.numerator, self.denominator] if self.kind == "ratio" else [self.column]
        if self.covariate:
            cols.append(self.covariate)
        return [c for c in cols if c]


@dataclass(frozen=True)
class GuardrailSpec:
    metric: MetricSpec
    margin: float
    """Largest tolerated relative degradation, e.g. 0.02 = 2 % worse."""


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    primary: MetricSpec
    variant_column: str = "variant"
    control: str = "control"
    treatment: str = "treatment"
    secondary: tuple[MetricSpec, ...] = ()
    guardrails: tuple[GuardrailSpec, ...] = ()
    expected_split: tuple[float, float] = (0.5, 0.5)
    alpha: float = 0.05
    guardrail_alpha: float = 0.05
    srm_threshold: float = 0.001
    correction: Correction = "holm"
    power: float = 0.8
    bootstrap_resamples: int = 2000
    seed: int = 0
    look_column: str | None = None
    msprt_tau_relative: float = 0.05

    @property
    def metrics(self) -> tuple[MetricSpec, ...]:
        return (self.primary, *self.secondary)

    @property
    def required_columns(self) -> list[str]:
        cols = [self.variant_column]
        if self.look_column:
            cols.append(self.look_column)
        for m in (*self.metrics, *(g.metric for g in self.guardrails)):
            cols.extend(m.columns)
        return list(dict.fromkeys(cols))


_METRIC_KEYS = {
    "name",
    "kind",
    "column",
    "numerator",
    "denominator",
    "covariate",
    "direction",
    "robust",
    "bootstrap",
}


def _metric(raw: Mapping[str, Any], where: str, extra: frozenset[str] = frozenset()) -> MetricSpec:
    unknown = set(raw) - _METRIC_KEYS - extra
    if unknown:
        raise SpecError(f"{where}: unknown keys {sorted(unknown)}")
    if "name" not in raw or "kind" not in raw:
        raise SpecError(f"{where}: 'name' and 'kind' are required")
    kind, direction = raw["kind"], raw.get("direction", "increase")
    if kind not in ("binary", "continuous", "ratio"):
        raise SpecError(f"{where}: kind must be binary, continuous or ratio, not {kind!r}")
    if direction not in ("increase", "decrease"):
        raise SpecError(f"{where}: direction must be 'increase' or 'decrease'")
    if kind == "ratio" and not (raw.get("numerator") and raw.get("denominator")):
        raise SpecError(f"{where}: a ratio metric needs 'numerator' and 'denominator'")
    if kind != "ratio" and not raw.get("column"):
        raise SpecError(f"{where}: a {kind} metric needs 'column'")
    if raw.get("covariate") and kind != "continuous":
        raise SpecError(f"{where}: CUPED covariates are supported for continuous metrics only")
    return MetricSpec(
        name=str(raw["name"]),
        kind=cast(MetricKind, kind),
        column=raw.get("column"),
        numerator=raw.get("numerator"),
        denominator=raw.get("denominator"),
        covariate=raw.get("covariate"),
        direction=cast(Direction, direction),
        robust=bool(raw.get("robust", False)),
        bootstrap=bool(raw.get("bootstrap", False)),
    )


def spec_from_dict(raw: Mapping[str, Any]) -> ExperimentSpec:
    if "primary" not in raw:
        raise SpecError("the spec needs a [primary] metric")
    split = tuple(float(x) for x in raw.get("expected_split", (0.5, 0.5)))
    if len(split) != 2 or min(split) <= 0:
        raise SpecError("expected_split must be two positive numbers")
    correction = raw.get("correction", "holm")
    if correction not in CORRECTIONS:
        raise SpecError(f"correction must be one of {CORRECTIONS}")
    alpha = float(raw.get("alpha", 0.05))
    if not 0 < alpha < 0.5:
        raise SpecError("alpha must lie in (0, 0.5)")
    guardrails = []
    for i, g in enumerate(raw.get("guardrails", [])):
        if "margin" not in g or float(g["margin"]) <= 0:
            raise SpecError(f"guardrails[{i}]: a positive relative 'margin' is required")
        metric = _metric(g, f"guardrails[{i}]", frozenset({"margin"}))
        guardrails.append(GuardrailSpec(metric=metric, margin=float(g["margin"])))
    spec = ExperimentSpec(
        name=str(raw.get("name", "experiment")),
        primary=_metric(raw["primary"], "primary"),
        variant_column=str(raw.get("variant_column", "variant")),
        control=str(raw.get("control", "control")),
        treatment=str(raw.get("treatment", "treatment")),
        secondary=tuple(
            _metric(m, f"secondary[{i}]") for i, m in enumerate(raw.get("secondary", []))
        ),
        guardrails=tuple(guardrails),
        expected_split=(split[0], split[1]),
        alpha=alpha,
        guardrail_alpha=float(raw.get("guardrail_alpha", 0.05)),
        srm_threshold=float(raw.get("srm_threshold", 0.001)),
        correction=cast(Correction, correction),
        power=float(raw.get("power", 0.8)),
        bootstrap_resamples=int(raw.get("bootstrap_resamples", 2000)),
        seed=int(raw.get("seed", 0)),
        look_column=raw.get("look_column"),
        msprt_tau_relative=float(raw.get("msprt_tau_relative", 0.05)),
    )
    names = [m.name for m in (*spec.metrics, *(g.metric for g in spec.guardrails))]
    if len(names) != len(set(names)):
        raise SpecError("metric names must be unique")
    return spec


def load_spec(path: str | Path) -> ExperimentSpec:
    with Path(path).open("rb") as fh:
        return spec_from_dict(tomllib.load(fh))


def load_data(path: str | Path, spec: ExperimentSpec) -> pd.DataFrame:
    """Read a CSV and check it against the spec (columns, arms, numeric values, no gaps)."""
    frame = pd.read_csv(path)
    return validate_data(frame, spec)


def validate_data(frame: pd.DataFrame, spec: ExperimentSpec) -> pd.DataFrame:
    missing = [c for c in spec.required_columns if c not in frame.columns]
    if missing:
        raise SpecError(f"columns missing from the data: {missing}")
    variants = frame[spec.variant_column].astype(str)
    unknown = sorted(set(variants) - {spec.control, spec.treatment})
    if unknown:
        raise SpecError(
            f"unexpected variants {unknown}; expected {spec.control!r}, {spec.treatment!r}"
        )
    out = frame.assign(**{spec.variant_column: variants})
    for col in spec.required_columns[1:]:
        values = pd.to_numeric(out[col], errors="coerce")
        if values.isna().any():
            raise SpecError(
                f"column {col!r} has {int(values.isna().sum())} missing or non-numeric values"
            )
        out[col] = values
    for m in spec.metrics + tuple(g.metric for g in spec.guardrails):
        if m.kind == "binary" and not out[m.column].isin([0, 1]).all():
            raise SpecError(f"binary metric {m.name!r}: column {m.column!r} must contain 0/1")
    return out
