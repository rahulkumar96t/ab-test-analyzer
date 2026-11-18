from pathlib import Path

import pandas as pd
import pytest

from abtest.simulate import COLUMNS, ExperimentConfig, simulate_experiment
from abtest.spec import SpecError, load_data, load_spec, spec_from_dict, validate_data

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).parent / "fixtures"


def test_same_seed_same_frame_and_different_seed_differs() -> None:
    a = simulate_experiment(ExperimentConfig(n_users=2000, seed=5))
    b = simulate_experiment(ExperimentConfig(n_users=2000, seed=5))
    c = simulate_experiment(ExperimentConfig(n_users=2000, seed=6))
    pd.testing.assert_frame_equal(a, b)
    assert not a.equals(c)


def test_schema_and_ranges() -> None:
    df = simulate_experiment(ExperimentConfig(n_users=5000, seed=1))
    assert tuple(df.columns) == COLUMNS
    assert set(df.variant) == {"control", "treatment"}
    assert df.day.between(1, 14).all()
    assert (df.sessions >= 1).all()
    assert set(df.converted.unique()) <= {0, 1}
    assert (df.revenue[df.converted == 0] == 0).all()
    assert (df.revenue[df.converted == 1] > 0).all()
    assert df.converted.mean() == pytest.approx(0.08, abs=0.012)


def test_planted_effects_are_recovered() -> None:
    cfg = ExperimentConfig(
        n_users=400_000, conversion_lift=0.20, sessions_lift=0.05, page_load_lift=0.03, seed=2
    )
    df = simulate_experiment(cfg)
    g = df.groupby("variant")[["converted", "page_load_ms", "sessions"]].mean()
    lift = g.loc["treatment"] / g.loc["control"] - 1
    assert lift.converted == pytest.approx(0.20, abs=0.03)
    assert lift.page_load_ms == pytest.approx(0.03, abs=0.004)
    extra = (g.loc["treatment", "sessions"] - 1) / (g.loc["control", "sessions"] - 1) - 1
    assert extra == pytest.approx(0.05, abs=0.01)


def test_assigned_share_plants_a_sample_ratio_mismatch() -> None:
    df = simulate_experiment(ExperimentConfig(n_users=50_000, assigned_treatment_share=0.52))
    assert (df.variant == "treatment").mean() == pytest.approx(0.52, abs=0.006)
    with pytest.raises(ValueError):
        ExperimentConfig(treatment_share=1.0)


def test_example_spec_loads() -> None:
    spec = load_spec(ROOT / "examples" / "checkout.toml")
    assert spec.primary.kind == "binary"
    assert [m.name for m in spec.secondary] == [
        "revenue_per_user",
        "revenue_per_session",
        "sessions_per_user",
    ]
    assert spec.guardrails[0].metric.direction == "decrease"
    assert spec.look_column == "day"
    assert set(spec.required_columns) <= set(COLUMNS)


def test_fixture_csv_with_custom_arm_names_loads() -> None:
    spec = load_spec(FIXTURES / "tiny.toml")
    df = load_data(FIXTURES / "tiny_experiment.csv", spec)
    assert set(df.variant) == {"old_checkout", "new_checkout"}
    assert len(df) == 60


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        ({}, "primary"),
        ({"primary": {"name": "x", "kind": "count", "column": "c"}}, "kind"),
        ({"primary": {"name": "x", "kind": "ratio", "numerator": "a"}}, "denominator"),
        ({"primary": {"name": "x", "kind": "binary", "column": "c", "colour": 1}}, "unknown"),
        ({"primary": {"name": "x", "kind": "binary", "column": "c", "covariate": "p"}}, "CUPED"),
        (
            {"primary": {"name": "x", "kind": "binary", "column": "c"}, "correction": "fdr"},
            "correction",
        ),
        (
            {
                "primary": {"name": "x", "kind": "binary", "column": "c"},
                "secondary": [{"name": "x", "kind": "binary", "column": "d"}],
            },
            "unique",
        ),
        (
            {
                "primary": {"name": "x", "kind": "binary", "column": "c"},
                "guardrails": [{"name": "g", "kind": "continuous", "column": "d"}],
            },
            "margin",
        ),
    ],
)
def test_invalid_specs_are_rejected(raw: dict, message: str) -> None:
    with pytest.raises(SpecError, match=message):
        spec_from_dict(raw)


def test_invalid_data_is_rejected() -> None:
    spec = spec_from_dict({"primary": {"name": "conv", "kind": "binary", "column": "converted"}})
    good = pd.DataFrame({"variant": ["control", "treatment"] * 3, "converted": [0, 1] * 3})
    assert len(validate_data(good, spec)) == 6
    with pytest.raises(SpecError, match="missing"):
        validate_data(good.drop(columns="converted"), spec)
    with pytest.raises(SpecError, match="unexpected variants"):
        validate_data(good.assign(variant=["control", "treatment", "other"] * 2), spec)
    with pytest.raises(SpecError, match="0/1"):
        validate_data(good.assign(converted=[0, 2] * 3), spec)
    with pytest.raises(SpecError, match="non-numeric"):
        validate_data(good.assign(converted=["0", "x"] * 3), spec)
