from pathlib import Path

import pytest

from abtest.analysis import ExperimentReport, analyze
from abtest.cli import main
from abtest.report import render_html, render_markdown, render_text
from abtest.simulate import ExperimentConfig, simulate_experiment
from abtest.spec import load_spec, validate_data

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "examples" / "checkout.toml"
FIXTURES = Path(__file__).parent / "fixtures"


def _report(**kwargs: float) -> ExperimentReport:
    spec = load_spec(SPEC)
    cfg = ExperimentConfig(
        n_users=int(kwargs.pop("n_users", 60_000)), seed=int(kwargs.pop("seed", 3)), **kwargs
    )
    return analyze(validate_data(simulate_experiment(cfg), spec), spec)


@pytest.fixture(scope="module")
def winning_report() -> ExperimentReport:
    return _report(n_users=120_000, conversion_lift=0.15, sessions_lift=0.03)


def test_clear_win_ships(winning_report: ExperimentReport) -> None:
    r = winning_report
    assert r.decision.verdict == "ship"
    assert r.primary.outcome(0.05) == "better"
    assert all(g.status == "pass" for g in r.guardrails)
    assert r.sequential is not None and r.sequential.first_rejection is not None


def test_secondary_metrics_are_corrected(winning_report: ExperimentReport) -> None:
    for m in winning_report.secondary:
        assert m.p_adjusted >= m.comparison.p_value
    sessions = winning_report.secondary[2]
    assert sessions.cuped is not None and sessions.cuped.variance_reduction > 0.5
    revenue = winning_report.secondary[0]
    assert revenue.rank is not None and revenue.bootstrap is not None


def test_sample_ratio_mismatch_invalidates_everything() -> None:
    r = _report(conversion_lift=0.15, assigned_treatment_share=0.53)
    assert r.srm.mismatch
    assert r.decision.verdict == "invalid"


def test_guardrail_regression_blocks_shipping() -> None:
    r = _report(n_users=120_000, conversion_lift=0.15, page_load_lift=0.08)
    page_load = r.guardrails[0]
    assert page_load.status == "fail"
    assert r.decision.verdict == "do not ship"


def test_no_effect_is_inconclusive_and_reports_the_mde() -> None:
    r = _report(seed=11)
    assert r.decision.verdict == "inconclusive"
    assert 0.03 < r.primary_mde_relative < 0.20
    assert "could detect" in r.decision.reasons[0]


def test_renderers(winning_report: ExperimentReport) -> None:
    text, md, page = (f(winning_report) for f in (render_text, render_markdown, render_html))
    assert "Verdict: SHIP" in text
    assert "| metric |" in md and "## Guardrails" in md
    assert page.startswith("<!doctype html>") and "Sequential monitoring" in page


def test_cli_end_to_end(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    csv = tmp_path / "exp.csv"
    assert (
        main(
            [
                "simulate",
                "--users",
                "20000",
                "--seed",
                "4",
                "--conversion-lift",
                "0.3",
                "--out",
                str(csv),
            ]
        )
        == 0
    )
    report = tmp_path / "out" / "report.html"
    assert main(["analyze", str(csv), "--spec", str(SPEC), "--report", str(report)]) == 0
    out = capsys.readouterr().out
    assert "Verdict:" in out and "Sample-ratio check" in out
    assert report.read_text().startswith("<!doctype html>")


def test_cli_on_committed_fixture(capsys: pytest.CaptureFixture[str]) -> None:
    args = ["analyze", str(FIXTURES / "tiny_experiment.csv"), "--spec", str(FIXTURES / "tiny.toml")]
    assert main(args) == 0
    assert "new_checkout vs old_checkout" in capsys.readouterr().out


def test_cli_errors(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    bad = tmp_path / "bad.toml"
    bad.write_text('name = "x"\n')
    assert main(["analyze", str(FIXTURES / "tiny_experiment.csv"), "--spec", str(bad)]) == 2
    assert main(["power", "binary", "--mde", "0.01"]) == 2
    assert "error" in capsys.readouterr().err


def test_cli_power_and_srm(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["power", "continuous", "--sd", "1", "--mde", "0.5", "--t"]) == 0
    assert "control 64" in capsys.readouterr().out
    assert main(["power", "binary", "--baseline", "0.1", "--n", "3841"]) == 0
    assert "MDE (absolute)" in capsys.readouterr().out
    assert main(["srm", "50000", "50000"]) == 0
    assert main(["srm", "51000", "49000"]) == 1


def test_cli_validate_and_peeking(tmp_path: Path) -> None:
    out = tmp_path / "v.json"
    assert main(["validate", "--sims", "30", "--only", "ztest", "ratio", "--out", str(out)]) == 0
    assert '"checks"' in out.read_text()
    png, js = tmp_path / "peek.png", tmp_path / "peek.json"
    args = ["peeking", "--sims", "300", "--looks", "1", "5", "--out", str(js), "--plot", str(png)]
    assert main(args) == 0
    assert png.stat().st_size > 1000 and '"msprt"' in js.read_text()
