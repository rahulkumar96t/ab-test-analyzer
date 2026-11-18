"""``abtest`` command line: simulate, analyze, power, srm, validate, peeking."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Sequence
from pathlib import Path

from abtest import __version__
from abtest.analysis import analyze
from abtest.power import (
    mde_means,
    mde_proportions,
    sample_size_means,
    sample_size_proportions,
)
from abtest.report import render_html, render_markdown, render_text
from abtest.simulate import ExperimentConfig, simulate_experiment
from abtest.spec import SpecError, load_data, load_spec
from abtest.srm import srm_check
from abtest.validation import PeekingRow, Rate, ValidationReport, run_validation, simulate_peeking


def _fmt_rate(r: Rate) -> str:
    lo, hi = r.wilson()
    return f"{r.value:.2%} [{lo:.2%}, {hi:.2%}]"


def _rate_json(r: Rate) -> dict[str, float]:
    lo, hi = r.wilson()
    return {"rate": r.value, "hits": r.hits, "trials": r.trials, "ci95_low": lo, "ci95_high": hi}


# ---------------------------------------------------------------------------- commands


def cmd_simulate(args: argparse.Namespace) -> int:
    cfg = ExperimentConfig(
        n_users=args.users,
        assigned_treatment_share=args.assigned_share,
        days=args.days,
        baseline_conversion=args.baseline,
        conversion_lift=args.conversion_lift,
        order_value_lift=args.order_value_lift,
        sessions_lift=args.sessions_lift,
        page_load_lift=args.page_load_lift,
        bounce_lift=args.bounce_lift,
        seed=args.seed,
    )
    frame = simulate_experiment(cfg)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)
    print(f"wrote {len(frame):,} users to {out}")
    return 0


def cmd_analyze(args: argparse.Namespace) -> int:
    try:
        spec = load_spec(args.spec)
        data = load_data(args.data, spec)
    except (SpecError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    report = analyze(data, spec)
    print(render_text(report), end="")
    if args.report:
        path = Path(args.report)
        path.parent.mkdir(parents=True, exist_ok=True)
        html_out = path.suffix.lower() in (".html", ".htm")
        path.write_text(render_html(report) if html_out else render_markdown(report))
        print(f"\nreport written to {path}")
    return 0


def cmd_power(args: argparse.Namespace) -> int:
    if args.metric == "binary":
        if args.n:
            mde = mde_proportions(
                args.baseline, args.n, args.alpha, args.power, args.relative, args.ratio
            )
            unit = "relative" if args.relative else "absolute"
            print(f"MDE ({unit}) with {args.n:,} control units: {mde:.4%}")
        else:
            size = sample_size_proportions(
                args.baseline, args.mde, args.alpha, args.power, args.relative, args.ratio
            )
            print(f"control {size.control:,}, treatment {size.treatment:,}, total {size.total:,}")
        return 0
    if args.n:
        mde = mde_means(
            args.sd, args.n, args.alpha, args.power, args.ratio, args.variance_reduction
        )
        print(f"MDE (absolute) with {args.n:,} control units: {mde:.6g}")
    else:
        size = sample_size_means(
            args.mde,
            args.sd,
            args.alpha,
            args.power,
            args.ratio,
            args.variance_reduction,
            args.t,
            args.sd_treatment,
        )
        print(f"control {size.control:,}, treatment {size.treatment:,}, total {size.total:,}")
    return 0


def cmd_srm(args: argparse.Namespace) -> int:
    result = srm_check(args.counts, args.expected, args.threshold)
    shares = ", ".join(f"{s:.3%}" for s in result.observed_shares)
    verdict = "MISMATCH" if result.mismatch else "OK"
    stat = f"chi-square {result.statistic:.3f}, p {result.p_value:.3g}"
    print(f"observed shares {shares}; {stat} -> {verdict}")
    return 1 if result.mismatch else 0


def _validation_json(report: ValidationReport, seconds: float) -> dict[str, object]:
    return {
        "seed": report.seed,
        "sims": report.sims,
        "seconds": round(seconds, 1),
        "checks": [
            {
                "procedure": c.procedure,
                "scenario": c.scenario,
                "quantity": c.quantity,
                "target": c.target,
                "consistent_with_target": c.consistent,
                **_rate_json(c.measured),
            }
            for c in report.checks
        ],
        "extras": report.extras,
    }


def cmd_validate(args: argparse.Namespace) -> int:
    start = time.perf_counter()
    report = run_validation(args.sims, args.seed, args.only)
    seconds = time.perf_counter() - start
    for c in report.checks:
        target = "-" if c.target is None else f"{c.target:.1%}"
        flag = {True: "ok", False: "OFF", None: ""}[c.consistent]
        print(
            f"{c.procedure:<42} {c.quantity:<20} target {target:>6}  measured "
            f"{_fmt_rate(c.measured):<26} n={c.measured.trials:<6} {flag:<3} | {c.scenario}"
        )
    for key, value in report.extras.items():
        print(f"{key}: {value:.6g}")
    print(f"{len(report.checks)} checks in {seconds:.1f} s (seed {report.seed})")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(_validation_json(report, seconds), indent=2) + "\n")
    return 0


def _peeking_json(row: PeekingRow) -> dict[str, object]:
    return {
        "looks": row.looks,
        "lift": row.lift,
        "naive": _rate_json(row.naive),
        "fixed_horizon": _rate_json(row.fixed),
        "msprt": _rate_json(row.msprt),
        "group_sequential_obf": _rate_json(row.gst),
        "msprt_mean_sample_fraction": row.msprt_mean_fraction,
        "gst_mean_sample_fraction": row.gst_mean_fraction,
    }


def cmd_peeking(args: argparse.Namespace) -> int:
    start = time.perf_counter()
    rows: list[PeekingRow] = []
    for lift in (0.0, args.lift):
        for looks in args.looks:
            rows.append(
                simulate_peeking(
                    args.sims,
                    looks,
                    args.n_per_arm,
                    args.baseline,
                    lift,
                    0.05,
                    args.tau,
                    args.seed + looks,
                )
            )
    seconds = time.perf_counter() - start
    header = ["naive", "fixed horizon", "mSPRT", "group-seq. OBF"]
    print(
        f"{'lift':>5} {'looks':>5}  " + "".join(f"{h:<24}" for h in header) + "stop at (mSPRT/GST)"
    )
    for r in rows:
        cells = "".join(f"{_fmt_rate(x):<24}" for x in (r.naive, r.fixed, r.msprt, r.gst))
        stop = f"{r.msprt_mean_fraction:.2f}/{r.gst_mean_fraction:.2f}"
        print(f"{r.lift:>5.0%} {r.looks:>5}  {cells}{stop}")
    print(f"{len(rows)} scenarios x {args.sims:,} simulations in {seconds:.1f} s")
    if args.out:
        payload = {
            "sims": args.sims,
            "n_per_arm": args.n_per_arm,
            "baseline": args.baseline,
            "tau_relative": args.tau,
            "seed": args.seed,
            "seconds": round(seconds, 1),
            "rows": [_peeking_json(r) for r in rows],
        }
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(payload, indent=2) + "\n")
    if args.plot:
        from abtest.plots import plot_peeking

        print(f"chart written to {plot_peeking(rows, args.plot)}")
    return 0


# ------------------------------------------------------------------------------ parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="abtest", description=__doc__)
    parser.add_argument("--version", action="version", version=f"abtest {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("simulate", help="write a seeded synthetic experiment as CSV")
    p.add_argument("--out", default="data/experiment.csv")
    p.add_argument("--users", type=int, default=40_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--days", type=int, default=14)
    p.add_argument("--baseline", type=float, default=0.08, help="control conversion rate")
    p.add_argument("--conversion-lift", type=float, default=0.0, help="relative, e.g. 0.05")
    p.add_argument("--order-value-lift", type=float, default=0.0)
    p.add_argument("--sessions-lift", type=float, default=0.0)
    p.add_argument("--page-load-lift", type=float, default=0.0, help="positive = slower")
    p.add_argument("--bounce-lift", type=float, default=0.0, help="positive = more bounces")
    p.add_argument(
        "--assigned-share", type=float, default=None, help="actual treatment share (plants an SRM)"
    )
    p.set_defaults(func=cmd_simulate)

    p = sub.add_parser("analyze", help="analyse a CSV with a TOML spec and print a decision report")
    p.add_argument("data")
    p.add_argument("--spec", required=True)
    p.add_argument("--report", help="also write the report (.md or .html)")
    p.set_defaults(func=cmd_analyze)

    p = sub.add_parser("power", help="sample size or minimum detectable effect")
    p.add_argument("metric", choices=["binary", "continuous"])
    p.add_argument("--baseline", type=float, help="control conversion rate (binary)")
    p.add_argument("--sd", type=float, help="standard deviation per unit (continuous)")
    p.add_argument("--mde", type=float, help="effect to detect")
    p.add_argument("--n", type=int, help="control units available: report the MDE instead")
    p.add_argument("--relative", action="store_true", help="--mde is relative (binary)")
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--power", type=float, default=0.8)
    p.add_argument("--ratio", type=float, default=1.0, help="treatment/control allocation")
    p.add_argument("--variance-reduction", type=float, default=0.0, help="CUPED rho^2 (continuous)")
    p.add_argument("--sd-treatment", type=float, help="treatment SD if it differs (continuous)")
    p.add_argument("--t", action="store_true", help="noncentral t refinement (continuous)")
    p.set_defaults(func=cmd_power)

    p = sub.add_parser("srm", help="sample-ratio mismatch check for arm counts")
    p.add_argument("counts", type=int, nargs="+")
    p.add_argument("--expected", type=float, nargs="+", default=None)
    p.add_argument("--threshold", type=float, default=0.001)
    p.set_defaults(func=cmd_srm)

    p = sub.add_parser("validate", help="Monte-Carlo check of alpha, power and coverage")
    p.add_argument("--sims", type=int, default=2000)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--only", nargs="+", default=None)
    p.add_argument("--out", help="write the results as JSON")
    p.set_defaults(func=cmd_validate)

    p = sub.add_parser(
        "peeking", help="simulate repeated looks: naive vs mSPRT vs group-sequential"
    )
    p.add_argument("--sims", type=int, default=10_000)
    p.add_argument("--looks", type=int, nargs="+", default=[1, 2, 5, 10, 20, 50, 100])
    p.add_argument("--n-per-arm", type=int, default=20_000)
    p.add_argument("--baseline", type=float, default=0.10)
    p.add_argument("--lift", type=float, default=0.10, help="relative lift for the power rows")
    p.add_argument("--tau", type=float, default=0.10, help="mSPRT mixture sd, relative to baseline")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--out", help="write the results as JSON")
    p.add_argument("--plot", help="write the false-positive chart as PNG")
    p.set_defaults(func=cmd_peeking)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "power":
        need = ["baseline"] if args.metric == "binary" else ["sd"]
        if not args.n:
            need.append("mde")
        missing = [f"--{n}" for n in need if getattr(args, n) is None]
        if missing:
            print(f"error: power {args.metric} needs {', '.join(missing)}", file=sys.stderr)
            return 2
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
