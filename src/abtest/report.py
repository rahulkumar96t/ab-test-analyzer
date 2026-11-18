"""Render an :class:`ExperimentReport` as plain text, Markdown or self-contained HTML.

The content is built once as a list of sections made of paragraphs, bullet lists and tables;
the three renderers only differ in markup.
"""

from __future__ import annotations

import html
import math
import re
from dataclasses import dataclass, field

from abtest.analysis import ExperimentReport, GuardrailResult, MetricResult
from abtest.frequentist import Comparison
from abtest.spec import MetricSpec


@dataclass
class Table:
    headers: list[str]
    rows: list[list[str]]


@dataclass
class Bullets:
    items: list[str]


Block = str | Table | Bullets


@dataclass
class Section:
    title: str
    blocks: list[Block] = field(default_factory=list)


# ---------------------------------------------------------------------------- formatting


def fmt_p(p: float) -> str:
    return "<0.0001" if p < 1e-4 else f"{p:.4f}"


def fmt_num(x: float) -> str:
    if not math.isfinite(x):
        return "n/a"
    if x != 0 and abs(x) < 0.01:
        return f"{x:.3g}"
    return f"{x:,.4g}" if abs(x) < 1e4 else f"{x:,.0f}"


def fmt_value(metric: MetricSpec, x: float) -> str:
    return f"{x:.2%}" if metric.kind == "binary" else fmt_num(x)


def fmt_diff(metric: MetricSpec, x: float) -> str:
    if metric.kind == "binary":
        return f"{x * 100:+.2f} pp"
    return f"{x:+.3g}" if x != 0 and abs(x) < 0.01 else f"{x:+,.4g}"


def fmt_loss(metric: MetricSpec, x: float) -> str:
    """Expected loss in metric units (percentage points for rates)."""
    return f"{x * 100:.3g} pp" if metric.kind == "binary" else fmt_num(x)


def _ci(metric: MetricSpec, lo: float, hi: float) -> str:
    return f"[{fmt_diff(metric, lo)}, {fmt_diff(metric, hi)}]"


def _rel(c: Comparison) -> str:
    lo, hi = c.relative_ci
    return f"{c.relative_effect:+.2%} [{lo:+.2%}, {hi:+.2%}]"


def _metric_row(m: MetricResult, alpha: float, adjusted: bool) -> list[str]:
    c, spec = m.comparison, m.spec
    row = [
        spec.name + (" (CUPED)" if m.cuped else ""),
        fmt_value(spec, c.control_mean),
        fmt_value(spec, c.treatment_mean),
        f"{fmt_diff(spec, c.effect)} {_ci(spec, *c.ci)}",
        _rel(c),
        fmt_p(c.p_value),
    ]
    if adjusted:
        row.append(fmt_p(m.p_adjusted))
    row += [f"{m.bayes.prob_better:.1%}", fmt_loss(spec, m.bayes.expected_loss_ship)]
    row.append(m.outcome(alpha))
    return row


def _metric_headers(adjusted: bool) -> list[str]:
    h = ["metric", "control", "treatment", "difference [CI]", "relative [CI]", "p"]
    if adjusted:
        h.append("adj. p")
    return [*h, "P(better)", "exp. loss if shipped", "result"]


def _details(m: MetricResult) -> list[str]:
    spec, out = m.spec, [f"{m.spec.name}: {m.comparison.method}; Bayesian: {m.bayes.model}"]
    if m.cuped:
        raw, adj = m.cuped.raw, m.cuped.adjusted
        out.append(
            f"{spec.name}: CUPED on {spec.covariate} (theta {m.cuped.theta:.4g}, correlation "
            f"{m.cuped.correlation:.3f}) reduced the variance of the estimate by "
            f"{m.cuped.variance_reduction:.1%}; CI width {fmt_num(raw.ci[1] - raw.ci[0])} -> "
            f"{fmt_num(adj.ci[1] - adj.ci[0])}, raw p {fmt_p(raw.p_value)}"
        )
    if m.rank:
        out.append(
            f"{spec.name}: Mann-Whitney p {fmt_p(m.rank.p_value)}, "
            f"P(treatment > control) {m.rank.prob_superiority:.4f} (tests ordering, not means)"
        )
    if m.bootstrap:
        b = m.bootstrap
        out.append(
            f"{spec.name}: bootstrap ({b.n_resamples} resamples, {b.method}) of the raw difference "
            f"{fmt_diff(spec, b.estimate)} {_ci(spec, b.ci_low, b.ci_high)}"
        )
    return out


GUARDRAIL_HEADERS = [
    "metric",
    "control",
    "treatment",
    "degradation",
    "worst case",
    "margin",
    "status",
]


def _guardrail_row(g: GuardrailResult) -> list[str]:
    spec = g.spec.metric
    return [
        f"{spec.name} ({'lower' if spec.direction == 'decrease' else 'higher'} is better)",
        fmt_value(spec, g.comparison.control_mean),
        fmt_value(spec, g.comparison.treatment_mean),
        f"{g.degradation:+.2%}",
        f"{g.degradation_upper:+.2%}",
        f"{g.spec.margin:.1%}",
        g.status.upper(),
    ]


def build_sections(report: ExperimentReport) -> list[Section]:
    spec = report.spec
    level = f"{1 - spec.alpha:.0%}"
    d = report.decision
    sections = [Section("Decision", [f"Verdict: {d.verdict.upper()}", Bullets(list(d.reasons))])]

    total = report.n_control + report.n_treatment
    arms = [
        [name, f"{n:,}", f"{n / total:.2%}", f"{share / sum(spec.expected_split):.2%}"]
        for name, n, share in zip(
            (spec.control, spec.treatment),
            (report.n_control, report.n_treatment),
            spec.expected_split,
            strict=True,
        )
    ]
    srm = report.srm
    srm_line = (
        f"Sample-ratio check: chi-square {srm.statistic:.3f}, p {fmt_p(srm.p_value)} "
        f"(threshold {srm.threshold:g}) -> {'MISMATCH' if srm.mismatch else 'OK'}"
    )
    sections.append(
        Section("Sample", [Table(["arm", "units", "share", "expected share"], arms), srm_line])
    )
    sections.append(
        Section(
            "Primary metric",
            [
                Table(_metric_headers(False), [_metric_row(report.primary, spec.alpha, False)]),
                f"{level} confidence intervals, two-sided alpha {spec.alpha:g}. "
                f"Minimum detectable effect at this sample size ({spec.power:.0%} power): "
                f"{report.primary_mde_relative:.2%} relative.",
            ],
        )
    )
    if report.secondary:
        sections.append(
            Section(
                f"Secondary metrics ({spec.correction} correction)",
                [
                    Table(
                        _metric_headers(True),
                        [_metric_row(m, spec.alpha, True) for m in report.secondary],
                    )
                ],
            )
        )
    if report.guardrails:
        sections.append(
            Section(
                "Guardrails",
                [
                    Table(GUARDRAIL_HEADERS, [_guardrail_row(g) for g in report.guardrails]),
                    f"Worst case = one-sided {1 - spec.guardrail_alpha:.0%} upper bound of the "
                    "relative degradation. PASS: below the margin; FAIL: significant degradation "
                    "that may exceed it; INCONCLUSIVE: neither can be shown.",
                ],
            )
        )
    if report.sequential:
        s = report.sequential
        stop = s.first_rejection
        sections.append(
            Section(
                "Sequential monitoring (mSPRT)",
                [
                    f"{s.looks} cumulative looks on {spec.look_column!r}; always-valid p "
                    f"{fmt_p(s.p_value)}; "
                    + (f"first below {s.alpha:g} at look {stop}" if stop else "never below alpha")
                    + "; confidence sequence for the difference "
                    + f"{_ci(spec.primary, s.ci_low, s.ci_high)} (mixture tau {s.tau:.3g}).",
                    "Always-valid p by look: " + ", ".join(fmt_p(p) for p in s.p_values),
                ],
            )
        )
    details = [line for m in (report.primary, *report.secondary) for line in _details(m)]
    sections.append(Section("Method details", [Bullets(details)]))
    return sections


# ----------------------------------------------------------------------------- renderers


def _title(report: ExperimentReport) -> str:
    s = report.spec
    return f"Experiment report: {s.name} ({s.treatment} vs {s.control})"


def render_text(report: ExperimentReport) -> str:
    lines = [_title(report), "=" * len(_title(report))]
    for section in build_sections(report):
        lines += ["", section.title, "-" * len(section.title)]
        for block in section.blocks:
            if isinstance(block, Table):
                widths = [
                    max(len(r[i]) for r in [block.headers, *block.rows])
                    for i in range(len(block.headers))
                ]
                for j, row in enumerate([block.headers, *block.rows]):
                    lines.append(
                        "  ".join(
                            cell.ljust(w) for cell, w in zip(row, widths, strict=True)
                        ).rstrip()
                    )
                    if j == 0:
                        lines.append("  ".join("-" * w for w in widths))
            elif isinstance(block, Bullets):
                lines += [f"- {item}" for item in block.items]
            else:
                lines.append(block)
    return "\n".join(lines) + "\n"


def render_markdown(report: ExperimentReport) -> str:
    out = [f"# {_title(report)}"]
    for section in build_sections(report):
        out += ["", f"## {section.title}", ""]
        for block in section.blocks:
            if isinstance(block, Table):
                out.append("| " + " | ".join(block.headers) + " |")
                out.append("|" + "|".join("---" for _ in block.headers) + "|")
                out += ["| " + " | ".join(row) + " |" for row in block.rows]
                out.append("")
            elif isinstance(block, Bullets):
                out += [f"- {item}" for item in block.items]
                out.append("")
            else:
                out += [block, ""]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(out)).rstrip() + "\n"


_CSS = """
:root { color-scheme: light dark; --fg: #1d2330; --muted: #5b6474; --line: #d9dee7;
  --bg: #ffffff; --head: #f3f5f9; --ok: #1f7a3d; --bad: #b42318; --warn: #9a6700; }
@media (prefers-color-scheme: dark) { :root { --fg: #e6e9ef; --muted: #a0a8b8;
  --line: #343b4a; --bg: #14171f; --head: #1d2230; --ok: #5cc98a; --bad: #ff8a80;
  --warn: #e3b341; } }
body { font: 15px/1.5 system-ui, sans-serif; color: var(--fg); background: var(--bg);
  max-width: 1100px; margin: 2rem auto; padding: 0 16px; }
h1 { font-size: 1.4rem; } h2 { font-size: 1.1rem; margin-top: 2rem; }
.scroll { overflow-x: auto; }
table { border-collapse: collapse; margin: .5rem 0; font-variant-numeric: tabular-nums; }
th, td { border-bottom: 1px solid var(--line); padding: .35rem .7rem; text-align: left;
  white-space: nowrap; }
th { background: var(--head); font-weight: 600; }
p, li { color: var(--fg); } .muted { color: var(--muted); }
.verdict { font-size: 1.2rem; font-weight: 700; }
.ship, .PASS { color: var(--ok); } .do-not-ship, .invalid, .FAIL { color: var(--bad); }
.inconclusive, .INCONCLUSIVE { color: var(--warn); }
"""


def render_html(report: ExperimentReport) -> str:
    esc = html.escape
    body = [f"<h1>{esc(_title(report))}</h1>"]
    verdict_class = report.decision.verdict.replace(" ", "-")
    for section in build_sections(report):
        body.append(f"<h2>{esc(section.title)}</h2>")
        for block in section.blocks:
            if isinstance(block, Table):
                head = "".join(f"<th>{esc(h)}</th>" for h in block.headers)
                rows = "".join(
                    "<tr>"
                    + "".join(
                        f'<td class="{esc(c)}">{esc(c)}</td>'
                        if c.isupper()
                        else f"<td>{esc(c)}</td>"
                        for c in r
                    )
                    + "</tr>"
                    for r in block.rows
                )
                body.append(f'<div class="scroll"><table><tr>{head}</tr>{rows}</table></div>')
            elif isinstance(block, Bullets):
                body.append("<ul>" + "".join(f"<li>{esc(i)}</li>" for i in block.items) + "</ul>")
            elif block.startswith("Verdict:"):
                body.append(f'<p class="verdict {verdict_class}">{esc(block)}</p>')
            else:
                body.append(f'<p class="muted">{esc(block)}</p>')
    return (
        '<!doctype html>\n<html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>{esc(report.spec.name)} report</title><style>{_CSS}</style></head>"
        f"<body>{''.join(body)}</body></html>\n"
    )
