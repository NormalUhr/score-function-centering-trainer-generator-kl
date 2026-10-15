"""Build the results page: the experiments-at-a-glance table (at_a_glance.py),
the payload timing table, then two charts per setting, training reward and
held-out evaluation reward against the training step, for exactly the runs
the table shows. Charts use chartkit (analysis/chartkit: legend below the
figure, click and double-click to hide and solo, zoom, pan), inlined so the
page is one self-contained file.

    python analysis/progress_page.py results/runs glance.html results/panel.html
"""
import argparse
import datetime
import html
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "chartkit"))

import at_a_glance
import chartkit

STYLE = """
<style>
:root{--bg:#f6f7f5;--ink:#1d2430;--muted:#5f6b7a;--line:#d8dde3;--accent:#0f6e7a;--panel:#ffffff;
 --good:#2e7d32;--warn:#b26a00;--bad:#b3261e;--mono:"IBM Plex Mono",ui-monospace,Menlo,monospace;
 --s1:#0072B2;--s2:#E69F00;--s3:#009E73;--s4:#D55E00}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#15191e;--ink:#e6e8eb;--muted:#9aa4b1;
 --line:#2c333b;--accent:#5cbcc7;--panel:#1c2127}}
:root[data-theme="dark"]{--bg:#15191e;--ink:#e6e8eb;--muted:#9aa4b1;--line:#2c333b;--accent:#5cbcc7;--panel:#1c2127}
body{background:var(--bg);color:var(--ink);font-family:"IBM Plex Sans",system-ui,-apple-system,sans-serif;
 font-size:15px;line-height:1.5;margin:0;padding-block:0 48px;padding-inline:16px}
main{max-width:1040px;margin:0 auto}
header.top{position:sticky;top:env(safe-area-inset-top,0px);background:var(--bg);border-bottom:1px solid var(--line);
 padding-block:12px;margin-inline:-16px;padding-inline:16px;z-index:2;display:flex;flex-wrap:wrap;gap:8px 24px;
 align-items:baseline;justify-content:space-between}
header.top h1{font-size:20px;margin:0;font-weight:600}
header.top .updated{color:var(--muted);font-family:var(--mono);font-size:13px}
h2{font-size:18px;margin:36px 0 12px;color:var(--accent);text-wrap:balance}
h3{font-size:15px;margin:20px 0 8px}
p,li{max-width:70ch}
table{border-collapse:collapse;font-family:var(--mono);font-size:13px;font-variant-numeric:tabular-nums}
th,td{border-bottom:1px solid var(--line);padding:6px 10px;text-align:right;white-space:nowrap}
th:first-child,td:first-child{text-align:left}
thead th{color:var(--muted);font-weight:500;border-bottom:2px solid var(--line)}
.scroll{overflow-x:auto;max-width:100%}
.chip{display:inline-block;padding:1px 8px;border-radius:10px;font-size:12px;font-family:var(--mono);
 border:1px solid var(--line);color:var(--muted)}
.chip.good{color:var(--good);border-color:var(--good)}.chip.warn{color:var(--warn);border-color:var(--warn)}
.chip.bad{color:var(--bad);border-color:var(--bad)}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:6px;padding:12px 16px;margin:12px 0}
.muted{color:var(--muted)}
.pair{display:grid;grid-template-columns:1fr 1fr;gap:16px;align-items:start}
.pair h4{font-size:14px;margin:0 0 4px;font-weight:500;color:var(--muted)}
@media (max-width:760px){.pair{grid-template-columns:1fr}}
@media (prefers-reduced-motion: reduce){*{animation:none!important;transition:none!important}}
</style>
"""


def thin(points, max_points):
    """Every k-th point, plus the last one, so a curve carries at most
    max_points points. The curves are already smoothed over a window wider
    than the stride, so the chart loses nothing visible, and the page stays
    small (one point per step for every run makes it larger than 16 MB)."""
    if len(points) <= max_points:
        return points
    stride = -(-(len(points) - 1) // (max_points - 1))
    kept = points[::stride]
    if kept and kept[-1] is not points[-1]:
        kept.append(points[-1])
    return kept


def training_points(rows, max_points):
    """[(step, training reward)] from a run's logged rows, raw (the chart
    smooths client-side, with the viewer's window), thinned to about
    max_points points with four-decimal values."""
    points = [(row["step"], round(float(row["reward"]), 4)) for row in rows if row.get("reward") is not None]
    return thin(points, max_points)


def eval_points(rows):
    """[(step, held-out evaluation reward)] for every evaluation in the rows."""
    return [(row["step"], round(float(row["eval_reward"]), 4)) for row in rows if row.get("eval_reward") is not None]


# Settings whose runs share one chart pair: the context-length study puts the
# SFC+TIS runs at 4,096, 8,192 and 16,384 tokens in one figure. Each member
# names a setting, the method it contributes (None for every method) and the
# tag that labels its series; a member with a method filter keeps its own pair
# as well, a member without one is shown only in the group, which takes the
# place of the first such member.
CHART_GROUPS = [
    ("Countdown, 4-prompt slices, SFC+TIS at context lengths 4,096, 8,192 and 16,384", [
        ("Countdown, 4-prompt slices", "SFC+TIS", "4,096 tokens"),
        ("Countdown, 4-prompt slices, context length 8,192", None, "8,192 tokens"),
        ("Countdown, 4-prompt slices, context length 16,384", None, "16,384 tokens"),
    ]),
]


def chart_pair(title, series_of, runs, window, max_points):
    """A heading and the two charts for `runs`; `series_of(run)` names each series."""
    train_series, eval_series = [], []
    for run in runs:
        label = series_of(run)
        train_series.append({"id": run["run"], "label": label, "pts": training_points(run["rows"], max_points)})
        points = eval_points(run["rows"])
        if points:
            eval_series.append({"id": run["run"], "label": label, "pts": points})
    return "\n".join([
        f"<h3>{html.escape(title)}</h3>",
        '<div class="pair">',
        '<div><h4>training reward</h4>'
        + chartkit.chart_div(train_series, xlabel="training step", ylabel="training reward", ydigits=3,
                             smooth={"on": True, "window": window, "max": 200}, expand=True) + '</div>',
        '<div><h4>held-out evaluation reward</h4>'
        + (chartkit.chart_div(eval_series, xlabel="training step", ylabel="evaluation reward", ydigits=3,
                              smooth={"on": False, "window": 3, "max": 20}, expand=True)
           if eval_series else '<p class="muted">no evaluation yet</p>') + '</div>',
        '</div>'])


def setting_charts(shown, window=100, max_points=3300):
    """Per setting, in the table's order, a heading and a two-column pair of
    charts: the training reward and the evaluation reward of every shown run
    of that setting, one series per run labelled by its method and seed. Both
    charts carry the smoothing switch (on with `window` points for the training
    reward; off for the evaluation reward, three evaluations when turned on) and
    the expand button. Settings listed in CHART_GROUPS are merged as described there."""
    parts = []
    consumed = {member[0]: title for title, members in CHART_GROUPS for member in members if member[1] is None}
    emitted = set()
    for setting in list(dict.fromkeys(run["setting"] for run in shown)):
        group_title = consumed.get(setting)
        if group_title is not None:
            if group_title in emitted:
                continue
            members = next(members for title, members in CHART_GROUPS if title == group_title)
            runs = [(run, tag) for member_setting, method, tag in members for run in shown
                    if run["setting"] == member_setting and run["rows"] and (method is None or run["method"] == method)]
            if runs:
                tags = {run["run"]: tag for run, tag in runs}
                parts.append(chart_pair(group_title, lambda run: f"{run['method']}, {tags[run['run']]}, seed {run['seed']}",
                                        [run for run, _ in runs], window, max_points))
                emitted.add(group_title)
            continue
        runs = [run for run in shown if run["setting"] == setting and run["rows"]]
        if runs:
            parts.append(chart_pair(setting, lambda run: f"{run['method']}, seed {run['seed']}",
                                    runs, window, max_points))
    return "\n".join(parts)


TIMING_PREFIX = "Tpay"  # the payload timing probes (experiments/payload_timing.yaml)


def _median(values):
    values = sorted(v for v in values if v is not None)
    return values[len(values) // 2] if values else None


def timing_rows(outputs):
    """One row per timing probe run under `outputs`, in payload order (width
    descending, the all-head layout first): the payload, the rollout size, the
    generator's seconds per decode stage of 128 tokens, the trainer's compute
    per step (its wall time minus its wait), and the time to read and to write
    a rollout."""
    rows = []
    for run_dir in sorted(Path(outputs).glob(f"{TIMING_PREFIX}-*")):
        log = at_a_glance.log_path(run_dir, "logs.jsonl")
        if ".partial" in run_dir.name or log is None:
            continue
        config = json.loads((run_dir / "config.json").read_text()) if (run_dir / "config.json").exists() else {}
        head = config.get("sampler", {}).get("vocab_logprobs")
        tail = config.get("sampler", {}).get("tail_samples", 0) or 0
        steps = at_a_glance.read_jsonl(log)
        elapsed = [r.get("elapsed_s") for r in steps]
        compute = [elapsed[i] - elapsed[i - 1] - steps[i].get("wait_s", 0.0)
                   for i in range(1, len(steps)) if elapsed[i] is not None and elapsed[i - 1] is not None]
        batches = []
        for generator_log in sorted(list(run_dir.glob("generator-*.jsonl")) + list(run_dir.glob("generator-*.jsonl.gz"))):
            batches += at_a_glance.read_jsonl(generator_log)
        rows.append({
            "layout": f"top-{head}" + (f" + {tail} tail samples" if tail else ""),
            "width": (head or 0) + tail, "head": head or 0,
            "rollout_mb": _median([b.get("rollout_mb") for b in batches]),
            "s_per_stage": _median([b["decode_s"] / b["stages"] for b in batches if b.get("stages")]),
            "compute_s": _median([c for c in compute if c > 0]),
            "read_s": _median([r.get("rollout_read_s") for r in steps]),
            "push_s": _median([b.get("push_s") for b in batches]),
        })
    rows.sort(key=lambda r: (-r["width"], -r["head"]))
    return rows


def timing_table(outputs):
    """The payload timing table (HTML), or a note when no probe has run."""
    rows = timing_rows(outputs)
    if not rows:
        return '<p class="muted">no timing probes in this results folder (experiments/payload_timing.yaml).</p>'
    def cell(value, digits):
        return "" if value is None else f"{value:.{digits}f}"
    out = ['<div class="scroll"><table><thead><tr><th>payload</th><th>rollout MB</th><th>generator s per 128-token stage</th>'
           '<th>trainer compute s per step</th><th>rollout read s</th><th>rollout write s</th></tr></thead><tbody>']
    for r in rows:
        out.append(f"<tr><td>{html.escape(r['layout'])}</td><td>{cell(r['rollout_mb'], 0)}</td><td>{cell(r['s_per_stage'], 2)}</td>"
                   f"<td>{cell(r['compute_s'], 2)}</td><td>{cell(r['read_s'], 2)}</td><td>{cell(r['push_s'], 2)}</td></tr>")
    out.append("</tbody></table></div>")
    return "\n".join(out)


def build(outputs, glance_html, updated=None, title="SFC+KL: experiments", window=100, max_points=3300):
    updated = updated or datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    shown = at_a_glance.shown_runs(outputs)
    parts = [f"<title>{html.escape(title)}</title>", STYLE, chartkit.inline_js(), "<main>",
             f'<header class="top"><h1>{html.escape(title)}</h1>'
             f'<span class="updated">built {html.escape(updated)}</span></header>',
             glance_html,
             '<section id="timing"><h2>Payload timing probes</h2>',
             '<p class="muted">Cost of each payload at the Countdown 4-prompt setting, with 100 steps per probe, no evaluation '
             'and the same seed. The columns are the size of a rollout file, the generator\'s decode time per stage of 128 tokens, '
             'the trainer\'s compute per step (its wall time minus its wait for a batch), and the time to read and to write a '
             'rollout file, each the median over the probe.</p>',
             timing_table(outputs),
             '</section>',
             '<section id="curves"><h2>Reward curves per setting</h2>',
             '<p class="muted">One pair of charts per setting of the table above, for exactly the runs the table shows: '
             'the training reward (the mean reward of each trainer batch) and the held-out evaluation reward, against the '
             'training step. The smoothing switch above each chart applies a centered moving average whose window the slider '
             'sets (in logged steps for the training reward; off by default for the evaluation reward, in evaluations when on); "expand" opens the '
             'chart in a floating window with the same controls. Click a legend entry to hide a curve, double-click to show '
             'it alone; drag to zoom, double-click the empty plot to reset.</p>',
             setting_charts(shown, window, max_points),
             '</section>', '</main>']
    return "\n".join(parts)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("outputs", help="the runs directory (contains <run>/logs.jsonl or logs.jsonl.gz)")
    parser.add_argument("glance", help="HTML fragment from at_a_glance.py")
    parser.add_argument("out", help="page path")
    parser.add_argument("--title", default="SFC+KL: experiments")
    parser.add_argument("--window", type=int, default=100)
    args = parser.parse_args()
    Path(args.out).write_text(build(args.outputs, Path(args.glance).read_text(), title=args.title, window=args.window))
    print(args.out)


if __name__ == "__main__":
    main()
