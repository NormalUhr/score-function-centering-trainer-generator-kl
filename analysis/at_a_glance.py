"""Experiments at a glance: one row per run with its status (running at step N
of the budget; done or stopped at step N), its best held-out evaluation, and
the share of responses cut at the context length. Emits an HTML section for
the results page.

    python analysis/at_a_glance.py results/runs > glance.html

A run is a directory `<prefix>-<method>-s<seed>` holding `logs.jsonl` (or
`logs.jsonl.gz`), `config.json` and, when the run did not reach its budget,
a `STOPPED` marker whose text says why. Set-aside copies (`.partial`) and the
timing probes are left out.
"""
import gzip
import html
import json
import re
import sys
from pathlib import Path

import numpy as np

BUDGET = 3200  # steps; the quantized-sampler study (Q prefixes) stops at 800
SETTINGS = [  # run prefix, label, optional step budget when not BUDGET; the order is the table order
    ("Lst4", "Countdown, 4-prompt slices"),
    ("Lst4b8k", "Countdown, 4-prompt slices, context length 8,192"),
    ("Lst4b16k", "Countdown, 4-prompt slices, context length 16,384"),
    ("Lst2", "Countdown, 2-prompt slices"),
    ("Lat4", "AdamW, 4-prompt slices"),
    ("Lag4", "GSM8K, 4-prompt slices"),
    ("M4st1", "Qwen3-4B, 1-prompt slices"),
    ("Lp30ba1", "Qwen3-30B-A3B, 1-prompt slices"),
    ("Lglma1", "GLM-4.5-Air, 1-prompt slices"),
    ("Qf8", "Qwen3-30B-A3B-Base, INTELLECT math, FP8 sampler", 800),
    ("Qf8k4", "Qwen3-30B-A3B-Base, INTELLECT math, FP8 sampler, FP4 KV cache", 800),
    ("Qi8k4", "Qwen3-30B-A3B-Base, INTELLECT math, INT8 sampler, INT4 KV cache", 800),
]
# Method token of a run name -> its label. "klr0.3" is the KL term toward the
# sampler with coefficient beta_0 = 0.3; "k<head>-t<draws>" is a payload of a
# top-<head> head and <draws> tail samples (top-128 without tail samples when
# absent).
METHODS = [("sc", "SFC"), ("tis-sc", "SFC+TIS"), ("mis-sc", "SFC+MIS"), ("sc-klr0.3", "SFC+KL"),
           ("sc-klr0.03", "SFC+KL, β₀ = 0.03"),
           ("ppo", "PPO"), ("dapo", "DAPO"), ("gspo", "GSPO"), ("dppo", "DPPO"), ("topr", "TOPR"),
           ("topr-sc-klr0.3", "TOPR + SFC+KL"), ("gspo-sc-klr0.3", "GSPO + SFC+KL"),
           ("sc-klr0.3-k64-t64", "SFC+KL, top-64 + 64 tail samples"),
           ("sc-klr0.3-k32-t96", "SFC+KL, top-32 + 96 tail samples"),
           ("sc-klr0.3-k32", "SFC+KL, top-32"),
           ("sc-klr0.3-k16-t16", "SFC+KL, top-16 + 16 tail samples"),
           ("sc-klr0.3-k8-t24", "SFC+KL, top-8 + 24 tail samples"),
           ("sc-klr0.3-k16", "SFC+KL, top-16"),
           ("sc-klr0.3-k8-t8", "SFC+KL, top-8 + 8 tail samples"),
           ("sc-klr0.3-k4-t12", "SFC+KL, top-4 + 12 tail samples")]
# The order of a setting's rows and of every chart's legend follows the
# articles: the four SFC methods, the other published methods, SFC+KL on top
# of one of them, and the payload layouts by width (128, 32, 16), the
# all-head layout first and then growing tail samples.
ORDER = [label for _, label in METHODS]


def parse_name(name):
    """(prefix, setting label, method label, seed, step budget) of a run
    directory name, or None for a name that is not a table row."""
    match = re.match(r"([A-Za-z0-9]+)-(.+)-s(\d+)$", name)
    if not match:
        return None
    prefix, rest, seed = match.groups()
    method = next((label for token, label in METHODS if rest == token), None)
    entry = next((e for e in SETTINGS if prefix == e[0]), None)
    if method is None or entry is None:
        return None
    budget = entry[2] if len(entry) > 2 else BUDGET
    return prefix, entry[1], method, int(seed), budget


def log_path(run_dir, name):
    """The plain or gzipped log `name` under `run_dir`, or None."""
    for candidate in (run_dir / name, run_dir / f"{name}.gz"):
        if candidate.exists():
            return candidate
    return None


def read_jsonl(path):
    """The records of a plain or gzipped JSON-lines file, up to the first
    unreadable line (a run that is still writing)."""
    rows = []
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                break
    return rows


def read_run(run_dir):
    """Logged rows (none for a run that has not logged a step yet) and the STOPPED marker."""
    path = log_path(run_dir, "logs.jsonl")
    rows = read_jsonl(path) if path else []
    stopped = (run_dir / "STOPPED").read_text().strip() if (run_dir / "STOPPED").exists() else None
    return rows, stopped


def status_of(rows, stopped, budget=BUDGET):
    """(status text, rows to read)."""
    steps = len(rows)
    if stopped is None:
        if steps >= budget:
            return f"done at {budget:,}", rows[:budget + 1]
        return f"running, step {steps:,} of {budget:,}", rows
    if "budget fixed at 3,200" in stopped:
        # The record of step 3,200 holds that step's evaluation.
        return "done at 3,200", rows[:BUDGET + 1]
    if "collapse rule" in stopped:
        return f"stopped at {steps:,} (collapsed)", rows
    if stopped.startswith("stopped by hand"):
        return f"stopped at {steps:,} (by hand)", rows
    return f"stopped at {steps:,}", rows


def readings(rows):
    """The best held-out evaluation, and the truncation rate over the last 200
    logged steps."""
    cut = [r["frac_truncated"] for r in rows[-200:] if r.get("frac_truncated") is not None]
    truncated = float(np.mean(cut)) if cut else None
    evals = [r["eval_reward"] for r in rows if r.get("eval_reward") is not None]
    return (max(evals) if evals else None), truncated


# The styles ship inside the section so the colors and the setting boundary
# survive a splice into another page.
STYLE = ('<style>#glance .setting-first td{border-top:3px solid #333}'
         '#glance .st-collapsed{color:#c0392b;font-weight:600}#glance .st-running{color:#b7791f;font-weight:600}'
         '#glance .st-done{color:#1e8449;font-weight:600}#glance .st-stopped{color:#7f8c8d}'
         '@media (prefers-color-scheme: dark){#glance .setting-first td{border-top-color:#ccc}'
         '#glance .st-collapsed{color:#ff7b6b}#glance .st-running{color:#f2c744}#glance .st-done{color:#5ad28a}#glance .st-stopped{color:#a0a8ad}}</style>')


def status_class(status):
    if "(collapsed)" in status:
        return "st-collapsed"
    if status.startswith("running"):
        return "st-running"
    if status.startswith("done"):
        return "st-done"
    return "st-stopped"


def shown_runs(outputs):
    """The runs the table shows, in table order: for each, the directory name,
    the setting and method labels, the seed, the status text and the rows the
    table reads (up to the budget). Set-aside copies and names that are not
    table rows are left out, so the page's charts follow the table."""
    shown = []
    for run_dir in sorted(Path(outputs).iterdir()):
        # A launched run writes its config before its first logged step, so a
        # directory with only a config is a run that is loading or in its
        # step-0 evaluation; it is listed rather than omitted.
        if ".partial" in run_dir.name or not (log_path(run_dir, "logs.jsonl") or (run_dir / "config.json").exists()):
            continue
        info = parse_name(run_dir.name)
        if info is None:
            continue
        prefix, setting, method, seed, budget = info
        rows, stopped = read_run(run_dir)
        status, read_rows = status_of(rows, stopped, budget)
        shown.append({"run": run_dir.name, "prefix": prefix, "setting": setting, "method": method, "seed": seed,
                      "status": status, "rows": read_rows})
    setting_rank = {entry[1]: i for i, entry in enumerate(SETTINGS)}
    method_rank = {label: i for i, label in enumerate(ORDER)}
    shown.sort(key=lambda r: (setting_rank[r["setting"]], method_rank.get(r["method"], len(ORDER)), r["seed"]))
    return shown


def build(outputs, note):
    """The table section; `note` says where the rows come from."""
    entries = []
    for shown in shown_runs(outputs):
        best, truncated = readings(shown["rows"])
        best_text = "none yet" if not shown["rows"] else ("" if best is None else f"{best:.3f}")
        entries.append((shown["seed"], shown["setting"], shown["method"], shown["status"], best_text, truncated))
    out = ['<section id="glance"><h2>Experiments at a glance</h2>',
           STYLE,
           f'<p>{html.escape(note)}. Budget 3,200 steps (800 for the quantized-sampler study). Best eval is the highest held-out '
           'evaluation of the run. Truncated is the share of responses cut at the context length over the last 200 steps. A run '
           'stopped by the collapse rule (scripts/stop_rule.py) is read at its stopped length.</p>',
           '<div class="scroll"><table><thead><tr><th>setting</th><th>method</th><th>seed</th><th>status</th>'
           '<th>best eval</th><th>truncated</th></tr></thead><tbody>']
    last_setting = None
    for seed, setting, method, status, best, truncated in entries:
        first = setting != last_setting
        show = setting if first else ""
        last_setting = setting
        pct = "" if truncated is None else f"{100 * truncated:.0f}%"
        out.append(f'<tr{" class=\"setting-first\"" if first else ""}><td>{html.escape(show)}</td><td>{html.escape(method)}</td><td>{seed}</td>'
                   f'<td class="{status_class(status)}">{html.escape(status)}</td><td>{html.escape(best)}</td><td>{pct}</td></tr>')
    out.append("</tbody></table></div></section>")
    return "\n".join(out)


def main():
    outputs = sys.argv[1]
    note = sys.argv[2] if len(sys.argv) > 2 else f"Built from {outputs}"
    print(build(outputs, note))


if __name__ == "__main__":
    main()
