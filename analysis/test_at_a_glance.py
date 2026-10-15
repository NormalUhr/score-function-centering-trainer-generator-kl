"""Fixture test for the results table: synthetic runs (running, done at the
budget, stopped by the collapse rule, stopped by hand) give the expected rows,
statuses and labels, and a run whose log is gzipped reads like a plain one."""
import gzip
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import at_a_glance


def write(root, name, n, reward, eval_at=(), stopped=None, truncated=0.25, gz=False):
    run_dir = Path(root) / name
    run_dir.mkdir(parents=True)
    rows = []
    for step in range(n):
        row = {"step": step, "reward": reward, "frac_truncated": truncated}
        if step in eval_at:
            row["eval_reward"] = 0.1 if step == 0 else 0.6 + 0.0001 * step
        rows.append(json.dumps(row))
    if gz:
        with gzip.open(run_dir / "logs.jsonl.gz", "wt") as f:
            f.write("\n".join(rows) + "\n")
    else:
        (run_dir / "logs.jsonl").write_text("\n".join(rows) + "\n")
    (run_dir / "config.json").write_text("{}")
    if stopped:
        (run_dir / "STOPPED").write_text(stopped)


def main():
    with tempfile.TemporaryDirectory() as root:
        write(root, "Lst4-tis-sc-s0", 3300, 0.0, eval_at={0, 3200}, truncated=0.99,
              stopped="budget fixed at 3,200 steps: read this run at step 3,200 and ignore later steps.")
        write(root, "Lst4-sc-klr0.3-s0", 3200, 0.75, eval_at={0, 400, 3199})
        write(root, "Lst2-sc-s0", 640, 0.0, stopped="stopped by the collapse rule (200-step mean training reward below the floor after step 400); read at the stopped length.")
        write(root, "Lat4-ppo-s0", 500, 0.3, eval_at={0, 400})
        write(root, "Qi8k4-tis-sc-s0", 800, 0.3, eval_at={0, 200, 400, 600, 799})
        write(root, "Qf8-sc-klr0.03-s0", 300, 0.5, eval_at={0, 200}, gz=True)
        write(root, "M4st1-dapo-s0", 1226, 0.5, eval_at={0, 400, 800}, stopped="stopped by hand; read at the stopped length.")
        (Path(root) / "Lst4-sc-klr0.3-s0.partial-x").mkdir()  # a set-aside copy is never a run
        page = at_a_glance.build(root, "built from the results folder")
        assert '<tr class="setting-first"><td>Countdown, 4-prompt slices</td><td>SFC+TIS</td><td>0</td><td class="st-done">done at 3,200</td><td>0.920</td><td>99%</td>' in page, page
        assert '<td>SFC+KL</td><td>0</td><td class="st-done">done at 3,200</td><td>0.920</td>' in page, page
        assert '<td>SFC</td><td>0</td><td class="st-collapsed">stopped at 640 (collapsed)</td><td></td><td>25%</td>' in page, page
        assert '<td class="st-running">running, step 500 of 3,200</td><td>0.640</td><td>25%</td>' in page, page
        assert '<td class="st-stopped">stopped at 1,226 (by hand)</td>' in page, page
        # The quantized study's budget is 800 steps; the gzipped log reads like a plain one.
        assert '<td>Qwen3-30B-A3B-Base, INTELLECT math, INT8 sampler, INT4 KV cache</td><td>SFC+TIS</td><td>0</td><td class="st-done">done at 800</td>' in page, page
        assert '<td>Qwen3-30B-A3B-Base, INTELLECT math, FP8 sampler</td><td>SFC+KL, β₀ = 0.03</td><td>0</td><td class="st-running">running, step 300 of 800</td><td>0.620</td>' in page, page
        assert "built from the results folder" in page and ".st-collapsed" in page and ".setting-first" in page
        # Within a setting the rows follow ORDER: SFC+TIS before SFC+KL.
        rows = [line for line in page.splitlines() if line.startswith("<tr")]
        lst4 = [r for r in rows if "Countdown, 4-prompt slices</td>" in r or ("Countdown" not in r and "<td></td><td>SFC" in r)]
        assert lst4[0].count("SFC+TIS") == 1 and "<td>SFC+KL</td>" in lst4[1], lst4[:2]
    print("ok: experiments-at-a-glance table from synthetic runs")


def shown_runs_test():
    """shown_runs returns exactly the runs the table shows, in table order,
    with the rows the table reads, so the page's charts follow the table."""
    with tempfile.TemporaryDirectory() as root:
        write(root, "Lst4-tis-sc-s0", 3300, 0.0, eval_at={3200}, stopped="budget fixed at 3,200 steps: read this run at step 3,200 and ignore later steps.")
        write(root, "Lst4-sc-klr0.3-s0", 900, 0.75, eval_at={0, 400, 800})
        write(root, "Lat4-ppo-s0", 500, 0.3)
        write(root, "Qf8-tis-sc-s0", 120, 0.3, gz=True)
        shown = at_a_glance.shown_runs(root)
        assert [r["run"] for r in shown] == ["Lst4-tis-sc-s0", "Lst4-sc-klr0.3-s0", "Lat4-ppo-s0", "Qf8-tis-sc-s0"], [r["run"] for r in shown]
        assert shown[0]["setting"] == "Countdown, 4-prompt slices" and shown[0]["method"] == "SFC+TIS" and shown[0]["seed"] == 0
        assert shown[0]["status"] == "done at 3,200" and len(shown[0]["rows"]) == 3201, (shown[0]["status"], len(shown[0]["rows"]))
        assert shown[2]["status"] == "running, step 500 of 3,200" and len(shown[2]["rows"]) == 500
        assert shown[3]["status"] == "running, step 120 of 800" and len(shown[3]["rows"]) == 120
    print("ok: shown runs")


def names():
    assert at_a_glance.parse_name("Lst4b16k-tis-sc-s0") == ("Lst4b16k", "Countdown, 4-prompt slices, context length 16,384", "SFC+TIS", 0, 3200)
    assert at_a_glance.parse_name("Qf8k4-sc-klr0.03-s0")[1:3] == ("Qwen3-30B-A3B-Base, INTELLECT math, FP8 sampler, FP4 KV cache", "SFC+KL, β₀ = 0.03")
    assert at_a_glance.parse_name("Lglma1-sc-klr0.3-s1")[3] == 1
    assert at_a_glance.parse_name("Lst4-sc-klr0.3-k8-t8-s0")[2] == "SFC+KL, top-8 + 8 tail samples"
    assert at_a_glance.parse_name("Lst4-topr-sc-klr0.3-s0")[2] == "TOPR + SFC+KL"
    assert at_a_glance.parse_name("Tpay-sc-klr0.3-k8-t8-s0") is None, "timing probes are not table rows"
    assert at_a_glance.parse_name("Lst4-sc-klr0.3-s0.partial-x") is None
    labels = [label for _, label in at_a_glance.METHODS]
    assert len(set(labels)) == len(labels), labels
    print("ok: run names and labels")


if __name__ == "__main__":
    main()
    shown_runs_test()
    names()
