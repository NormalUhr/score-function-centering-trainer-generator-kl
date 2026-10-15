"""Fixture test for the results page builder: the table fragment, the payload
timing table, then two charts per setting (training reward and evaluation
reward against the step) for exactly the runs the table shows."""
import gzip
import html as html_module
import json
import os
import re
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import at_a_glance
import progress_page


def write_run(root, name, rows, stopped=None, gz=False):
    run_dir = Path(root) / name
    run_dir.mkdir(parents=True)
    text = "".join(json.dumps(row) + "\n" for row in rows)
    if gz:
        with gzip.open(run_dir / "logs.jsonl.gz", "wt") as f:
            f.write(text)
    else:
        (run_dir / "logs.jsonl").write_text(text)
    if stopped:
        (run_dir / "STOPPED").write_text(stopped)


def rows_of(n, reward, evals):
    rows = []
    for step in range(n):
        row = {"step": step, "reward": reward + 0.0001 * step, "frac_truncated": 0.2}
        if step in evals:
            row["eval_reward"] = evals[step]
        rows.append(row)
    return rows


def charts_of(page):
    body = page.split("</script>", 1)[1]  # the inlined chart library mentions the class too
    payload = html_module.unescape(body)
    return [json.loads(c.split('"></div>', 1)[0]) for c in payload.split('<div class="ck-chart" data-ck="')[1:]]


def main():
    with tempfile.TemporaryDirectory() as tmp:
        outputs = Path(tmp) / "runs"
        write_run(outputs, "Lst4-tis-sc-s0", rows_of(3300, 0.5, {0: 0.1, 400: 0.6, 800: 0.65, 3200: 0.0}),
                  stopped="budget fixed at 3,200 steps: read this run at step 3,200 and ignore later steps.")
        write_run(outputs, "Lst4-sc-klr0.3-s0", rows_of(900, 0.7, {0: 0.1, 400: 0.68, 800: 0.7}), gz=True)
        write_run(outputs, "Lat4-ppo-s0", rows_of(500, 0.3, {0: 0.1, 400: 0.4}))
        # The context-length study: the three SFC+TIS runs share one chart pair.
        write_run(outputs, "Lst4b8k-tis-sc-s0", rows_of(1024, 0.45, {0: 0.1, 400: 0.64, 800: 0.3}), stopped="stopped by the collapse rule; read at the stopped length.")
        write_run(outputs, "Lst4b16k-tis-sc-s0", rows_of(600, 0.6, {0: 0.1, 400: 0.58}))
        glance = at_a_glance.build(outputs, "test build")
        page = progress_page.build(outputs, glance, updated="2026-10-15 01:00 UTC", title="SFC+KL: experiments")
        assert "<title>SFC+KL: experiments</title>" in page and "2026-10-15 01:00 UTC" in page
        assert "Experiments at a glance" in page and 'id="timing"' in page and 'id="curves"' in page
        assert page.index('id="glance"') < page.index('id="timing"') < page.index('id="curves"')
        body = page.split("</script>", 1)[1]
        # Three chart pairs: the 4,096-token setting, the context-length group, AdamW.
        assert body.count('<div class="ck-chart"') == 6, body.count('<div class="ck-chart"')
        curves = body.split('id="curves"', 1)[1]
        headings = re.findall(r"<h3>([^<]*)</h3>", curves)
        assert headings == ["Countdown, 4-prompt slices",
                            "Countdown, 4-prompt slices, SFC+TIS at context lengths 4,096, 8,192 and 16,384",
                            "AdamW, 4-prompt slices"], headings
        payload = html_module.unescape(body)
        assert '"id": "Lst4-tis-sc-s0"' in payload and '"id": "Lat4-ppo-s0"' in payload
        assert '"label": "SFC+TIS, seed 0"' in payload and '"label": "SFC+KL, seed 0"' in payload, "the page's labels"
        charts = charts_of(page)
        evals = [c for c in charts if c["ylabel"] == "evaluation reward"]
        tis = next(s for chart in evals for s in chart["series"] if s["id"] == "Lst4-tis-sc-s0")
        assert tis["pts"] == [[0.0, 0.1], [400.0, 0.6], [800.0, 0.65], [3200.0, 0.0]], tis["pts"]
        ours = next(s for chart in evals for s in chart["series"] if s["id"] == "Lst4-sc-klr0.3-s0")
        assert ours["pts"] == [[0.0, 0.1], [400.0, 0.68], [800.0, 0.7]], "the gzipped log is charted"
        trains = [c for c in charts if c["ylabel"] == "training reward"]
        tis_train = next(s for chart in trains for s in chart["series"] if s["id"] == "Lst4-tis-sc-s0")
        assert len(tis_train["pts"]) == 3201 and tis_train["pts"][0] == [0.0, 0.5] and tis_train["pts"][-1][0] == 3200.0
        assert trains[0]["smooth"] == {"on": True, "window": 100, "max": 200} and trains[0]["expand"] is True
        assert evals[0]["smooth"] == {"on": False, "window": 3, "max": 20} and evals[0]["expand"] is True
        group = [c for c in charts if any(s["id"] == "Lst4b16k-tis-sc-s0" for s in c["series"])]
        assert len(group) == 2, len(group)
        labels = sorted(s["label"] for s in group[0]["series"])
        assert labels == ["SFC+TIS, 16,384 tokens, seed 0", "SFC+TIS, 4,096 tokens, seed 0", "SFC+TIS, 8,192 tokens, seed 0"], labels
        assert "no timing probes" in page
    print("ok: results page fixture")


def thinning():
    """Long curves are thinned to a bounded number of points; the first and
    last steps survive and values carry four decimals."""
    rows = [{"step": s, "reward": 0.123456789} for s in range(2000)]
    pts = progress_page.training_points(rows, max_points=400)
    assert len(pts) <= 400 and pts[0][0] == 0 and pts[-1][0] == 1999 and pts[0][1] == 0.1235, (len(pts), pts[0], pts[-1])
    assert len(progress_page.training_points(rows, max_points=4000)) == 2000
    print("ok: curve thinning")


def timing_probes():
    """The payload timing table reads the Tpay probe runs (not table rows),
    plain or gzipped, and reports per payload the rollout size, the
    generator's time per stage, the trainer's compute per step, and the
    rollout read and write times."""
    with tempfile.TemporaryDirectory() as tmp:
        outputs = Path(tmp) / "runs"

        def probe(name, head, tail, compute, stage_s, mb, gz):
            d = outputs / name
            d.mkdir(parents=True)
            cfg = {"sampler": {"vocab_logprobs": head, "tail_samples": tail}, "rl": {"seq_len": 4096}}
            (d / "config.json").write_text(json.dumps(cfg))
            steps = "".join(json.dumps({"step": s, "elapsed_s": s * (compute + 2.0), "wait_s": 2.0,
                                        "rollout_read_s": 0.3}) + "\n" for s in range(100))
            batches = "".join(json.dumps({"batch": b, "stages": 10, "decode_s": 10 * stage_s,
                                          "rollout_mb": mb, "push_s": 0.5}) + "\n" for b in range(3))
            if gz:
                with gzip.open(d / "logs.jsonl.gz", "wt") as f:
                    f.write(steps)
                with gzip.open(d / "generator-0.jsonl.gz", "wt") as f:
                    f.write(batches)
            else:
                (d / "logs.jsonl").write_text(steps)
                (d / "generator-0.jsonl").write_text(batches)
        probe("Tpay-sc-klr0.3-k128-s0", 128, 0, 1.3, 5.0, 2000.0, gz=True)
        probe("Tpay-sc-klr0.3-k16-s0", 16, 0, 1.2, 4.0, 250.0, gz=False)
        probe("Tpay-sc-klr0.3-k8-t8-s0", 8, 8, 1.2, 4.1, 250.0, gz=True)
        table = progress_page.timing_table(outputs)
        assert "<table" in table and "top-128" in table and "top-8 + 8 tail samples" in table
        assert "1.30" in table and "1.20" in table and "0.30" in table and "0.50" in table
        assert "5.00" in table and "4.10" in table and "2000" in table and "250" in table
        assert "<tr><td>top-128</td><td>2000</td><td>5.00</td><td>1.30</td><td>0.30</td><td>0.50</td></tr>" in table, table
        order = re.findall(r"<tr><td>([^<]*)</td>", table)
        assert order == ["top-128", "top-16", "top-8 + 8 tail samples"], order  # width descending, all head first
        assert "no timing probes" in progress_page.timing_table(Path(tmp) / "empty")
        assert at_a_glance.shown_runs(outputs) == [], "probes are not table rows"
    print("ok: timing probes table")


if __name__ == "__main__":
    main()
    thinning()
    timing_probes()
