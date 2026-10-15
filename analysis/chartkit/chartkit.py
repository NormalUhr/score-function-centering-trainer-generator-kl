"""Python builder for Chartkit charts (see chartkit.js for the interaction contract).

    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from chartkit import chart_div, inline_js

    html = inline_js()                       # once per page, in <head> or before first chart
    html += chart_div(
        [{"id": "run-a", "label": "run-a (formal name)", "pts": [(0, 3.1), (100, 2.9)]},
         {"id": "run-b", "label": "run-b", "pts": [(0, 3.2), (100, 3.0)]}],
        xlabel="training step", ylabel="validation loss", ydigits=4,
        smooth={"on": True, "window": 10, "max": 200},   # optional: smoothing switch + window slider
        expand=True)                                       # optional: "expand" button (floating window)

Series ids must be unique within a chart. Points are (x, y) pairs; they are sorted by x at
render time. Colors/dashes are assigned by series order (4 validated hues x 4 dashes).
With `smooth`, pass the RAW points: the chart applies a centered moving average over
`window` points client-side, and the viewer changes the window with the slider.
"""
import html as _html
import json
import os

_DIR = os.path.dirname(os.path.abspath(__file__))


def inline_js():
    with open(os.path.join(_DIR, "chartkit.js")) as f:
        return "<script>\n" + f.read() + "\n</script>"


def chart_div(series, xlabel="", ylabel="", ydigits=4, smooth=None, expand=False):
    for s in series:
        s["pts"] = [[float(x), float(y)] for x, y in s["pts"]]
    payload = {"series": series, "xlabel": xlabel, "ylabel": ylabel, "ydigits": ydigits}
    if smooth:
        payload["smooth"] = {"on": bool(smooth.get("on", True)), "window": int(smooth.get("window", 1)),
                             "max": int(smooth.get("max", 200))}
    if expand:
        payload["expand"] = True
    return ('<div class="ck-chart" data-ck="'
            + _html.escape(json.dumps(payload), quote=True) + '"></div>')
