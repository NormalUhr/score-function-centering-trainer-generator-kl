/* Chartkit: self-contained interactive line charts for HTML pages (no dependencies,
 * CSP-safe when inlined). Interaction contract:
 *   1. Legends BELOW the figure, one row per series, each row led by a swatch showing the
 *      exact hue + dash of its curve.
 *   2. Click a curve or its legend row -> hide that curve. Double-click a curve or legend
 *      row -> solo it (everything else hides); double-click the solo'd series again ->
 *      show all. Clicking a hidden series' legend row shows it again.
 *   3. Hidden series keep their legend row visible but faded (still clickable).
 *   4. On every show/hide, BOTH axes refit tightly to the visible curves, unless the user
 *      holds a manual view (5).
 *   5. Left-click-drag draws a translucent selection rectangle (live preview); on release
 *      the view zooms to it and becomes MANUAL: show/hide no longer changes the range.
 *      Double-click on empty plot area returns to auto-fit of the currently visible curves.
 *   6. Right-click-drag pans a manual view (zoom ratio unchanged).
 *   7. Charts pair with a stats table below (the table is the page's job, not chartkit's).
 *   8. Optional per-chart tools above the figure: a smoothing switch with a
 *      window slider (a centered moving average over the raw points, applied client-side,
 *      on by default when the payload asks for it), and an "expand" button that opens the
 *      same chart in a floating window with every interaction intact; closing the window
 *      hands its state (hidden curves, smoothing, zoom) back to the inline chart.
 *   9. Hover: the cursor on a curve pops up the series' name (and id when it
 *      differs) with the nearest point's x and y; the hovered curve is drawn thicker.
 *
 * Usage: emit  <div class="ck-chart" data-ck="{escaped JSON}"></div>  with payload
 *   {series:[{id, label, pts:[[x,y],...]}], xlabel, ylabel, ydigits,
 *    smooth: {on: true, window: 10, max: 200} (optional), expand: true (optional)}
 * and inline this file once; charts boot on DOMContentLoaded (or call Chartkit.boot()).
 * Theming: hues come from --s1..--s4 (falling back to the validated Okabe-Ito subset);
 * chrome uses --ink/--muted/--line. >4 series cycle the 4 hues with dash patterns, so
 * every series stays identifiable by its swatch.
 */
(function (root) {
  "use strict";

  // ---------- pure geometry/scale helpers (exported for headless tests) ----------
  function fitRange(series, hidden) {
    var xs = [], ys = [];
    series.forEach(function (s) {
      if (hidden.has(s.id)) return;
      s.pts.forEach(function (p) { xs.push(p[0]); ys.push(p[1]); });
    });
    if (!xs.length) return null; // caller keeps previous range
    var x0 = Math.min.apply(null, xs), x1 = Math.max.apply(null, xs);
    var y0 = Math.min.apply(null, ys), y1 = Math.max.apply(null, ys);
    if (x0 === x1) { x0 -= 0.5; x1 += 0.5; }
    if (y0 === y1) { y0 -= Math.abs(y0) * 0.05 + 1e-9; y1 += Math.abs(y1) * 0.05 + 1e-9; }
    var px = (x1 - x0) * 0.02, py = (y1 - y0) * 0.04; // slim pad so strokes are not clipped
    return { x0: x0 - px, x1: x1 + px, y0: y0 - py, y1: y1 + py };
  }

  function niceTicks(lo, hi, n) {
    n = n || 5;
    var span = hi - lo;
    if (!(span > 0) || !isFinite(span)) return [lo];
    var raw = span / n;
    var mag = Math.pow(10, Math.floor(Math.log(raw) / Math.LN10));
    var step = [1, 2, 5, 10].map(function (m) { return m * mag; })
      .reduce(function (a, b) { return Math.abs(b - raw) < Math.abs(a - raw) ? b : a; });
    var t0 = Math.ceil(lo / step) * step, out = [];
    for (var t = t0; t <= hi + step * 1e-6; t += step) out.push(Math.abs(t) < step * 1e-9 ? 0 : t);
    return out;
  }

  function normRect(ax, ay, bx, by) {
    return { x0: Math.min(ax, bx), x1: Math.max(ax, bx),
             y0: Math.min(ay, by), y1: Math.max(ay, by) };
  }

  function fmt(v, digits) {
    if (!isFinite(v)) return "";
    var a = Math.abs(v);
    if (a >= 1e6 || (a > 0 && a < 1e-3)) return v.toExponential(Math.max(1, (digits || 4) - 2));
    if (a >= 1000 && Number.isInteger(v)) return String(v);
    return v.toFixed(digits == null ? 4 : digits);
  }

  // Centered moving average over `window` points (x kept; the window shrinks at the
  // edges to the available neighbours, so the curve keeps its length and its ends; an
  // even window is centered like the next odd one, half = floor(window / 2) each side).
  function smoothPts(pts, window) {
    var w = Math.max(1, Math.floor(window || 1));
    if (w === 1 || pts.length < 2) return pts;
    var half = Math.floor(w / 2), out = new Array(pts.length), sum = 0, lo = 0, hi = -1;
    for (var i = 0; i < pts.length; i++) {
      var wantLo = Math.max(0, i - half), wantHi = Math.min(pts.length - 1, i + half);
      while (hi < wantHi) { hi++; sum += pts[hi][1]; }
      while (lo < wantLo) { sum -= pts[lo][1]; lo++; }
      out[i] = [pts[i][0], sum / (hi - lo + 1)];
    }
    return out;
  }

  // Index of the point whose x is nearest to `x` (pts sorted by x).
  function nearestIndex(pts, x) {
    var lo = 0, hi = pts.length - 1;
    if (hi < 0) return -1;
    while (hi - lo > 1) {
      var mid = (lo + hi) >> 1;
      if (pts[mid][0] < x) lo = mid; else hi = mid;
    }
    return (Math.abs(pts[hi][0] - x) < Math.abs(pts[lo][0] - x)) ? hi : lo;
  }

  // ---------- styling ----------
  var HUES = ["var(--s1,#0072B2)", "var(--s2,#E69F00)", "var(--s3,#009E73)", "var(--s4,#D55E00)"];
  var DASH = ["", "7 4", "2 3", "9 3 2 3"]; // solid, dashed, dotted, dash-dot
  function styleOf(i) { return { hue: HUES[i % 4], dash: DASH[Math.floor(i / 4) % 4] }; }

  var CSS = [
    ".ck-chart{margin:6px 0 2px;position:relative}",
    ".ck-tip{position:absolute;pointer-events:none;z-index:5;background:var(--panel,#fff);color:var(--ink,#222);" +
      "border:1px solid var(--line,#8886);border-radius:4px;padding:4px 8px;font-size:12px;" +
      "font-family:var(--mono,ui-monospace,monospace);white-space:nowrap;box-shadow:0 4px 14px rgba(0,0,0,.18)}",
    ".ck-tip .ck-tip-name{font-weight:600}",
    ".ck-chart .ck-hover .ck-line{stroke-width:3.2}",
    // user-select:none: a marquee drag must never highlight axis-tick text
    ".ck-chart svg.ck-plot{width:100%;height:auto;display:block;user-select:none;-webkit-user-select:none}",
    ".ck-chart .ck-hit{cursor:pointer}",
    ".ck-chart .ck-line{transition:opacity .15s;pointer-events:none}",
    ".ck-chart .ck-gone .ck-line,.ck-chart .ck-gone .ck-dots{opacity:0}",
    ".ck-chart .ck-gone .ck-hit{pointer-events:none}",
    // Entries flow into as many columns as the width allows; each entry is an unbreakable
    // one-line unit (swatch + name + value never wrap internally).
    ".ck-legend{display:flex;flex-wrap:wrap;column-gap:22px;row-gap:2px;margin:4px 0 8px;" +
      "font-size:12.5px}",
    ".ck-leg{display:flex;align-items:center;gap:8px;cursor:pointer;padding:1px 4px;" +
      "border-radius:4px;transition:opacity .15s;user-select:none;white-space:nowrap}",
    ".ck-leg:hover{background:color-mix(in srgb,currentColor 8%,transparent)}",
    ".ck-leg.ck-off{opacity:.35}",
    ".ck-leg .ck-name{font-family:var(--mono,ui-monospace,monospace)}",
    ".ck-leg .ck-last{color:var(--muted,#888);font-family:var(--mono,ui-monospace,monospace)}",
    // tools above the figure: smoothing switch + window slider, expand button
    ".ck-tools{display:flex;flex-wrap:wrap;align-items:center;gap:6px 14px;margin:0 0 4px;" +
      "font-size:12px;color:var(--muted,#888);font-family:var(--mono,ui-monospace,monospace)}",
    ".ck-tools label{display:flex;align-items:center;gap:5px;cursor:pointer;white-space:nowrap}",
    ".ck-tools input[type=range]{width:140px;accent-color:var(--accent,#0f6e7a)}",
    ".ck-tools button{font:inherit;color:inherit;background:transparent;border:1px solid var(--line,#8886);" +
      "border-radius:4px;padding:1px 8px;cursor:pointer;margin-left:auto}",
    ".ck-tools button:hover{background:color-mix(in srgb,currentColor 8%,transparent)}",
    // the floating window of an expanded chart
    ".ck-overlay{position:fixed;inset:0;z-index:1000;background:rgba(0,0,0,.45);display:flex;" +
      "align-items:center;justify-content:center;padding:24px}",
    ".ck-modal{background:var(--panel,var(--bg,#fff));color:var(--ink,#222);border:1px solid var(--line,#8886);" +
      "border-radius:8px;width:min(96vw,1400px);max-height:94vh;overflow:auto;padding:12px 18px 14px;" +
      "box-shadow:0 12px 40px rgba(0,0,0,.35)}",
    ".ck-modal-head{display:flex;align-items:center;justify-content:space-between;gap:12px;margin-bottom:4px}",
    ".ck-modal-head .ck-title{font-size:14px;color:var(--muted,#888)}",
    ".ck-modal-head button{font:inherit;color:inherit;background:transparent;border:1px solid var(--line,#8886);" +
      "border-radius:4px;padding:2px 10px;cursor:pointer}",
    ".ck-modal .ck-chart svg.ck-plot{height:auto}"
  ].join("\n");

  function ensureCss() {
    if (document.getElementById("ck-style")) return;
    var st = document.createElement("style");
    st.id = "ck-style";
    st.textContent = CSS;
    document.head.appendChild(st);
  }

  // ---------- chart ----------
  var W = 1000, H = 380, L = 64, R = 16, T = 12, B = 40;
  var uid = 0;

  // `init` (optional) seeds the state: {hidden: Set, view: range|null, smooth: {on, window}},
  // which is how an expanded copy starts where the inline chart is and hands its state back.
  function render(el, init) {
    var spec;
    try { spec = JSON.parse(el.getAttribute("data-ck")); } catch (e) { return; }
    if (!spec || !spec.series || !spec.series.length) return;
    ensureCss();
    var id = "ck" + (++uid);
    var smoothSpec = spec.smooth || null;
    var state = { hidden: new Set(init && init.hidden ? init.hidden : []),
                  view: init && init.view ? Object.assign({}, init.view) : null, range: null,
                  smooth: { on: !!(smoothSpec && smoothSpec.on !== false),
                            window: smoothSpec ? (smoothSpec.window || 1) : 1 } };
    if (init && init.smooth) state.smooth = { on: !!init.smooth.on, window: init.smooth.window || 1 };
    var series = spec.series.map(function (s, i) {
      var raw = s.pts.slice().sort(function (a, b) { return a[0] - b[0]; });
      return { id: s.id, label: s.label || s.id, raw: raw, pts: raw, style: styleOf(i) };
    });

    function applySmoothing() {
      var w = state.smooth.on ? state.smooth.window : 1;
      series.forEach(function (s) { s.pts = smoothPts(s.raw, w); });
    }
    applySmoothing();

    var ns = "http://www.w3.org/2000/svg";
    el.textContent = "";

    var tools = null, winInput = null, winLabel = null, onInput = null;
    if (smoothSpec || spec.expand) {
      tools = document.createElement("div");
      tools.className = "ck-tools";
      if (smoothSpec) {
        var max = smoothSpec.max || 200;
        tools.innerHTML =
          '<label><input type="checkbox" class="ck-sm-on"> smooth</label>' +
          '<input type="range" class="ck-sm-win" min="1" max="' + max + '" step="1">' +
          '<span class="ck-sm-val"></span>';
        onInput = tools.querySelector(".ck-sm-on");
        winInput = tools.querySelector(".ck-sm-win");
        winLabel = tools.querySelector(".ck-sm-val");
        onInput.checked = state.smooth.on;
        winInput.value = Math.min(max, state.smooth.window);
        winInput.disabled = !state.smooth.on;
        onInput.addEventListener("change", function () {
          state.smooth.on = onInput.checked;
          winInput.disabled = !state.smooth.on;
          resmooth();
        });
        winInput.addEventListener("input", function () {
          state.smooth.window = parseInt(winInput.value, 10) || 1;
          resmooth();
        });
      }
      if (spec.expand) {
        var btn = document.createElement("button");
        btn.type = "button";
        btn.className = "ck-expand";
        btn.textContent = "expand";
        btn.addEventListener("click", function () { expand(); });
        tools.appendChild(btn);
      }
      el.appendChild(tools);
    }

    var svg = document.createElementNS(ns, "svg");
    svg.setAttribute("viewBox", "0 0 " + W + " " + H);
    svg.setAttribute("role", "img");
    svg.setAttribute("class", "ck-plot");
    el.appendChild(svg);

    var tip = document.createElement("div");
    tip.className = "ck-tip";
    tip.style.display = "none";
    el.appendChild(tip);
    var hovered = null;

    var legend = document.createElement("div");
    legend.className = "ck-legend";
    el.appendChild(legend);
    series.forEach(function (s) {
      var row = document.createElement("div");
      row.className = "ck-leg";
      row.setAttribute("data-id", s.id);
      row.innerHTML =
        '<svg width="34" height="12" aria-hidden="true"><path d="M2 6 H32" fill="none" ' +
        'stroke="' + s.style.hue + '" stroke-width="2.4"' +
        (s.style.dash ? ' stroke-dasharray="' + s.style.dash + '"' : "") + "/></svg>" +
        '<span class="ck-name"></span><span class="ck-last"></span>';
      row.querySelector(".ck-name").textContent = s.label;
      legend.appendChild(row);
      row.addEventListener("click", function () { seriesClick(s.id, "leg"); });
      row.addEventListener("dblclick", function (ev) { ev.preventDefault(); });
    });

    function updateLegendValues() {
      series.forEach(function (s) {
        var row = legend.querySelector('.ck-leg[data-id="' + cssEsc(s.id) + '"]');
        if (!row) return;
        var last = s.pts[s.pts.length - 1];
        row.querySelector(".ck-last").textContent = last ? "→ " + fmt(last[1], spec.ydigits) : "";
      });
      if (winLabel) winLabel.textContent = "window " + state.smooth.window + (state.smooth.on ? "" : " (off)");
    }

    function resmooth() {
      applySmoothing();
      if (!state.view) refit();
      draw();
    }

    // Custom double-click detection: the OS-native dblclick window (~500 ms) promotes two
    // deliberate single clicks into an accidental solo. Two clicks on the
    // SAME target within DBL_MS count as a double; the pre-first-click hidden-set snapshot
    // is restored before applying the double action, so the first click's toggle leaves no
    // residue and "double-click the solo'd series restores all" evaluates against the state
    // the user actually saw.
    var lastClick = { key: null, t: 0, x: 0, y: 0, snap: null };

    function isDouble(key, x, y) {
      var now = Date.now(), win = api.dblMs || 260;
      var hit = lastClick.key === key && (now - lastClick.t) < win &&
        (x == null || (Math.abs(x - lastClick.x) < 14 && Math.abs(y - lastClick.y) < 14));
      if (hit) {
        var snap = lastClick.snap;
        lastClick = { key: null, t: 0, x: 0, y: 0, snap: null };
        return { snap: snap };
      }
      lastClick = { key: key, t: now, x: x || 0, y: y || 0,
                    snap: new Set(state.hidden) };
      return null;
    }

    function seriesClick(sid, kind) {
      var dbl = isDouble(kind + ":" + sid);
      if (dbl) {
        state.hidden = dbl.snap; // revert the first click's toggle
        solo(sid);
      } else {
        toggle(sid);
      }
    }

    function refit() {
      var r = fitRange(series, state.hidden);
      if (r) state.range = r;
    }

    function toggle(sid) {
      if (state.hidden.has(sid)) state.hidden.delete(sid); else state.hidden.add(sid);
      if (!state.view) refit();
      draw();
    }

    function solo(sid) {
      var already = !state.hidden.has(sid) && state.hidden.size === series.length - 1;
      state.hidden = new Set();
      if (!already) series.forEach(function (s) { if (s.id !== sid) state.hidden.add(s.id); });
      if (!state.view) refit();
      draw();
    }

    // coordinate transforms
    function sx(r, x) { return L + (x - r.x0) / (r.x1 - r.x0) * (W - L - R); }
    function sy(r, y) { return T + (r.y1 - y) / (r.y1 - r.y0) * (H - T - B); }
    function dx(r, px) { return r.x0 + (px - L) / (W - L - R) * (r.x1 - r.x0); }
    function dy(r, py) { return r.y1 - (py - T) / (H - T - B) * (r.y1 - r.y0); }

    function svgPoint(ev) {
      var pt = svg.createSVGPoint();
      pt.x = ev.clientX; pt.y = ev.clientY;
      var m = svg.getScreenCTM();
      if (!m) return { x: 0, y: 0 };
      var p = pt.matrixTransform(m.inverse());
      return { x: p.x, y: p.y };
    }

    function draw() {
      var r = state.view || state.range;
      if (!r) { refit(); r = state.range; }
      if (!r) return;
      var parts = [];
      parts.push('<defs><clipPath id="' + id + 'c"><rect x="' + L + '" y="' + T +
        '" width="' + (W - L - R) + '" height="' + (H - T - B) + '"/></clipPath></defs>');
      // plot background (the drag/dblclick surface)
      parts.push('<rect class="ck-bg" x="' + L + '" y="' + T + '" width="' + (W - L - R) +
        '" height="' + (H - T - B) + '" fill="transparent"/>');
      niceTicks(r.y0, r.y1).forEach(function (tv) {
        var y = sy(r, tv);
        parts.push('<line x1="' + L + '" y1="' + y + '" x2="' + (W - R) + '" y2="' + y +
          '" stroke="var(--line,#8886)" stroke-width="0.5" opacity="0.55"/>');
        parts.push('<text x="' + (L - 7) + '" y="' + (y + 3.5) + '" text-anchor="end" font-size="11"' +
          ' fill="var(--muted,#888)" font-family="var(--mono,monospace)">' + fmt(tv, spec.ydigits) + "</text>");
      });
      niceTicks(r.x0, r.x1, 6).forEach(function (tv) {
        var x = sx(r, tv);
        parts.push('<line x1="' + x + '" y1="' + (H - B) + '" x2="' + x + '" y2="' + (H - B + 4) +
          '" stroke="var(--line,#8886)"/>');
        parts.push('<text x="' + x + '" y="' + (H - B + 16) + '" text-anchor="middle" font-size="11"' +
          ' fill="var(--muted,#888)" font-family="var(--mono,monospace)">' +
          fmt(tv, Number.isInteger(tv) ? 0 : 2) + "</text>");
      });
      parts.push('<line x1="' + L + '" y1="' + (H - B) + '" x2="' + (W - R) + '" y2="' + (H - B) +
        '" stroke="var(--line,#888)"/>');
      if (spec.xlabel) parts.push('<text x="' + (L + (W - L - R) / 2) + '" y="' + (H - 6) +
        '" text-anchor="middle" font-size="11.5" fill="var(--muted,#888)">' + esc(spec.xlabel) + "</text>");
      if (spec.ylabel) parts.push('<text x="14" y="' + (T + (H - T - B) / 2) +
        '" text-anchor="middle" font-size="11.5" fill="var(--muted,#888)" transform="rotate(-90 14 ' +
        (T + (H - T - B) / 2) + ')">' + esc(spec.ylabel) + "</text>");

      series.forEach(function (s) {
        var d = s.pts.map(function (p, j) {
          return (j ? "L" : "M") + sx(r, p[0]).toFixed(2) + " " + sy(r, p[1]).toFixed(2);
        }).join(" ");
        var gone = state.hidden.has(s.id) ? " ck-gone" : "";
        var hover = hovered === s.id ? " ck-hover" : "";
        parts.push('<g class="ck-series' + gone + hover + '" data-id="' + esc(s.id) + '" clip-path="url(#' + id + 'c)">' +
          '<path class="ck-line" d="' + d + '" fill="none" stroke="' + s.style.hue + '" stroke-width="1.9"' +
          (s.style.dash ? ' stroke-dasharray="' + s.style.dash + '"' : "") + "/>" +
          '<path class="ck-hit" d="' + d + '" fill="none" stroke="transparent" stroke-width="11"/>' +
          '<circle class="ck-mark" r="4" fill="' + s.style.hue + '" style="display:none"/></g>');
      });
      svg.innerHTML = parts.join("");
      legend.querySelectorAll(".ck-leg").forEach(function (row) {
        row.classList.toggle("ck-off", state.hidden.has(row.getAttribute("data-id")));
      });
      updateLegendValues();
    }

    function esc(t) {
      return String(t).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/"/g, "&quot;");
    }

    function cssEsc(t) {
      return (root && root.CSS && root.CSS.escape) ? root.CSS.escape(t) : String(t).replace(/["\\]/g, "\\$&");
    }

    // ---------- the floating window ----------
    function getState() {
      return { hidden: new Set(state.hidden), view: state.view ? Object.assign({}, state.view) : null,
               smooth: { on: state.smooth.on, window: state.smooth.window } };
    }

    function setState(next) {
      state.hidden = new Set(next.hidden || []);
      state.view = next.view ? Object.assign({}, next.view) : null;
      if (next.smooth) {
        state.smooth = { on: !!next.smooth.on, window: next.smooth.window || 1 };
        if (onInput) onInput.checked = state.smooth.on;
        if (winInput) { winInput.value = state.smooth.window; winInput.disabled = !state.smooth.on; }
      }
      applySmoothing();
      if (!state.view) refit();
      draw();
    }

    function expand() {
      var overlay = document.createElement("div");
      overlay.className = "ck-overlay";
      var modal = document.createElement("div");
      modal.className = "ck-modal";
      var head = document.createElement("div");
      head.className = "ck-modal-head";
      var title = document.createElement("span");
      title.className = "ck-title";
      title.textContent = (spec.ylabel || "") + (spec.xlabel ? " against " + spec.xlabel : "");
      var close = document.createElement("button");
      close.type = "button";
      close.textContent = "close";
      head.appendChild(title);
      head.appendChild(close);
      var big = document.createElement("div");
      big.className = "ck-chart";
      // the copy carries the expand flag off: one floating window at a time
      var copy = Object.assign({}, spec, { expand: false });
      big.setAttribute("data-ck", JSON.stringify(copy));
      modal.appendChild(head);
      modal.appendChild(big);
      overlay.appendChild(modal);
      document.body.appendChild(overlay);
      render(big, getState());
      function finish() {
        if (big._ck) setState(big._ck.getState());
        overlay.remove();
        document.removeEventListener("keydown", onKey);
      }
      function onKey(ev) { if (ev.key === "Escape") finish(); }
      close.addEventListener("click", finish);
      overlay.addEventListener("click", function (ev) { if (ev.target === overlay) finish(); });
      document.addEventListener("keydown", onKey);
    }

    el._ck = { getState: getState, setState: setState };

    // ---------- pointer interactions ----------
    var drag = null;   // {x,y, rect|null} left-drag zoom
    var pan = null;    // {x,y, view} right-drag pan
    var suppressClick = false;

    svg.addEventListener("mousedown", function (ev) {
      var p = svgPoint(ev);
      if (ev.button === 0) {
        drag = { x: p.x, y: p.y, rect: null };
        ev.preventDefault(); // suppress native text selection of tick labels during drag
      } else if (ev.button === 2 && state.view) {
        pan = { x: p.x, y: p.y, view: Object.assign({}, state.view) };
        ev.preventDefault();
      }
    });

    window.addEventListener("mousemove", function (ev) {
      if (drag) {
        var p = svgPoint(ev);
        if (!drag.rect && (Math.abs(p.x - drag.x) > 4 || Math.abs(p.y - drag.y) > 4)) {
          drag.rect = document.createElementNS(ns, "rect");
          drag.rect.setAttribute("fill", "color-mix(in srgb, var(--s1,#0072B2) 14%, transparent)");
          drag.rect.setAttribute("stroke", "var(--muted,#888)");
          drag.rect.setAttribute("stroke-dasharray", "4 3");
          svg.appendChild(drag.rect);
        }
        if (drag.rect) {
          var q = normRect(drag.x, drag.y, p.x, p.y);
          drag.rect.setAttribute("x", q.x0); drag.rect.setAttribute("y", q.y0);
          drag.rect.setAttribute("width", q.x1 - q.x0); drag.rect.setAttribute("height", q.y1 - q.y0);
        }
      } else if (pan) {
        var p2 = svgPoint(ev);
        var r = pan.view;
        var ddx = (p2.x - pan.x) / (W - L - R) * (r.x1 - r.x0);
        var ddy = (p2.y - pan.y) / (H - T - B) * (r.y1 - r.y0);
        state.view = { x0: r.x0 - ddx, x1: r.x1 - ddx, y0: r.y0 + ddy, y1: r.y1 + ddy };
        draw();
      }
    });

    window.addEventListener("mouseup", function (ev) {
      if (drag) {
        var had = !!drag.rect;
        if (had) {
          var p = svgPoint(ev);
          var q = normRect(drag.x, drag.y, p.x, p.y);
          drag.rect.remove();
          if (q.x1 - q.x0 > 8 && q.y1 - q.y0 > 8) {
            var r = state.view || state.range;
            state.view = { x0: dx(r, q.x0), x1: dx(r, q.x1), y0: dy(r, q.y1), y1: dy(r, q.y0) };
            draw();
          }
          suppressClick = true;
          setTimeout(function () { suppressClick = false; }, 0);
        }
        drag = null;
      }
      if (pan) pan = null;
    });

    svg.addEventListener("click", function (ev) {
      if (suppressClick) return;
      var g = ev.target.closest ? ev.target.closest(".ck-series") : null;
      if (g && ev.target.classList.contains("ck-hit")) {
        seriesClick(g.getAttribute("data-id"), "curve");
      } else {
        var p = svgPoint(ev);
        if (isDouble("bg", p.x, p.y)) {
          state.view = null;  // back to auto-fit of the currently visible curves
          refit();
          draw();
        }
      }
    });

    // Hover: the series whose nearest-x point is closest to the cursor (within a
    // vertical band) is highlighted and its name and value pop up by the cursor.
    function hideTip() {
      if (hovered !== null) {
        hovered = null;
        svg.querySelectorAll(".ck-series.ck-hover").forEach(function (g) { g.classList.remove("ck-hover"); });
        svg.querySelectorAll(".ck-mark").forEach(function (c) { c.style.display = "none"; });
      }
      tip.style.display = "none";
    }

    svg.addEventListener("mousemove", function (ev) {
      if (drag || pan) { hideTip(); return; }
      var r = state.view || state.range;
      if (!r) return;
      var p = svgPoint(ev);
      if (p.x < L || p.x > W - R || p.y < T || p.y > H - B) { hideTip(); return; }
      var x = dx(r, p.x), best = null;
      series.forEach(function (s) {
        if (state.hidden.has(s.id) || !s.pts.length) return;
        var i = nearestIndex(s.pts, x), q = s.pts[i];
        var dyPx = Math.abs(sy(r, q[1]) - p.y), dxPx = Math.abs(sx(r, q[0]) - p.x);
        var dist = Math.sqrt(dyPx * dyPx + dxPx * dxPx);
        if (dist <= 28 && (!best || dist < best.dist)) best = { s: s, q: q, dist: dist };
      });
      if (!best) { hideTip(); return; }
      if (hovered !== best.s.id) {
        svg.querySelectorAll(".ck-series.ck-hover").forEach(function (g) { g.classList.remove("ck-hover"); });
        var g = svg.querySelector('.ck-series[data-id="' + cssEsc(best.s.id) + '"]');
        if (g) g.classList.add("ck-hover");
        hovered = best.s.id;
      }
      svg.querySelectorAll(".ck-mark").forEach(function (c) { c.style.display = "none"; });
      var mark = svg.querySelector('.ck-series[data-id="' + cssEsc(best.s.id) + '"] .ck-mark');
      if (mark) {
        mark.setAttribute("cx", sx(r, best.q[0]).toFixed(2));
        mark.setAttribute("cy", sy(r, best.q[1]).toFixed(2));
        mark.style.display = "";
      }
      tip.innerHTML = '<div class="ck-tip-name"></div><div class="ck-tip-id"></div><div class="ck-tip-val"></div>';
      tip.querySelector(".ck-tip-name").textContent = best.s.label;
      var idRow = tip.querySelector(".ck-tip-id");
      if (best.s.id !== best.s.label) idRow.textContent = best.s.id; else idRow.remove();
      tip.querySelector(".ck-tip-val").textContent =
        (spec.xlabel ? spec.xlabel + " " : "x ") + fmt(best.q[0], Number.isInteger(best.q[0]) ? 0 : 2) +
        ", " + (spec.ylabel ? spec.ylabel + " " : "y ") + fmt(best.q[1], spec.ydigits);
      // place the tip by the cursor, inside the chart element, flipping left near the right edge
      var box = el.getBoundingClientRect();
      var left = ev.clientX - box.left + 14, top = ev.clientY - box.top - 10;
      tip.style.display = "";
      if (left + tip.offsetWidth > box.width - 4) left = ev.clientX - box.left - tip.offsetWidth - 14;
      tip.style.left = Math.max(0, left) + "px";
      tip.style.top = Math.max(0, top) + "px";
    });
    svg.addEventListener("mouseleave", hideTip);

    // Native dblclick is intentionally inert (custom detection above owns the semantics).
    svg.addEventListener("dblclick", function (ev) { ev.preventDefault(); });

    svg.addEventListener("contextmenu", function (ev) { ev.preventDefault(); });

    if (!state.view) refit();
    draw();
  }

  function boot(scope) {
    (scope || document).querySelectorAll(".ck-chart[data-ck]").forEach(function (el) {
      if (!el._ck) render(el);
    });
  }

  var api = { boot: boot, render: render,
              _test: { fitRange: fitRange, niceTicks: niceTicks, normRect: normRect, fmt: fmt, smoothPts: smoothPts, nearestIndex: nearestIndex } };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  if (root && root.document) {
    root.Chartkit = api;
    if (root.document.readyState === "loading")
      root.document.addEventListener("DOMContentLoaded", function () { boot(); });
    else boot();
  }
})(typeof window !== "undefined" ? window : null);
