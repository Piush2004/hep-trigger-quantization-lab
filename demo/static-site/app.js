/* HEP-Trigger Quantization Lab — result explorer.
 * Vanilla JS + Plotly. Reads only /api/* (exported, frozen results). No training, no new numbers.
 * Missing values are shown as "not recorded"; nothing is estimated in the browser.
 */
(function () {
  "use strict";

  const TIER = {
    MEASURED: { cls: "measured", text: "PyTorch evaluation on held-out test split" },
    SIMULATED: { cls: "simulated", text: "hls4ml C-simulation (fixed-point arithmetic only)" },
    ESTIMATED: { cls: "estimated", text: "software hardware-resource heuristic; not validated by synthesis" },
    "NOT AVAILABLE": { cls: "na", text: "no tool or device in the experimental environment" },
  };
  const PRECISIONS = ["FP32", "INT8", "INT6", "INT4"];
  const ARCH_CONFIGS = ["FP32", "INT8 PTQ", "INT6 PTQ", "INT4 PTQ", "INT4 QAT"];
  const COLORS = { cuda: "#4FA6FF", cpu: "#B9A6F7", b: "#D9A441", mixed: "#B9A6F7", grid: "#262E42", text: "#EDF1F9", muted: "#93A1BD" };

  const $ = (sel) => document.querySelector(sel);
  // All data comes from this app's own exported JSON (never third-party or user input), but values
  // are still escaped before going into innerHTML, as defense-in-depth against a future data change.
  const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const fmt = (x, d = 4) => (x === null || x === undefined || Number.isNaN(x) ? "not recorded" : Number(x).toFixed(d));
  const signed = (x, d = 4) => (x === null || x === undefined ? "—" : (x >= 0 ? "+" : "") + Number(x).toFixed(d));
  const kb = (bytes) => {
    if (bytes === null || bytes === undefined) return "not recorded";
    const v = bytes / 1000;
    return (v < 10 ? v.toFixed(2) : v.toFixed(1)) + " KB";
  };
  const badge = (tier) => {
    const t = TIER[tier] || { cls: "na", text: tier };
    return `<span class="badge ${t.cls}" title="${t.text}">${tier}</span>`;
  };
  const getJSON = (url) => fetch(url).then((r) => {
    if (!r.ok) throw new Error(url + " → " + r.status);
    return r.json();
  });
  // Same app.js serves two deployments: the live FastAPI app (/api/<name>) and a static mirror for
  // GitHub Pages, which has no backend and instead ships pre-rendered ./data/<name>.json snapshots
  // (built by demo/scripts/build_static_site.py, which calls the exact same server-side route
  // functions so the two are byte-identical in shape and filtering). window.STATIC_MODE is injected
  // by the static build only.
  const apiUrl = (name) => (window.STATIC_MODE ? `./data/${name}.json` : `/api/${name}`);

  // Label a row the same way across the CUDA (baseline) and CPU (architecture) sources.
  const label = (r) => r.configuration || (r.method === "none" ? r.precision : `${r.precision} ${r.method}`);

  const baseLayout = (extra) => Object.assign({
    paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
    font: { family: "IBM Plex Sans, Segoe UI, Roboto, sans-serif", color: COLORS.text, size: 12 },
    margin: { l: 64, r: 24, t: 24, b: 56 },
    xaxis: { gridcolor: COLORS.grid, zerolinecolor: COLORS.grid, linecolor: COLORS.grid },
    yaxis: { gridcolor: COLORS.grid, zerolinecolor: COLORS.grid, linecolor: COLORS.grid },
    legend: { bgcolor: "rgba(0,0,0,0)", font: { color: COLORS.muted } },
    hoverlabel: { bgcolor: "#1A2133", bordercolor: COLORS.cuda, font: { family: "IBM Plex Mono, Menlo, monospace", color: COLORS.text } },
    // Named motion: chart-state transitions answer the user's own action (segmented control /
    // select change) rather than decorating — see reports/design_brief.md §4.
    transition: { duration: 250, easing: "cubic-in-out" },
  }, extra || {});
  const plotCfg = { responsive: true, displayModeBar: false };
  const plot = (id, traces, layout, ariaLabel) => {
    const el = $("#" + id);
    if (!window.Plotly) {
      el.innerHTML = '<p class="muted">The chart library did not load (offline or blocked CDN); the numbers above and the surrounding text still apply.</p>';
      el.setAttribute("role", "note");
      return;
    }
    // Plotly.react diffs against the existing plot (and is a safe no-op on first render), which is
    // what lets layout.transition above animate marker/bar moves instead of hard-redrawing.
    window.Plotly.react(id, traces, baseLayout(layout), plotCfg).then(() => {
      el.setAttribute("role", "img");
      if (ariaLabel) el.setAttribute("aria-label", ariaLabel);
    });
  };

  /* ---------- 1. main result ---------- */
  function renderMain(baseline) {
    const cuda = baseline.rows.filter((r) => r.architecture.startsWith("A primary"));
    const wanted = ["FP32", "INT8 PTQ", "INT6 PTQ", "INT4 PTQ", "INT4 QAT"];
    let pts = cuda.filter((r) => wanted.includes(r.configuration));
    const mixed = cuda.find((r) => r.configuration.includes("size-constrained"));
    if (mixed) pts.push(mixed);
    pts.sort((a, b) => a.storage_bytes - b.storage_bytes);

    const xs = pts.map((r) => r.storage_bytes);
    const ys = pts.map((r) => r.roc_auc_mean);
    const es = pts.map((r) => r.roc_auc_sd);
    const custom = pts.map((r) => [
      r.configuration.includes("size-constrained") ? "Mixed (validation-selected, ≤ 3,724 B)" : r.configuration,
      r.roc_auc_sd, kb(r.storage_bytes), r.delta_vs_fp32, r.device, r.n,
    ]);
    const isINT6 = pts.map((r) => r.configuration === "INT6 PTQ");
    const size = pts.map((r) => (r.configuration === "INT6 PTQ" ? 18 : 11));
    const colour = pts.map((r) => (r.configuration.includes("size-constrained") ? COLORS.mixed : r.configuration.includes("QAT") ? COLORS.cuda : COLORS.cpu));

    plot("chart-main", [{
      x: xs, y: ys, error_y: { type: "data", array: es, visible: true, color: COLORS.muted, thickness: 1 },
      mode: window.innerWidth < 600 ? "markers" : "markers+text", type: "scatter",
      text: pts.map((r) => (r.configuration === "INT6 PTQ" ? "INT6 PTQ ★" : r.configuration.replace("size-constrained", "").replace("(, ", "(").trim())),
      textposition: "top center", textfont: { size: 11, color: COLORS.muted },
      marker: { size: size, color: colour, line: { color: COLORS.text, width: isINT6.map((b) => (b ? 2 : 0)) } },
      customdata: custom,
      hovertemplate: "<b>%{customdata[0]}</b><br>ROC-AUC %{y:.4f} ± %{customdata[1]:.4f}<br>Storage %{customdata[2]}<br>Δ FP32 %{customdata[3]:+.4f}<br>Device: %{customdata[4]}<br>n = %{customdata[5]}<extra></extra>",
      name: "configurations",
    }], {
      xaxis: { title: "Theoretical weight storage (bytes, log scale; analytical)", type: "log", gridcolor: COLORS.grid },
      yaxis: { title: "Test macro ROC-AUC (mean ± SD)", gridcolor: COLORS.grid, tickformat: ".3f" },
      showlegend: false,
    }, "Scatter chart of test ROC-AUC against theoretical weight storage for each precision and method. See the headline text below for the key numbers.");

    const fp = cuda.find((r) => r.configuration === "FP32");
    const int6 = cuda.find((r) => r.configuration === "INT6 PTQ");
    if (fp && int6) {
      const red = (100 * (1 - int6.storage_bytes / fp.storage_bytes)).toFixed(1);
      $("#headline").innerHTML =
        `<strong>INT6 PTQ</strong> reaches ROC-AUC ${fmt(int6.roc_auc_mean)} ± ${fmt(int6.roc_auc_sd)} ` +
        `(Δ FP32 ${signed(int6.delta_vs_fp32)}) at ${kb(int6.storage_bytes)} theoretical weight storage — ` +
        `${red}% below FP32 (${kb(fp.storage_bytes)}). Accuracy is ${badge("MEASURED")}; storage is analytical (theoretical, not file size).`;
    }
  }

  /* ---------- 2. precision explorer ---------- */
  const explorer = { precision: "INT6", method: "PTQ" };
  function renderExplorer(baseline, quant) {
    const seg = (el, opts, key, onPick) => {
      el.innerHTML = opts.map((o) => `<button type="button" data-v="${o.v}" aria-pressed="${o.v === explorer[key]}" ${o.disabled ? "disabled" : ""}>${o.t}</button>`).join("");
      el.querySelectorAll("button").forEach((b) => b.addEventListener("click", () => { explorer[key] = b.dataset.v; onPick(); }));
    };
    const update = () => {
      seg($("#seg-precision"), PRECISIONS.map((p) => ({ v: p, t: p === "FP32" ? "FP32" : p, disabled: false })), "precision", update);
      const isFP = explorer.precision === "FP32";
      if (isFP) explorer.method = "none";
      if (!isFP && explorer.method === "none") explorer.method = "PTQ";
      seg($("#seg-method"), [{ v: "PTQ", t: "PTQ", disabled: isFP }, { v: "QAT", t: "QAT", disabled: isFP }], "method", update);

      const key = isFP ? "FP32" : `${explorer.precision} ${explorer.method}`;
      const row = baseline.rows.find((r) => r.architecture.startsWith("A primary") && r.configuration === key);
      const m = $("#metrics");
      if (!row) {
        m.innerHTML = `<div><dt>configuration</dt><dd class="na">not recorded for this combination</dd></div>`;
        $("#explorer-note").textContent = "No frozen record exists for this combination.";
        $("#chart-seeds").innerHTML = "";
        return;
      }
      m.innerHTML = [
        ["ROC-AUC (mean ± SD)", `${fmt(row.roc_auc_mean)} ± ${fmt(row.roc_auc_sd)}`],
        ["PR-AUC", fmt(row.pr_auc_mean)],
        ["Macro F1", fmt(row.f1_mean)],
        ["Accuracy", fmt(row.accuracy_mean)],
        ["Storage (theoretical)", kb(row.storage_bytes)],
        ["Δ ROC-AUC vs FP32", row.delta_vs_fp32 === null ? "—" : signed(row.delta_vs_fp32)],
        ["Seed range (min–max)", `${fmt(row.roc_auc_min)} – ${fmt(row.roc_auc_max)}`],
      ].map(([k, v]) => `<div><dt>${k}</dt><dd>${v}</dd></div>`).join("");
      $("#explorer-note").innerHTML =
        `${badge("MEASURED")} CUDA-era frozen record, n = ${row.n}. Storage is analytical (theoretical), not a measurement and not file size. ` +
        `The INT4 values are device-sensitive; see the architecture section for the CPU re-run.`;

      const q = quant.per_seed[key === "FP32" ? "FP32" : key];
      if (q) {
        plot("chart-seeds", [{
          x: q.map((r) => "seed " + r.seed), y: q.map((r) => r.roc_auc), type: "bar",
          marker: { color: COLORS.cuda }, hovertemplate: "%{x}<br>ROC-AUC %{y:.4f}<extra></extra>",
        }], {
          yaxis: { title: "Test ROC-AUC per seed", tickformat: ".3f", gridcolor: COLORS.grid },
          xaxis: { gridcolor: COLORS.grid },
          title: { text: "Seed-to-seed variance", font: { size: 13, color: COLORS.muted } },
        });
      } else {
        $("#chart-seeds").innerHTML = '<p class="muted">Per-seed values are not exported for this configuration.</p>';
      }
    };
    update();
  }

  /* ---------- 3. PTQ vs QAT (both devices, separate) ---------- */
  function renderPtqQat(arch) {
    const groups = [];
    arch.panels.filter((p) => p.architecture === "A [64,32,32]").forEach((p) => groups.push(p));
    const order = ["FP32", "INT4 PTQ", "INT4 QAT"];
    const traces = [];
    const recovery = [];
    groups.forEach((p, gi) => {
      const rows = order.map((c) => p.rows.find((r) => label(r) === c)).filter(Boolean);
      traces.push({
        name: p.label, type: "bar",
        x: rows.map((r) => label(r)), y: rows.map((r) => r.roc_auc_mean),
        error_y: { type: "data", array: rows.map((r) => r.roc_auc_sd), color: COLORS.muted },
        marker: { color: gi === 0 ? COLORS.cuda : COLORS.cpu },
        customdata: rows.map((r) => [r.roc_auc_sd, p.device]),
        hovertemplate: "<b>%{x}</b><br>%{y:.4f} ± %{customdata[0]:.4f}<br>%{customdata[1]}<extra>" + p.label + "</extra>",
      });
      const fp = rows.find((r) => label(r) === "FP32"), ptq = rows.find((r) => label(r) === "INT4 PTQ"), qat = rows.find((r) => label(r) === "INT4 QAT");
      if (fp && ptq && qat) {
        const share = (qat.roc_auc_mean - ptq.roc_auc_mean) / (fp.roc_auc_mean - ptq.roc_auc_mean);
        recovery.push(`<strong>${esc(p.device)}</strong>: QAT recovered ≈${(100 * share).toFixed(0)}% of the INT4 PTQ loss ` +
          `(INT4 PTQ ${fmt(ptq.roc_auc_mean)} → INT4 QAT ${fmt(qat.roc_auc_mean)}, FP32 ${fmt(fp.roc_auc_mean)}).`);
      }
    });
    plot("chart-ptq-qat", traces, { barmode: "group", yaxis: { title: "Test macro ROC-AUC (mean ± SD)", tickformat: ".3f", gridcolor: COLORS.grid, range: [0.85, 0.95] }, legend: { orientation: "h", y: 1.12 } });
    $("#recovery").innerHTML = recovery.join("<br>") +
      `<br><span class="muted">The two devices are reported separately. The difference between them is observational: ` +
      `a controlled CUDA reproduction of the Phase 6 protocol was unavailable, so device dependence is not established.</span>`;
  }

  /* ---------- 4. architecture (A CUDA | A CPU | B CPU, side by side) ---------- */
  function renderArchitecture(arch) {
    const sel = $("#sel-arch-config");
    sel.innerHTML = ARCH_CONFIGS.map((c) => `<option value="${c}">${c}</option>`).join("");
    sel.value = "INT4 QAT";
    const grid = $("#arch-grid");
    grid.innerHTML = arch.panels.map((p, i) => `
      <article class="arch-card">
        <h3>${esc(p.label)}</h3>
        <p class="device">device: ${esc(p.device)} · ${Number(p.parameters).toLocaleString("en-GB")} parameters</p>
        <div class="chart" id="arch-chart-${i}"></div>
        <dl class="metrics" id="arch-metrics-${i}" style="margin-top:10px"></dl>
      </article>`).join("");
    $("#arch-caveat").innerHTML = `<strong>Observational comparison.</strong> ${esc(arch.caveat)}`;

    const draw = () => {
      const chosen = sel.value;
      arch.panels.forEach((p, i) => {
        const rows = ARCH_CONFIGS.map((c) => p.rows.find((r) => label(r) === c)).filter(Boolean);
        plot(`arch-chart-${i}`, [{
          type: "bar", x: rows.map((r) => label(r)), y: rows.map((r) => r.roc_auc_mean),
          error_y: { type: "data", array: rows.map((r) => r.roc_auc_sd), color: COLORS.muted },
          marker: { color: rows.map((r) => (label(r) === chosen ? COLORS.cuda : "#3A4560")) },
          hovertemplate: "<b>%{x}</b><br>%{y:.4f}<extra></extra>",
        }], {
          margin: { l: 56, r: 12, t: 12, b: 70 },
          yaxis: { tickformat: ".3f", gridcolor: COLORS.grid, range: [0.85, 0.95] },
          xaxis: { tickangle: -25, gridcolor: COLORS.grid },
          showlegend: false,
        });
        const r = p.rows.find((x) => label(x) === chosen);
        $(`#arch-metrics-${i}`).innerHTML = r ? [
          ["ROC-AUC", `${fmt(r.roc_auc_mean)} ± ${fmt(r.roc_auc_sd)}`],
          ["Δ vs own FP32", r.delta_vs_own_fp32_mean !== undefined ? signed(r.delta_vs_own_fp32_mean) : (r.delta_vs_fp32 === null || r.delta_vs_fp32 === undefined ? "—" : signed(r.delta_vs_fp32))],
          ["Storage (theoretical)", kb(r.storage_bytes)],
        ].map(([k, v]) => `<div><dt>${k}</dt><dd>${v}</dd></div>`).join("") : '<div><dd class="na">not recorded</dd></div>';
      });
    };
    sel.addEventListener("change", draw);
    draw();
  }

  /* ---------- 5. calibration ---------- */
  function renderCalibration(cal) {
    const series = [["INT8", COLORS.cuda], ["INT6", COLORS.cpu], ["INT4", COLORS.b]];
    const traces = series.map(([p, c]) => {
      const rs = cal.rows.filter((r) => r.precision === p).sort((a, b) => a.calib_samples - b.calib_samples);
      return {
        type: "scatter", mode: "lines+markers", name: p + " PTQ",
        x: rs.map((r) => r.calib_samples), y: rs.map((r) => r.roc_auc_mean),
        error_y: { type: "data", array: rs.map((r) => r.roc_auc_sd), color: COLORS.muted },
        line: { color: c }, marker: { color: c },
        hovertemplate: "%{x} calibration jets<br>ROC-AUC %{y:.4f} ± %{error_y.array:.4f}<extra>" + p + " PTQ</extra>",
      };
    });
    plot("chart-calib", traces, {
      xaxis: { title: "Calibration set size (training jets, log scale)", type: "log", gridcolor: COLORS.grid },
      yaxis: { title: "Test macro ROC-AUC (mean ± SD)", tickformat: ".3f", gridcolor: COLORS.grid },
      legend: { orientation: "h", y: 1.12 },
    });
  }

  /* ---------- 6. hardware evidence ---------- */
  function renderHardware(hls) {
    const rows = hls.rows;
    const fp = rows.find((r) => r.stage === "PyTorch FP32");
    const w16 = rows.find((r) => r.stage === "hls4ml C-simulation" && r.width_W === 16 && r.mode.startsWith("default"));
    const steps = [
      { t: "PyTorch FP32", tier: "MEASURED", note: `accuracy ${fmt(fp.accuracy)}, ROC-AUC ${fmt(fp.roc_auc)}` },
      { t: "PyTorch quantized model (INT8 / INT6 / INT4, PTQ and QAT)", tier: "MEASURED", note: "fake-quantized in float32; ROC-AUC per configuration in the Results section" },
      { t: "hls4ml conversion, fixed-point ap_fixed<W,6>", tier: "SIMULATED", note: w16 ? `C-simulation at W = 16: accuracy ${fmt(w16.accuracy)}, agreement with FP32 ${fmt(w16.agreement_with_fp32)}` : "C-simulation rows in the chart below" },
      { t: "HLS synthesis (resources, latency, II)", tier: "NOT AVAILABLE", note: "HLS SYNTHESIS NOT AVAILABLE IN EXPERIMENTAL ENVIRONMENT" },
      { t: "Physical FPGA measurement", tier: "NOT AVAILABLE", note: "no FPGA device or measurement exists" },
    ];
    $("#pipeline").innerHTML = steps.map((s) => `
      <li class="${s.tier === "NOT AVAILABLE" ? "unavailable" : ""}">
        <div>${s.t}<span class="note">${s.note}</span></div>
        <div>${badge(s.tier)}</div>
      </li>`).join("");

    const modes = {};
    rows.filter((r) => r.stage === "hls4ml C-simulation").forEach((r) => {
      (modes[r.mode] = modes[r.mode] || []).push(r);
    });
    const palette = [COLORS.b, COLORS.cpu, COLORS.cuda, COLORS.mixed];
    const traces = Object.keys(modes).map((m, i) => {
      const rs = modes[m].sort((a, b) => a.width_W - b.width_W);
      return {
        type: "scatter", mode: "lines+markers", name: m + " (SIMULATED)",
        x: rs.map((r) => r.width_W), y: rs.map((r) => r.accuracy),
        line: { color: palette[i % palette.length] }, marker: { color: palette[i % palette.length] },
        hovertemplate: "W = %{x}<br>accuracy %{y:.4f}<extra>" + m + "</extra>",
      };
    });
    if (fp) {
      traces.push({
        type: "scatter", mode: "lines", name: "PyTorch FP32 (MEASURED)",
        x: [8, 16], y: [fp.accuracy, fp.accuracy], line: { color: COLORS.muted, dash: "dash" },
        hoverinfo: "skip",
      });
    }
    plot("chart-hls", traces, {
      xaxis: { title: "ap_fixed total width W (6 integer bits)", gridcolor: COLORS.grid, dtick: 2 },
      yaxis: { title: "Accuracy on 15,000 held-out jets", tickformat: ".2f", gridcolor: COLORS.grid, range: [0, 0.85] },
      legend: { orientation: "h", y: -0.28, font: { size: 11 } },
      margin: { l: 64, r: 24, t: 24, b: 110 },
    }, "Line chart of hls4ml C-simulation accuracy against fixed-point width, for four arithmetic modes, against a dashed PyTorch FP32 reference line. Values are in the table rows above.");
  }

  /* ---------- evidence tier legend ---------- */
  function renderLegend() {
    const el = $("#tier-legend");
    if (!el) return; // only present on the Methodology page
    el.innerHTML = Object.keys(TIER).map((k) =>
      `<span>${badge(k)} <span class="tierdesc">${TIER[k].text}</span></span>`).join("");
  }

  /* ---------- metadata ---------- */
  function renderMeta(meta) {
    if (meta.repository_url) {
      ["#nav-github", "#hero-github"].forEach((sel) => {
        const a = $(sel);
        if (!a) return; // #hero-github only exists on the home page
        a.href = meta.repository_url; a.hidden = false;
        if (sel === "#nav-github") a.textContent = "GitHub";
      });
    }
  }

  /* ---------- nav active state ---------- */
  function markActiveNav(page) {
    document.querySelectorAll(".topbar nav a[data-page]").forEach((a) => {
      if (a.dataset.page === page) a.setAttribute("aria-current", "page");
    });
  }

  /* ---------- per-page boot ----------
   * The site is a set of separate static pages (one per feature) sharing this one app.js. Each page
   * declares which it is via <body data-page="...">; this table says which /api/* endpoints that page
   * needs and which renderers to run once they've loaded, so a page only fetches what it shows. */
  const PAGE_ENDPOINTS = {
    home: ["metadata"],
    results: ["metadata", "results", "quantization", "architecture-comparison"],
    architecture: ["metadata", "architecture-comparison"],
    hardware: ["metadata", "calibration-sensitivity", "hls4ml"],
    methodology: ["metadata"],
    research: ["metadata"],
  };
  const PAGE_RENDER = {
    results: (data) => {
      if (data.results) {
        const baseline = { rows: data.results.rows };
        renderMain(baseline);
        if (data.quantization) renderExplorer(baseline, data.quantization);
      }
      if (data["architecture-comparison"]) renderPtqQat(data["architecture-comparison"]);
    },
    architecture: (data) => {
      if (data["architecture-comparison"]) renderArchitecture(data["architecture-comparison"]);
    },
    hardware: (data) => {
      if (data["calibration-sensitivity"]) renderCalibration(data["calibration-sensitivity"]);
      if (data.hls4ml) renderHardware(data.hls4ml);
    },
  };

  /* ---------- boot ---------- */
  function showLoadError(sectionName, err) {
    const note = document.createElement("p");
    note.className = "caveat";
    note.setAttribute("role", "alert");
    note.textContent = `Could not load "${sectionName}": ${err.message}. ` +
      (navigator.onLine ? "Try reloading the page." : "You appear to be offline; reconnect and reload.");
    document.querySelector("main").prepend(note);
  }

  function boot() {
    renderLegend();
    const page = document.body.dataset.page || "home";
    markActiveNav(page);
    // allSettled: one failing endpoint (e.g. offline) degrades only its own section, never blanks the page.
    const endpoints = PAGE_ENDPOINTS[page] || ["metadata"];
    Promise.allSettled(endpoints.map((name) => getJSON(apiUrl(name)))).then((outcomes) => {
      const data = {};
      outcomes.forEach((o, i) => {
        const name = endpoints[i];
        if (o.status === "fulfilled") data[name] = o.value;
        else showLoadError(name, o.reason);
      });
      if (data.metadata) renderMeta(data.metadata);
      const render = PAGE_RENDER[page];
      if (render) render(data);
    });
  }

  function registerServiceWorker() {
    if (!("serviceWorker" in navigator)) return;
    // Secure-context only by browser rule (localhost is exempt); fails silently and harmlessly elsewhere.
    const swPath = window.STATIC_MODE ? "./sw.js" : "/sw.js";
    navigator.serviceWorker.register(swPath).catch(() => { /* offline support is a progressive enhancement */ });
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", boot);
  else boot();
  window.addEventListener("load", registerServiceWorker);
})();
