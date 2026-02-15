"""Generate self-contained HTML reports with embedded charts and KPI tables."""

from __future__ import annotations

import base64
import contextlib
import io
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ab_harness.compare import RunComparison
from ab_harness.kpi import (
    SampleKpis,
    extract_timeseries,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Chart generation (returns base64 PNG)
# ---------------------------------------------------------------------------


def _fig_to_base64(fig: plt.Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def _make_timeseries_chart(
    title: str,
    ylabel: str,
    series_a: dict[int, list[SampleKpis]],
    series_b: dict[int, list[SampleKpis]] | None,
    field: str,
) -> str:
    """Create a timeseries chart with one line per rep, A vs B."""
    fig, ax = plt.subplots(figsize=(12, 4))

    # Plot B (baseline) in blue
    if series_b:
        for rep_idx, kpis in series_b.items():
            times = [k.t_rel_s for k in kpis]
            values = [getattr(k, field) for k in kpis]
            values = [v if v is not None else float("nan") for v in values]
            ax.plot(
                times,
                values,
                color="#4A90D9",
                alpha=0.4,
                linewidth=1,
                label=f"baseline rep {rep_idx}" if rep_idx == 1 else None,
            )

    # Plot A (iteration) in red/orange
    for rep_idx, kpis in series_a.items():
        times = [k.t_rel_s for k in kpis]
        values = [getattr(k, field) for k in kpis]
        values = [v if v is not None else float("nan") for v in values]
        ax.plot(
            times,
            values,
            color="#E74C3C",
            alpha=0.6,
            linewidth=1,
            label=f"iteration rep {rep_idx}" if rep_idx == 1 else None,
        )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.3)

    return _fig_to_base64(fig)


# ---------------------------------------------------------------------------
# HTML template
# ---------------------------------------------------------------------------

_VERDICT_COLORS = {
    "regression": "#E74C3C",
    "likely_regression": "#F39C12",
    "neutral": "#95A5A6",
    "likely_improvement": "#27AE60",
    "improvement": "#2ECC71",
    "no_data": "#BDC3C7",
    "topology_changed": "#8E44AD",
}

_VERDICT_ICONS = {
    "regression": "&#x2716;",  # X
    "likely_regression": "&#x26A0;",  # Warning
    "neutral": "&#x2014;",  # Em dash
    "likely_improvement": "&#x25B2;",  # Triangle up
    "improvement": "&#x2714;",  # Checkmark
    "no_data": "?",
    "topology_changed": "&#x21C4;",  # Left-right arrows
}


def _fmt(val: float | None, decimals: int = 2) -> str:
    if val is None:
        return "N/A"
    return f"{val:.{decimals}f}"


def _fmt_pct(val: float | None) -> str:
    if val is None:
        return "N/A"
    return f"{val:+.1f}%"


def _fmt_pval(val: float | None) -> str:
    if val is None:
        return "N/A"
    if val < 0.001:
        return "<0.001"
    return f"{val:.3f}"


def _fmt_ci(lo: float | None, hi: float | None) -> str:
    if lo is None or hi is None:
        return "N/A"
    return f"[{lo:+.2f}, {hi:+.2f}]"


def _fmt_power(val: float | None) -> str:
    if val is None:
        return "N/A"
    return f"{val:.2f}"


_EFFECT_COLORS = {
    "negligible": "#95A5A6",
    "small": "#F39C12",
    "medium": "#E67E22",
    "large": "#E74C3C",
    "N/A": "#BDC3C7",
}


def generate_report(
    comparison: RunComparison,
    iteration_dir: Path,
    baseline_dir: Path,
    metadata: dict[str, Any] | None = None,
    prev_comparison: RunComparison | None = None,
) -> Path:
    """Generate a self-contained HTML report and write to iteration_dir/report.html."""

    # Load timeseries for charts
    try:
        series_a = extract_timeseries(iteration_dir)
        series_b = extract_timeseries(baseline_dir)
    except Exception as exc:
        log.warning("Could not load timeseries for charts: %s", exc)
        series_a, series_b = {}, None

    # Generate charts
    charts: dict[str, str] = {}
    chart_configs = [
        ("Throughput (FPS)", "FPS", "throughput_fps"),
        ("Freshness Delay", "ms", "freshness_delay_ms"),
        ("Pipeline CPU", "%", "pipeline_cpu_pct"),
        ("System CPU", "%", "system_cpu_pct"),
        ("System Temperature", "C", "system_temperature_c"),
        ("System Memory", "%", "system_mem_used_pct"),
    ]
    for title, ylabel, field_name in chart_configs:
        try:
            charts[field_name] = _make_timeseries_chart(
                title, ylabel, series_a, series_b, field_name
            )
        except Exception as exc:
            log.debug("Chart %s failed: %s", field_name, exc)

    meta = metadata or {}
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Load baseline metadata for side-by-side SHA display
    baseline_meta: dict[str, Any] = {}
    baseline_meta_path = baseline_dir / "metadata.json"
    if baseline_meta_path.exists():
        with contextlib.suppress(Exception):
            baseline_meta = json.loads(baseline_meta_path.read_text())

    # Load system baseline if available
    exp_dir = baseline_dir.parent
    system_baseline: dict[str, Any] | None = None
    sb_path = exp_dir / "system_baseline.json"
    if sb_path.exists():
        with contextlib.suppress(Exception):
            system_baseline = json.loads(sb_path.read_text())

    html = _build_html(
        comparison,
        charts,
        meta,
        now,
        prev_comparison,
        system_baseline,
        baseline_meta,
    )

    out_path = iteration_dir / "report.html"
    out_path.write_text(html)
    log.info("Report written to %s", out_path)
    return out_path


def _build_html(
    comp: RunComparison,
    charts: dict[str, str],
    meta: dict[str, Any],
    generated_at: str,
    prev_comp: RunComparison | None,
    system_baseline: dict[str, Any] | None = None,
    baseline_meta: dict[str, Any] | None = None,
) -> str:
    """Build the full HTML string."""

    # KPI table rows
    kpi_rows = ""
    invalidated_names = set()
    if comp.topology_diff and comp.topology_diff.changed:
        invalidated_names = set(comp.topology_diff.invalidated_kpis)

    for c in comp.kpis:
        color = _VERDICT_COLORS.get(c.verdict, "#95A5A6")
        icon = _VERDICT_ICONS.get(c.verdict, "")
        es_label_for_color = c.cliffs_delta_label if c.normality_warning else c.effect_size_label
        es_color = _EFFECT_COLORS.get(es_label_for_color, "#BDC3C7")
        power_str = _fmt_power(c.power)
        power_warn = (
            ' <span title="Low power: need more reps">&#x26A0;</span>'
            if (c.power is not None and c.power < 0.8)
            else ""
        )
        topo_badge = ""
        row_style = ""
        if c.name in invalidated_names:
            topo_badge = ' <span style="background:#8E44AD;color:white;padding:1px 6px;border-radius:3px;font-size:0.75em;" title="Topology changed: comparison unreliable">TOPO</span>'
            row_style = ' style="opacity:0.7;"'
        diag_parts: list[str] = []
        if c.normality_warning:
            diag_parts.append(
                '<span style="background:#E67E22;color:white;padding:1px 5px;border-radius:3px;font-size:0.72em;" title="Shapiro-Wilk normality test failed; using non-parametric methods">NON-NORMAL</span>'
            )
        if c.outlier_count_a > 0 or c.outlier_count_b > 0:
            diag_parts.append(
                f'<span style="background:#E74C3C;color:white;padding:1px 5px;border-radius:3px;font-size:0.72em;" title="IQR outliers: A={c.outlier_count_a}, B={c.outlier_count_b}">OUTLIERS ({c.outlier_count_a + c.outlier_count_b})</span>'
            )
        diag_html = (
            " ".join(diag_parts)
            if diag_parts
            else '<span style="color:#95A5A6;font-size:0.8em;">ok</span>'
        )
        kpi_rows += f"""
        <tr{row_style}>
            <td><strong>{c.name}</strong>{topo_badge}</td>
            <td>{c.unit}</td>
            <td>{_fmt(c.b_mean)} &plusmn; {_fmt(c.b_std)}</td>
            <td>{_fmt(c.a_mean)} &plusmn; {_fmt(c.a_std)}</td>
            <td>{_fmt(c.delta)}</td>
            <td>{_fmt_pct(c.delta_pct)}</td>
            <td>{_fmt_pval(c.p_value)}{'<sup title="Mann-Whitney U (non-parametric)">MW</sup>' if c.test_used == "mann_whitney" else ""}</td>
            <td>{_fmt_ci(c.bootstrap_ci_lower, c.bootstrap_ci_upper) if c.normality_warning and c.bootstrap_ci_lower is not None else _fmt_ci(c.ci_lower, c.ci_upper)}{'<sup title="Bootstrap BCa CI">B</sup>' if c.normality_warning and c.bootstrap_ci_lower is not None else ""}</td>
            <td style="color:{es_color}; font-weight:bold;" title="Cohen's d = {_fmt(c.cohens_d)}, Cliff's delta = {_fmt(c.cliffs_delta)}">{c.cliffs_delta_label if c.normality_warning else c.effect_size_label}</td>
            <td>{power_str}{power_warn}</td>
            <td>{diag_html}</td>
            <td style="color:{color}; font-weight:bold;">{icon} {c.verdict}</td>
        </tr>"""

    # Per-rep breakdown
    per_rep_rows = ""
    for c in comp.kpis:
        if c.a_per_rep or c.b_per_rep:
            a_vals = ", ".join(_fmt(v) for v in c.a_per_rep) if c.a_per_rep else "N/A"
            b_vals = ", ".join(_fmt(v) for v in c.b_per_rep) if c.b_per_rep else "N/A"
            per_rep_rows += f"""
            <tr>
                <td>{c.name}</td>
                <td>{b_vals}</td>
                <td>{a_vals}</td>
            </tr>"""

    # Robust stats detail section
    robust_kpis = [c for c in comp.kpis if c.normality_warning]
    robust_html = ""
    if robust_kpis:
        robust_rows = ""
        for c in robust_kpis:
            robust_rows += f"""
            <tr>
                <td><strong>{c.name}</strong></td>
                <td>{_fmt_pval(c.shapiro_p_a)} / {_fmt_pval(c.shapiro_p_b)}</td>
                <td>{_fmt_pval(c.mann_whitney_p)}</td>
                <td>{_fmt(c.cliffs_delta, 3)}</td>
                <td>{c.cliffs_delta_label}</td>
                <td>{_fmt_ci(c.bootstrap_ci_lower, c.bootstrap_ci_upper)}</td>
                <td>{c.outlier_count_a} / {c.outlier_count_b}</td>
            </tr>"""

        robust_html = f"""
<div class="section">
<h2>Robust Statistics Details</h2>
<p style="font-size:0.85em; color:#666;">KPIs where the Shapiro-Wilk normality test was rejected (p &lt; 0.05). Non-parametric methods were used for these KPIs.</p>
<table>
<thead>
    <tr>
        <th>KPI</th>
        <th>Shapiro-Wilk p (A/B)</th>
        <th>Mann-Whitney p</th>
        <th>Cliff's delta</th>
        <th>Effect Size</th>
        <th>Bootstrap 95% CI</th>
        <th>IQR Outliers (A/B)</th>
    </tr>
</thead>
<tbody>{robust_rows}
</tbody>
</table>
</div>
"""

    # Charts HTML
    charts_html = ""
    for field_name, b64 in charts.items():
        charts_html += f"""
        <div class="chart">
            <img src="data:image/png;base64,{b64}" alt="{field_name}" />
        </div>"""

    # --- Metadata extraction ---
    b_meta = baseline_meta or {}
    label = meta.get("label", "N/A")
    experiment = meta.get("experiment", "N/A")
    duration = meta.get("duration_seconds", "?")
    reps = meta.get("repetitions", "?")
    diff_stat = meta.get("mcm_git_diff_stat", "")
    diff_full = meta.get("mcm_git_diff", "")
    is_interleaved = meta.get("interleaved", False)
    mode_str = "Interleaved ABBA" if is_interleaved else "Sequential"

    # Command line
    cmdline = meta.get("command_line")
    cmdline_str = " ".join(cmdline) if isinstance(cmdline, list) else (cmdline or "N/A")

    # MCM SHAs (iteration and baseline)
    iter_mcm_hash = meta.get("mcm_git_hash", "")
    iter_mcm_branch = meta.get("mcm_git_branch", "")
    base_mcm_hash = b_meta.get("mcm_git_hash", "")
    base_mcm_branch = b_meta.get("mcm_git_branch", "")

    # Harness SHAs
    harness_hash = meta.get("harness_git_hash", "")
    harness_branch = meta.get("harness_git_branch", "")
    harness_dirty = meta.get("harness_git_dirty", False)
    harness_dirty_str = " (dirty)" if harness_dirty else ""

    # Env overrides
    env_overrides = meta.get("env_overrides", {})
    env_str = (
        "<br>".join(f"{k}={v}" for k, v in env_overrides.items()) if env_overrides else "none"
    )

    # Extra MCM args
    extra_mcm_args = meta.get("extra_mcm_args") or []
    extra_args_str = " ".join(extra_mcm_args) if extra_mcm_args else "none"

    # Client config
    client_cfg = meta.get("client_config") or {}
    client_parts: list[str] = []
    for spec_str in client_cfg.get("webrtc") or []:
        client_parts.append(f"webrtc: {spec_str}")
    for spec_str in client_cfg.get("rtsp") or []:
        client_parts.append(f"rtsp: {spec_str}")
    for spec_str in client_cfg.get("thumbnails") or []:
        client_parts.append(f"thumbnail: {spec_str}")
    thumb_interval = client_cfg.get("thumbnail_interval")
    if thumb_interval and thumb_interval != 1 and client_cfg.get("thumbnails"):
        client_parts.append(f"thumbnail interval: {thumb_interval}s")
    client_str = "<br>".join(client_parts) if client_parts else "none"

    # Deploy mode & container
    deploy_mode = meta.get("deploy_mode", "?")
    host = meta.get("host", "?")
    port = meta.get("port", "?")
    blueos_core_ver = meta.get("blueos_core_version", "")
    blueos_bootstrap_ver = meta.get("blueos_bootstrap_version", "")

    # System baseline section
    sys_baseline_html = ""
    if system_baseline:
        idle_cpu = system_baseline.get("idle_cpu_pct")
        load_1m = system_baseline.get("baseline_load_1m")
        temp_c = system_baseline.get("baseline_temp_c")
        sys_baseline_html = f"""
    <div class="meta-item" style="grid-column: span 2; background: #EBF5FB;">
        <strong>System Baseline (pre-MCM idle):</strong>
        CPU: {_fmt(idle_cpu)}%
        &bull; Load 1m: {_fmt(load_1m)}
        &bull; Temp: {_fmt(temp_c)}&deg;C
    </div>"""

    # --- Topology section ---
    topology_html = ""
    topo_warning_html = ""
    if comp.topology_diff is not None:
        td = comp.topology_diff
        if td.changed:
            change_lines = td.details.split("; ") if td.details else []
            changes_list = "".join(f"<li>{_escape_html(c)}</li>" for c in change_lines)
            inv_list = "".join(
                f"<li><code>{_escape_html(k)}</code></li>" for k in td.invalidated_kpis
            )
            topo_warning_html = f"""
<div class="banner" style="background:#8E44AD;">
    &#x26A0; Topology changed between baseline and iteration
</div>
<div class="section" style="border-left: 4px solid #8E44AD;">
    <h3>Topology Changes</h3>
    <ul style="margin:8px 0 8px 20px; font-size:0.9em;">{changes_list}</ul>
    <h3>Invalidated KPIs</h3>
    <p style="font-size:0.85em;color:#666;">These KPIs cannot be reliably compared because the pipeline structure or thread assignments differ.</p>
    <ul style="margin:8px 0 8px 20px; font-size:0.9em;">{inv_list}</ul>
</div>
"""

        # Always show topology details
        topology_html = _build_topology_section(comp)

    # Summary banner
    reg_count = len(comp.regressions)
    imp_count = len(comp.improvements)
    if comp.topology_diff and comp.topology_diff.changed:
        banner_color = "#8E44AD"
        inv_count = len(comp.topology_diff.invalidated_kpis)
        banner_text = f"Topology changed -- {inv_count} KPI(s) invalidated"
        if reg_count > 0:
            banner_text += f", {reg_count} regression(s) on remaining KPIs"
    elif reg_count > 0:
        banner_color = "#E74C3C"
        banner_text = f"{reg_count} regression(s) detected"
    elif imp_count > 0:
        banner_color = "#2ECC71"
        banner_text = f"{imp_count} improvement(s), no regressions"
    else:
        banner_color = "#95A5A6"
        banner_text = "No significant changes detected"

    # Build the SHA table showing both baseline and iteration
    sha_same = iter_mcm_hash == base_mcm_hash and bool(iter_mcm_hash) and bool(base_mcm_hash)
    sha_rows_html = ""
    if base_mcm_hash or iter_mcm_hash:
        sha_rows_html += f"""
        <tr>
            <td><strong>MCM</strong></td>
            <td><code>{_escape_html(base_mcm_hash[:12])}</code>{f" ({_escape_html(base_mcm_branch)})" if base_mcm_branch else ""}</td>
            <td><code>{_escape_html(iter_mcm_hash[:12])}</code>{f" ({_escape_html(iter_mcm_branch)})" if iter_mcm_branch else ""}</td>
            <td>{"same" if sha_same else '<strong style="color:#E67E22;">differs</strong>'}</td>
        </tr>"""
    if harness_hash:
        sha_rows_html += f"""
        <tr>
            <td><strong>Harness</strong></td>
            <td colspan="2"><code>{_escape_html(harness_hash[:12])}</code>{f" ({_escape_html(harness_branch)})" if harness_branch else ""}{_escape_html(harness_dirty_str)}</td>
            <td>&mdash;</td>
        </tr>"""
    if blueos_core_ver:
        sha_rows_html += f"""
        <tr>
            <td><strong>BlueOS core</strong></td>
            <td colspan="2"><code>{_escape_html(blueos_core_ver)}</code></td>
            <td>&mdash;</td>
        </tr>"""
    if blueos_bootstrap_ver:
        sha_rows_html += f"""
        <tr>
            <td><strong>BlueOS bootstrap</strong></td>
            <td colspan="2"><code>{_escape_html(blueos_bootstrap_ver)}</code></td>
            <td>&mdash;</td>
        </tr>"""

    sha_table_html = ""
    if sha_rows_html:
        sha_table_html = f"""
    <div class="section" style="margin-top:8px;">
    <h2>Version Info</h2>
    <table style="font-size:0.85em;">
        <thead><tr><th>Component</th><th>Baseline</th><th>Iteration</th><th>Match</th></tr></thead>
        <tbody>{sha_rows_html}</tbody>
    </table>
    </div>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>A/B Report: {label} | {experiment}</title>
<style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
           color: #333; background: #f5f5f5; padding: 20px; max-width: 1400px; margin: 0 auto; }}
    h1 {{ font-size: 1.6em; margin-bottom: 4px; }}
    h2 {{ font-size: 1.2em; margin: 24px 0 12px 0; border-bottom: 2px solid #ddd; padding-bottom: 4px; }}
    h3 {{ font-size: 1.05em; margin: 16px 0 8px 0; }}
    .banner {{ padding: 12px 20px; border-radius: 6px; color: white; font-weight: bold;
               font-size: 1.1em; margin: 12px 0; }}
    .meta-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
                  gap: 8px; margin: 12px 0; }}
    .meta-item {{ background: white; padding: 8px 12px; border-radius: 4px; font-size: 0.9em;
                  word-break: break-all; }}
    .meta-item strong {{ color: #555; }}
    table {{ width: 100%; border-collapse: collapse; margin: 8px 0; background: white;
             border-radius: 4px; overflow: hidden; font-size: 0.85em; }}
    th {{ background: #2C3E50; color: white; padding: 8px 10px; text-align: left; font-weight: 600; }}
    td {{ padding: 6px 10px; border-bottom: 1px solid #eee; }}
    tr:hover {{ background: #f9f9f9; }}
    .chart {{ margin: 12px 0; text-align: center; }}
    .chart img {{ max-width: 100%; border-radius: 4px; box-shadow: 0 1px 4px rgba(0,0,0,0.1); }}
    .diff-block {{ background: #1e1e1e; color: #d4d4d4; padding: 12px; border-radius: 4px;
                   overflow-x: auto; font-family: "Fira Code", monospace; font-size: 0.8em;
                   white-space: pre; max-height: 400px; overflow-y: auto; }}
    .cmd-block {{ background: #f0f0f0; padding: 8px 12px; border-radius: 4px; font-family: monospace;
                  font-size: 0.82em; word-break: break-all; overflow-x: auto; margin: 4px 0; }}
    .section {{ background: white; padding: 16px; border-radius: 6px; margin: 12px 0;
                box-shadow: 0 1px 3px rgba(0,0,0,0.08); }}
    footer {{ margin-top: 24px; font-size: 0.8em; color: #999; text-align: center; }}
</style>
</head>
<body>

<h1>A/B Test Report</h1>
<p><strong>{experiment}</strong> &mdash; iteration: <strong>{label}</strong></p>

<div class="banner" style="background:{banner_color};">{banner_text}</div>

{topo_warning_html}

<div class="section">
<h2>Invocation</h2>
<div class="cmd-block">{_escape_html(cmdline_str)}</div>
</div>

{sha_table_html}

<div class="meta-grid">
    <div class="meta-item"><strong>Target:</strong> {_escape_html(str(host))}:{port} ({_escape_html(deploy_mode)})</div>
    <div class="meta-item"><strong>Duration:</strong> {duration}s x {reps} reps</div>
    <div class="meta-item"><strong>Mode:</strong> {mode_str}</div>
    <div class="meta-item"><strong>Generated:</strong> {generated_at}</div>
    <div class="meta-item"><strong>Env overrides:</strong> {env_str}</div>
    <div class="meta-item"><strong>Extra MCM args:</strong> {_escape_html(extra_args_str)}</div>
    <div class="meta-item"><strong>Clients:</strong> {client_str}</div>
    {sys_baseline_html}
</div>

<div class="section">
<h2>KPI Comparison (iteration vs baseline)</h2>
<table>
<thead>
    <tr>
        <th>KPI</th>
        <th>Unit</th>
        <th>Baseline (mean +/- std)</th>
        <th>Iteration (mean +/- std)</th>
        <th>Delta</th>
        <th>Delta %</th>
        <th>p-value</th>
        <th>95% CI</th>
        <th>Effect Size</th>
        <th>Power</th>
        <th>Diagnostics</th>
        <th>Verdict</th>
    </tr>
</thead>
<tbody>{kpi_rows}
</tbody>
</table>
</div>

<div class="section">
<h2>Per-Repetition Breakdown</h2>
<table>
<thead>
    <tr><th>KPI</th><th>Baseline reps</th><th>Iteration reps</th></tr>
</thead>
<tbody>{per_rep_rows}
</tbody>
</table>
</div>

{robust_html}

<div class="section">
<h2>Timeseries Charts</h2>
<p style="font-size:0.85em; color:#666;">Blue = baseline, Red = iteration. One line per rep.</p>
{charts_html}
</div>

{topology_html}

<div class="section">
<h2>Git Diff Summary</h2>
<pre style="font-size:0.85em; background:#fafafa; padding:8px; border-radius:4px;">{diff_stat}</pre>

<h3>Full Diff</h3>
<div class="diff-block">{_escape_html(diff_full[:20000])}{" ... (truncated)" if len(diff_full) > 20000 else ""}</div>
</div>

<footer>
    Generated by <code>ab_harness</code> &mdash; {generated_at}
</footer>

</body>
</html>"""


def _build_topology_section(comp: RunComparison) -> str:
    """Build the HTML for the Topology section.

    Shows per-pipeline element/thread groupings and edge graph.
    Uses topology data from the iteration's aggregate if available.
    """
    # We read topology from the aggregate.json of run_a (iteration)
    topo_data: dict[str, Any] | None = None
    run_a_path = Path(comp.run_a_dir)
    agg_path = run_a_path / "aggregate.json"
    if agg_path.exists():
        try:
            agg = json.loads(agg_path.read_text())
            topo_data = agg.get("topology")
        except Exception:
            pass

    if not topo_data:
        return ""

    pipelines = topo_data.get("pipelines") or []
    if not pipelines:
        return ""

    pipe_html = ""
    for pipe in pipelines:
        pname = pipe.get("pipeline_name", "unknown")
        expected_ms = pipe.get("expected_interval_ms")
        restarts_raw = pipe.get("restarts", 0)
        restarts = (
            restarts_raw.get("restart_count", 0)
            if isinstance(restarts_raw, dict)
            else restarts_raw
        )
        elements = pipe.get("elements") or {}
        edges = pipe.get("edges") or []

        # Group elements by thread
        thread_groups: dict[str, list[str]] = {}
        for ename, edata in elements.items():
            tid = str(edata.get("thread_id", "?")) if isinstance(edata, dict) else "?"
            thread_groups.setdefault(tid, []).append(ename)

        thread_rows = ""
        for tid, elem_names in sorted(thread_groups.items()):
            elem_list = ", ".join(f"<code>{_escape_html(e)}</code>" for e in sorted(elem_names))
            thread_rows += f"<tr><td>{_escape_html(tid)}</td><td>{elem_list}</td></tr>"

        edge_rows = ""
        for edge in edges:
            fr = edge.get("from_element", "?")
            to = edge.get("to_element", "?")
            fd = edge.get("freshness_delay_ms")
            cc = edge.get("causal_confidence")
            fd_str = f"{fd:.1f}" if fd is not None else "N/A"
            cc_str = (
                f"{cc:.3f}"
                if isinstance(cc, int | float)
                else (str(cc) if cc is not None else "N/A")
            )
            edge_rows += (
                f"<tr><td><code>{_escape_html(str(fr))}</code></td>"
                f"<td><code>{_escape_html(str(to))}</code></td>"
                f"<td>{fd_str}</td><td>{cc_str}</td></tr>"
            )

        expected_str = f"{expected_ms:.1f}ms" if expected_ms is not None else "N/A"

        pipe_html += f"""
        <h3>{_escape_html(pname)}</h3>
        <p style="font-size:0.85em;color:#666;">
            Expected interval: {expected_str} &bull;
            Elements: {len(elements)} &bull;
            Edges: {len(edges)} &bull;
            Restarts: {restarts}
        </p>
        <h4 style="font-size:0.9em; margin-top:8px;">Thread Groupings</h4>
        <table style="font-size:0.82em;">
            <thead><tr><th>Thread ID</th><th>Elements</th></tr></thead>
            <tbody>{thread_rows}</tbody>
        </table>
        """
        if edge_rows:
            pipe_html += f"""
        <h4 style="font-size:0.9em; margin-top:8px;">Edge Graph</h4>
        <table style="font-size:0.82em;">
            <thead><tr><th>From</th><th>To</th><th>Freshness (ms)</th><th>Causal Confidence</th></tr></thead>
            <tbody>{edge_rows}</tbody>
        </table>
        """

    return f"""
<div class="section">
<h2>Pipeline Topology</h2>
{pipe_html}
</div>
"""


def _escape_html(text: str) -> str:
    return (
        text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
    )


# ---------------------------------------------------------------------------
# Batch summary report
# ---------------------------------------------------------------------------


def generate_batch_summary(
    state: dict[str, Any],
    runs_dir: Path,
    config_stem: str,
) -> Path:
    """Generate an HTML summary page for a batch of experiments.

    Reads each experiment's ``comparison.json`` to extract verdicts and
    KPI deltas, and combines them into a single overview page.

    Parameters
    ----------
    state : dict
        The batch state dict (as persisted in ``_batch_state_*.json``).
    runs_dir : Path
        Base directory for run data.
    config_stem : str
        Stem of the config file name (used for the output filename).

    Returns
    -------
    Path
        Path to the generated ``batch_summary_<config_stem>.html``.
    """
    experiments_state = state.get("experiments", [])
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Gather per-experiment data
    rows: list[dict[str, Any]] = []
    for exp_state in experiments_state:
        name = exp_state["experiment"]
        status = exp_state.get("status", "pending")
        elapsed = exp_state.get("elapsed_s")
        report_path = exp_state.get("report")
        error_msg = exp_state.get("error")

        row: dict[str, Any] = {
            "experiment": name,
            "status": status,
            "elapsed_s": elapsed,
            "report_path": report_path,
            "error": error_msg,
            "regressions": 0,
            "improvements": 0,
            "kpi_highlights": [],
            "source": "",
        }

        # Try to load comparison.json for completed experiments
        if status == "completed":
            exp_dir = runs_dir / name
            # Find latest iteration
            iterations_dir = exp_dir / "iterations"
            comparison_data = None
            if iterations_dir.is_dir():
                iter_dirs = sorted(iterations_dir.iterdir())
                if iter_dirs:
                    comp_path = iter_dirs[-1] / "comparison.json"
                    if comp_path.exists():
                        try:
                            comparison_data = json.loads(comp_path.read_text())
                        except Exception as exc:
                            log.debug("Could not load comparison for %s: %s", name, exc)

            if comparison_data:
                kpis = comparison_data.get("kpis", [])
                for kpi in kpis:
                    verdict = kpi.get("verdict", "")
                    if "regression" in verdict:
                        row["regressions"] += 1
                    elif "improvement" in verdict:
                        row["improvements"] += 1

                # Extract key KPI highlights (cpu, throughput)
                for kpi in kpis:
                    kpi_name = kpi.get("name", "")
                    if kpi_name in ("system_cpu_pct", "throughput_fps", "freshness_delay_ms"):
                        delta_pct = kpi.get("delta_pct")
                        if delta_pct is not None:
                            row["kpi_highlights"].append(f"{kpi_name}: {delta_pct:+.1f}%")

            # Try to determine source from metadata
            meta_path = None
            if iterations_dir.is_dir():
                iter_dirs = sorted(iterations_dir.iterdir())
                if iter_dirs:
                    meta_path = iter_dirs[-1] / "metadata.json"
            if meta_path and meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text())
                    github_pr = meta.get("github_pr")
                    mcm_hash = meta.get("mcm_git_hash", "")
                    if github_pr:
                        row["source"] = f"PR #{github_pr}"
                    elif mcm_hash:
                        row["source"] = f"commit {mcm_hash[:8]}"
                    elif meta.get("mcm_git_diff_stat"):
                        row["source"] = "patch"
                except Exception:
                    pass

        rows.append(row)

    # Count totals
    total = len(rows)
    completed = sum(1 for r in rows if r["status"] == "completed")
    failed = sum(1 for r in rows if r["status"] == "failed")
    any_regressions = sum(1 for r in rows if r["regressions"] > 0)

    # Build table rows
    table_rows = ""
    for r in rows:
        status = r["status"]
        if status == "completed":
            if r["regressions"] > 0:
                status_color = "#E74C3C"
                status_icon = "&#x2716;"
                verdict_text = f"{r['regressions']} regression(s)"
            elif r["improvements"] > 0:
                status_color = "#2ECC71"
                status_icon = "&#x2714;"
                verdict_text = f"{r['improvements']} improvement(s)"
            else:
                status_color = "#95A5A6"
                status_icon = "&#x2014;"
                verdict_text = "neutral"
        elif status == "failed":
            status_color = "#8E44AD"
            status_icon = "&#x26A0;"
            verdict_text = f"FAILED: {_escape_html(r.get('error', '?')[:80])}"
        elif status == "running":
            status_color = "#F39C12"
            status_icon = "&#x25B6;"
            verdict_text = "running"
        else:
            status_color = "#BDC3C7"
            status_icon = "&#x25CB;"
            verdict_text = "pending"

        elapsed_str = f"{r['elapsed_s']:.0f}s" if r.get("elapsed_s") is not None else "-"
        kpi_str = "<br>".join(r.get("kpi_highlights", [])) or "-"
        report_link = ""
        if r.get("report_path"):
            report_link = f'<a href="{_escape_html(r["report_path"])}">report</a>'

        table_rows += f"""
        <tr>
            <td><strong>{_escape_html(r["experiment"])}</strong></td>
            <td>{_escape_html(r.get("source", "") or "-")}</td>
            <td style="color:{status_color}; font-weight:bold;">{status_icon} {verdict_text}</td>
            <td>{kpi_str}</td>
            <td>{elapsed_str}</td>
            <td>{report_link}</td>
        </tr>"""

    # Banner
    if any_regressions:
        banner_color = "#E74C3C"
        banner_text = f"{any_regressions}/{completed} experiment(s) have regressions"
    elif failed > 0:
        banner_color = "#8E44AD"
        banner_text = f"{failed} experiment(s) failed, {completed} completed"
    else:
        banner_color = "#2ECC71"
        banner_text = f"All {completed} experiment(s) clean -- no regressions"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Batch Summary: {_escape_html(config_stem)}</title>
<style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
           color: #333; background: #f5f5f5; padding: 20px; max-width: 1200px; margin: 0 auto; }}
    h1 {{ font-size: 1.6em; margin-bottom: 4px; }}
    .banner {{ padding: 12px 20px; border-radius: 6px; color: white; font-weight: bold;
               font-size: 1.1em; margin: 12px 0; }}
    .meta-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
                  gap: 8px; margin: 12px 0; }}
    .meta-item {{ background: white; padding: 8px 12px; border-radius: 4px; font-size: 0.9em; }}
    .meta-item strong {{ color: #555; }}
    table {{ width: 100%; border-collapse: collapse; margin: 8px 0; background: white;
             border-radius: 4px; overflow: hidden; font-size: 0.9em; }}
    th {{ background: #2C3E50; color: white; padding: 10px 12px; text-align: left; font-weight: 600; }}
    td {{ padding: 8px 12px; border-bottom: 1px solid #eee; vertical-align: top; }}
    tr:hover {{ background: #f9f9f9; }}
    a {{ color: #2980B9; text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    .section {{ background: white; padding: 16px; border-radius: 6px; margin: 12px 0;
                box-shadow: 0 1px 3px rgba(0,0,0,0.08); }}
    footer {{ margin-top: 24px; font-size: 0.8em; color: #999; text-align: center; }}
</style>
</head>
<body>

<h1>Batch A/B Test Summary</h1>
<p><strong>{_escape_html(config_stem)}</strong></p>

<div class="banner" style="background:{banner_color};">{banner_text}</div>

<div class="meta-grid">
    <div class="meta-item"><strong>Config:</strong> {_escape_html(state.get("config_file", "?"))}</div>
    <div class="meta-item"><strong>Started:</strong> {_escape_html(state.get("started_at", "?"))}</div>
    <div class="meta-item"><strong>Generated:</strong> {now}</div>
    <div class="meta-item"><strong>Experiments:</strong> {completed} completed, {failed} failed, {total} total</div>
</div>

<div class="section">
<h2 style="font-size: 1.2em; margin: 0 0 12px 0; border-bottom: 2px solid #ddd; padding-bottom: 4px;">Experiments</h2>
<table>
<thead>
    <tr>
        <th>Experiment</th>
        <th>Source</th>
        <th>Verdict</th>
        <th>Key KPIs</th>
        <th>Time</th>
        <th>Report</th>
    </tr>
</thead>
<tbody>{table_rows}
</tbody>
</table>
</div>

<footer>
    Generated by <code>ab_harness batch</code> &mdash; {now}
</footer>

</body>
</html>"""

    out_path = runs_dir / f"batch_summary_{config_stem}.html"
    out_path.write_text(html)
    log.info("Batch summary written to %s", out_path)
    return out_path
