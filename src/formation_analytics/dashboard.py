"""Static HTML dashboard of the Formation KPIs (Plotly, opens in any browser)."""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

BLUE, ORANGE, GREEN, RED, GREY = "#2563eb", "#ea580c", "#16a34a", "#dc2626", "#6b7280"
CLASS_COLOR = {"calculation_error": "#a855f7", "measurement_error": ORANGE,
               "true_quality_shift": RED, "unresolved": GREY}


def _yield_fig(kpi: pd.DataFrame, episodes) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        subplot_titles=("Daily yield", "First-pass fail rate by mode"))
    fig.add_scatter(x=kpi.grade_day, y=kpi.first_pass_yield, name="First-pass yield",
                    line=dict(color=BLUE), row=1, col=1)
    fig.add_scatter(x=kpi.grade_day, y=kpi.final_yield, name="Final yield (after retest)",
                    line=dict(color=GREEN), row=1, col=1)
    for col, name, color in (("n_fail_k", "K-value high", RED),
                             ("n_fail_cap", "Capacity low", ORANGE),
                             ("n_fail_rls", "R_LS high", GREY)):
        fig.add_scatter(x=kpi.grade_day, y=kpi[col] / kpi.cells_graded, name=name,
                        line=dict(color=color), row=2, col=1)
    for ep in episodes:
        for r in (1, 2):
            fig.add_vrect(x0=ep.day_first - 0.5, x1=ep.day_last + 0.5, row=r, col=1,
                          fillcolor=CLASS_COLOR[ep.classification], opacity=0.15, line_width=0)
        fig.add_annotation(x=(ep.day_first + ep.day_last) / 2, y=1.0, yref="y domain",
                           text=f"{ep.mode}<br>{ep.classification.replace('_', ' ')}",
                           showarrow=False, font=dict(size=10), row=1, col=1)
    fig.update_yaxes(tickformat=".1%", row=1, col=1)
    fig.update_yaxes(tickformat=".1%", type="log", row=2, col=1)
    fig.update_xaxes(title_text="grading day", row=2, col=1)
    fig.update_layout(height=620, legend=dict(orientation="h", y=-0.12))
    return fig


def _evidence_fig(con, episodes) -> go.Figure:
    fig = make_subplots(rows=1, cols=2, subplot_titles=(
        "Capacity-low rate by grading tester, flagged window",
        "K-value: stored / recomputed from raw OCV"))
    cap = [e for e in episodes if e.mode == "CAP_LOW"]
    if cap:
        e = cap[0]
        t = con.execute("SELECT grading_tester, AVG(fail_cap::INT) r FROM cell_results "
                        "WHERE grade_day BETWEEN ? AND ? GROUP BY 1 ORDER BY 1",
                        [e.day_first, e.day_last]).df()
        fig.add_bar(x=t.grading_tester, y=t.r, marker_color=[
            ORANGE if g in e.hot_levels else GREY for g in t.grading_tester],
            showlegend=False, row=1, col=1)
        fig.update_yaxes(tickformat=".2%", row=1, col=1)
    k = con.execute(open(Path(__file__).resolve().parents[2] / "sql" / "k_value_recompute.sql")
                    .read()).df()
    fig.add_scatter(x=k.grade_day, y=k.median_stored_over_raw, mode="lines+markers",
                    line=dict(color="#a855f7"), showlegend=False, row=1, col=2)
    fig.update_xaxes(title_text="grading day", row=1, col=2)
    fig.update_layout(height=360)
    return fig


def _capacity_fig(ramps: dict) -> go.Figure:
    plan, base = ramps["plan"], ramps["baseline_only"]
    fig = make_subplots(rows=1, cols=2, subplot_titles=(
        "Demand vs station capacity (plan: pilot day 95, rollout day 110)",
        "Cycle time, build to graded"))
    fig.add_scatter(x=plan.day, y=plan.demand, name="Demand", line=dict(color="black"), row=1, col=1)
    for c, n, col in (("cap_formation", "Formation", BLUE), ("cap_aging", "Aging", GREEN),
                      ("cap_grading", "Grading", ORANGE)):
        fig.add_scatter(x=plan.day, y=plan[c], name=f"{n} capacity",
                        line=dict(color=col, dash="dot"), row=1, col=1)
    fig.add_scatter(x=base.day, y=base.cycle_time_h, name="Baseline formation only",
                    line=dict(color=RED), row=1, col=2)
    fig.add_scatter(x=plan.day, y=plan.cycle_time_h, name="Plan",
                    line=dict(color=BLUE), row=1, col=2)
    fig.update_yaxes(title_text="cells/day", row=1, col=1)
    fig.update_yaxes(title_text="hours", row=1, col=2)
    fig.update_layout(height=380, legend=dict(orientation="h", y=-0.2))
    return fig


def _limits_fig(sweep: pd.DataFrame, current: float) -> go.Figure:
    fig = make_subplots(rows=1, cols=2, subplot_titles=(
        "R_LS limit: yield vs life risk (baseline cells)",
        "Scrap + warranty cost per good cell"))
    fig.add_scatter(x=sweep.rls_usl_mohm, y=sweep.yield_rls, name="Yield at R_LS screen",
                    line=dict(color=BLUE), row=1, col=1)
    fig.add_scatter(x=sweep.rls_usl_mohm, y=sweep.cell_claim_prob, name="Cell misses life spec",
                    line=dict(color=ORANGE), row=1, col=1)
    fig.add_scatter(x=sweep.rls_usl_mohm, y=sweep.pack_claim_prob,
                    name="Pack misses life spec (96s4p)", line=dict(color=RED), row=1, col=1)
    fig.add_scatter(x=sweep.rls_usl_mohm, y=sweep.cost_cell_view, name="Cost, cell view",
                    line=dict(color=ORANGE), row=1, col=2)
    fig.add_scatter(x=sweep.rls_usl_mohm, y=sweep.cost_pack_view, name="Cost, pack view",
                    line=dict(color=RED), row=1, col=2)
    for c in (1, 2):
        fig.add_vline(x=current, line_dash="dash", line_color=GREY, row=1, col=c)
    fig.update_yaxes(tickformat=".0%", row=1, col=1)
    fig.update_yaxes(title_text="$ per good cell", type="log", row=1, col=2)
    fig.update_xaxes(title_text="R_LS upper limit (mOhm)")
    fig.update_layout(height=380, legend=dict(orientation="h", y=-0.25))
    return fig


def _public_fig(pub: pd.DataFrame) -> go.Figure:
    fig = make_subplots(rows=1, cols=2, subplot_titles=(
        "Public data: cycles to 70% by formation protocol",
        "Public data (RT): low-SOC resistance vs life"))
    for p, col in (("baseline", GREY), ("fast", BLUE)):
        g = pub[pub.protocol == p]
        fig.add_box(x=g.temperature, y=g.cycles_to_70, name=p, marker_color=col,
                    boxpoints="all", row=1, col=1)
        r = g[g.temperature == "RT"]
        fig.add_scatter(x=r.rls_mohm, y=r.cycles_to_70, mode="markers", name=f"{p} (RT)",
                        marker=dict(color=col, size=9), row=1, col=2)
    fig.update_layout(boxmode="group", height=380, legend=dict(orientation="h", y=-0.2))
    fig.update_xaxes(title_text="R_LS, 10 s at 5% SOC (mOhm)", row=1, col=2)
    return fig


def build(path: Path, kpi, episodes, con, ramps, sweep, current_usl, pub, tiles: dict) -> None:
    figs = [_yield_fig(kpi, episodes), _evidence_fig(con, episodes), _capacity_fig(ramps),
            _limits_fig(sweep, current_usl), _public_fig(pub)]
    tile_html = "".join(f"<div class='t'><div class='v'>{v}</div><div class='k'>{k}</div></div>"
                        for k, v in tiles.items())
    body = "".join(f.to_html(full_html=False, include_plotlyjs="cdn" if i == 0 else False)
                   for i, f in enumerate(figs))
    html = f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Formation KPI Dashboard</title>
<style>body{{font-family:system-ui,sans-serif;margin:16px auto;max-width:1150px;color:#111;background:#fff}}
.tiles{{display:flex;flex-wrap:wrap;gap:10px;margin:12px 0}}.t{{border:1px solid #e5e7eb;border-radius:8px;
padding:10px 14px;min-width:150px}}.v{{font-size:22px;font-weight:600}}.k{{font-size:12px;color:#6b7280}}
p.note{{font-size:13px;color:#6b7280}}</style></head><body>
<h1>Formation KPI Dashboard</h1>
<p class="note">Line data are simulated (cell capacity and R_LS distributions from the public Weng et al.
2021 fast-formation data; line layout, defects and costs assumed). Lifetime panels are measured public data.</p>
<div class="tiles">{tile_html}</div>{body}</body></html>"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html)
