"""Plotly presentation helpers for the SupplyChainX dashboard."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


COLORS = {
    "cyan": "#22D3EE",
    "teal": "#2DD4BF",
    "navy": "#0B1422",
    "slate": "#64748B",
    "light_slate": "#94A3B8",
    "red": "#F87171",
    "grid": "#26354A",
    "text": "#CBD5E1",
}

TERM_LABELS = {
    "log_demand_multiplier": "Log demand multiplier",
    "demand_capacity_interaction": "Demand × capacity reduction",
    "capacity_reduction": "Capacity reduction",
    "transport_cost_inflation": "Transport-cost inflation",
    "shortage_penalty_multiplier": "Shortage-penalty multiplier",
    "disrupted_warehouse_count": "Disrupted warehouse count",
}


def style_figure(fig: go.Figure, height: int = 360) -> go.Figure:
    """Apply the dashboard's restrained dark analytics styling."""
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": COLORS["text"], "size": 12},
        margin=dict(l=12, r=12, t=20, b=12),
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
        hoverlabel={"bgcolor": "#111C2C", "font_color": "#E5EEF7"},
    )
    fig.update_xaxes(gridcolor=COLORS["grid"], zerolinecolor=COLORS["grid"])
    fig.update_yaxes(gridcolor=COLORS["grid"], zerolinecolor=COLORS["grid"])
    return fig


def warehouse_utilization_chart(warehouses: pd.DataFrame) -> go.Figure:
    """Show utilization for every warehouse, ordered for easy comparison."""
    data = warehouses.sort_values("utilization", ascending=True).copy()
    data["utilization_pct"] = 100 * data["utilization"].fillna(0)
    colors = [COLORS["cyan"] if active > 0.5 else COLORS["slate"] for active in data["active"]]
    fig = go.Figure(
        go.Bar(
            x=data["utilization_pct"],
            y=data["warehouse_id"],
            orientation="h",
            marker_color=colors,
            customdata=data[["shipped_quantity", "capacity", "active"]],
            hovertemplate=(
                "<b>%{y}</b><br>Utilization %{x:.1f}%"
                "<br>Shipped %{customdata[0]:,.0f}"
                "<br>Capacity %{customdata[1]:,.0f}<extra></extra>"
            ),
        )
    )
    upper = max(105.0, float(data["utilization_pct"].max()) * 1.08)
    fig.update_xaxes(title="Utilization", ticksuffix="%", range=[0, upper])
    fig.update_yaxes(title=None)
    return style_figure(fig, height=430)


def top_utilized_warehouses_chart(warehouses: pd.DataFrame, top_n: int = 6) -> go.Figure:
    """Rank the most utilized active warehouses."""
    data = (
        warehouses.loc[warehouses["active"] > 0.5]
        .nlargest(top_n, "utilization")
        .sort_values("utilization")
        .copy()
    )
    data["utilization_pct"] = 100 * data["utilization"]
    fig = go.Figure(
        go.Bar(
            x=data["utilization_pct"],
            y=data["warehouse_id"],
            orientation="h",
            marker_color=COLORS["cyan"],
            text=data["utilization_pct"].map(lambda value: f"{value:.1f}%"),
            textposition="outside",
            cliponaxis=False,
            hovertemplate="<b>%{y}</b><br>Utilization %{x:.1f}%<extra></extra>",
        )
    )
    upper = max(108.0, float(data["utilization_pct"].max()) * 1.15)
    fig.update_xaxes(title=None, ticksuffix="%", range=[0, upper])
    fig.update_yaxes(title=None)
    return style_figure(fig, height=310)


def largest_shipment_lanes_chart(shipments: pd.DataFrame, top_n: int = 6) -> go.Figure:
    """Rank the largest optimized warehouse-to-zone shipment lanes."""
    if shipments.empty:
        return style_figure(
            go.Figure().add_annotation(text="No active shipment lanes", showarrow=False),
            height=310,
        )
    data = shipments.nlargest(top_n, "quantity").sort_values("quantity").copy()
    data["lane"] = data["warehouse_id"] + " → " + data["demand_zone_id"]
    fig = go.Figure(
        go.Bar(
            x=data["quantity"],
            y=data["lane"],
            orientation="h",
            marker_color=COLORS["teal"],
            text=data["quantity"].map(lambda value: f"{value:,.0f}"),
            textposition="outside",
            cliponaxis=False,
            hovertemplate="<b>%{y}</b><br>Shipment %{x:,.0f} units<extra></extra>",
        )
    )
    fig.update_xaxes(title=None, range=[0, float(data["quantity"].max()) * 1.18])
    fig.update_yaxes(title=None)
    return style_figure(fig, height=310)


def cost_breakdown_chart(metrics: dict) -> go.Figure:
    """Show the composition of optimized network cost."""
    labels = ["Transport", "Fixed operations", "Shortage"]
    values = [metrics["transport_cost"], metrics["fixed_cost"], metrics["shortage_cost"]]
    fig = go.Figure(
        go.Pie(
            labels=labels,
            values=values,
            hole=0.64,
            marker_colors=[COLORS["cyan"], COLORS["teal"], COLORS["red"]],
            textinfo="label+percent",
            hovertemplate="<b>%{label}</b><br>₹%{value:,.0f}<br>%{percent}<extra></extra>",
        )
    )
    fig.add_annotation(
        text=f"₹{metrics['total_cost'] / 1_000_000:.2f}M<br><span style='font-size:11px'>TOTAL</span>",
        showarrow=False,
        font={"color": COLORS["text"], "size": 18},
    )
    return style_figure(fig, height=355)


def demand_service_chart(demand_zones: pd.DataFrame) -> go.Figure:
    """Compare served and unmet demand across zones."""
    data = demand_zones.sort_values("demand_zone_id")
    fig = go.Figure()
    fig.add_bar(
        name="Served",
        x=data["demand_zone_id"],
        y=data["served_demand"],
        marker_color=COLORS["teal"],
        hovertemplate="<b>%{x}</b><br>Served %{y:,.0f}<extra></extra>",
    )
    fig.add_bar(
        name="Unmet",
        x=data["demand_zone_id"],
        y=data["unmet_demand"],
        marker_color=COLORS["red"],
        hovertemplate="<b>%{x}</b><br>Unmet %{y:,.0f}<extra></extra>",
    )
    fig.update_layout(barmode="stack")
    fig.update_xaxes(title="Demand zone", tickangle=-45)
    fig.update_yaxes(title="Units")
    return style_figure(fig, height=380)


def scenario_comparison_chart(summary: pd.DataFrame, metric: str = "total_cost") -> go.Figure:
    """Rank saved scenarios on a selected decision metric."""
    labels = {
        "total_cost": "Total cost",
        "service_level": "Service level",
        "active_warehouses": "Active warehouses",
        "shortage_risk_score": "Shortage risk",
    }
    ascending = metric in {"total_cost", "active_warehouses", "shortage_risk_score"}
    data = summary.sort_values(metric, ascending=not ascending).copy()
    display_values = data[metric] * 100 if metric == "service_level" else data[metric]
    text = (
        display_values.map(lambda value: f"{value:.1f}%")
        if metric == "service_level"
        else display_values.map(lambda value: f"{value:,.0f}")
    )
    colors = [
        COLORS["red"] if metric == "shortage_risk_score" and value > 0 else COLORS["cyan"]
        for value in display_values
    ]
    fig = go.Figure(
        go.Bar(
            x=display_values,
            y=data["scenario"],
            orientation="h",
            marker_color=colors,
            text=text,
            textposition="outside",
            cliponaxis=False,
            hovertemplate="<b>%{y}</b><br>%{x:,.2f}<extra></extra>",
        )
    )
    suffix = "%" if metric == "service_level" else ""
    fig.update_xaxes(title=labels[metric], ticksuffix=suffix)
    fig.update_yaxes(title=None)
    return style_figure(fig, height=max(280, 58 * len(data)))


def shipment_heatmap(shipments: pd.DataFrame) -> go.Figure:
    """Show shipment intensity for all used warehouse-to-zone combinations."""
    if shipments.empty:
        return style_figure(
            go.Figure().add_annotation(text="No shipments in this solution", showarrow=False),
            height=380,
        )
    matrix = shipments.pivot_table(
        index="warehouse_id",
        columns="demand_zone_id",
        values="quantity",
        aggfunc="sum",
        fill_value=0,
    )
    fig = px.imshow(
        matrix,
        aspect="auto",
        color_continuous_scale=["#0B1422", "#155E75", COLORS["cyan"]],
        labels={"x": "Demand zone", "y": "Warehouse", "color": "Units"},
    )
    fig.update_xaxes(tickangle=-45)
    return style_figure(fig, height=460)


def coefficient_ranking_chart(coefficients: pd.DataFrame) -> go.Figure:
    """Rank simulation-response coefficients with their 95% confidence intervals."""
    data = coefficients.loc[coefficients["term"].isin(TERM_LABELS)].copy()
    data["label"] = data["term"].map(TERM_LABELS)
    data = data.sort_values("estimate")
    colors = [COLORS["light_slate"] if value < 0 else COLORS["cyan"] for value in data["estimate"]]
    fig = go.Figure(
        go.Bar(
            x=data["estimate"],
            y=data["label"],
            orientation="h",
            marker_color=colors,
            error_x={
                "type": "data",
                "symmetric": False,
                "array": data["ci_upper"] - data["estimate"],
                "arrayminus": data["estimate"] - data["ci_lower"],
                "color": COLORS["text"],
                "thickness": 1.2,
            },
            text=data["estimate"].map(lambda value: f"{value:.3f}"),
            textposition="outside",
            cliponaxis=False,
            hovertemplate=(
                "<b>%{y}</b><br>Estimate %{x:.4f}"
                "<br>95% CI [%{customdata[0]:.4f}, %{customdata[1]:.4f}]<extra></extra>"
            ),
            customdata=data[["ci_lower", "ci_upper"]],
        )
    )
    fig.add_vline(x=0, line_dash="dash", line_color=COLORS["slate"])
    fig.update_xaxes(title="Estimated log-cost response coefficient")
    fig.update_yaxes(title=None)
    return style_figure(fig, height=390)


def standardized_sensitivity_chart(ranking: pd.DataFrame) -> go.Figure:
    """Rank response drivers using standardized coefficient magnitudes."""
    data = ranking.sort_values("magnitude", ascending=True).copy()
    colors = [
        COLORS["light_slate"] if value < 0 else COLORS["cyan"]
        for value in data["standardized_effect"]
    ]
    fig = go.Figure(
        go.Bar(
            x=data["magnitude"],
            y=data["driver"],
            orientation="h",
            marker_color=colors,
            text=data["magnitude"].map(lambda value: f"{value:.3f}"),
            textposition="outside",
            cliponaxis=False,
            customdata=data[["standardized_effect"]],
            hovertemplate=(
                "<b>%{y}</b><br>Standardized magnitude %{x:.4f}"
                "<br>Signed standardized effect %{customdata[0]:+.4f}<extra></extra>"
            ),
        )
    )
    fig.update_xaxes(title="Absolute standardized response effect")
    fig.update_yaxes(title=None)
    return style_figure(fig, height=370)

