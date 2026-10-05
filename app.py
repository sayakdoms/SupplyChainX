"""Premium Streamlit decision-support dashboard for SupplyChainX."""

from __future__ import annotations

from datetime import datetime
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from src.analytics import scenario_summary
from src.econometric_experiments import run_experiments
from src.econometrics import (
    actual_vs_predicted_chart,
    correlation_matrix_chart,
    fit_ols_response,
    fit_shortage_logit,
    predict_optimized_cost,
    predict_shortage_probability,
    residual_distribution_chart,
    residual_vs_fitted_chart,
    successful_scenarios,
)
from src.scenario_engine import ScenarioConfig, apply_scenario
from src.solver import solve_network
from src.visualizations import (
    TERM_LABELS,
    coefficient_ranking_chart,
    cost_breakdown_chart,
    demand_service_chart,
    largest_shipment_lanes_chart,
    scenario_comparison_chart,
    shipment_heatmap,
    style_figure,
    standardized_sensitivity_chart,
    top_utilized_warehouses_chart,
    warehouse_utilization_chart,
)


PROJECT_DIR = Path(__file__).resolve().parent

SCENARIO_PRESETS = {
    "Base Network": {
        "demand_multiplier": 1.00,
        "transport_inflation_pct": 0,
        "capacity_reduction_pct": 0,
        "capacity_disruption": False,
        "warehouse_outage": False,
        "penalty_multiplier": 1.00,
    },
    "Demand Surge": {
        "demand_multiplier": 1.25,
        "transport_inflation_pct": 0,
        "capacity_reduction_pct": 0,
        "capacity_disruption": False,
        "warehouse_outage": False,
        "penalty_multiplier": 1.00,
    },
    "Transport Cost Shock": {
        "demand_multiplier": 1.00,
        "transport_inflation_pct": 25,
        "capacity_reduction_pct": 0,
        "capacity_disruption": False,
        "warehouse_outage": False,
        "penalty_multiplier": 1.00,
    },
    "Capacity Failure": {
        "demand_multiplier": 1.00,
        "transport_inflation_pct": 0,
        "capacity_reduction_pct": 30,
        "capacity_disruption": True,
        "warehouse_outage": False,
        "penalty_multiplier": 1.00,
    },
    "Warehouse Outage": {
        "demand_multiplier": 1.00,
        "transport_inflation_pct": 0,
        "capacity_reduction_pct": 0,
        "capacity_disruption": False,
        "warehouse_outage": True,
        "penalty_multiplier": 1.00,
    },
    "Combined Stress Test": {
        "demand_multiplier": 1.25,
        "transport_inflation_pct": 20,
        "capacity_reduction_pct": 25,
        "capacity_disruption": True,
        "warehouse_outage": True,
        "penalty_multiplier": 1.50,
    },
    "Custom Scenario": None,
}


@st.cache_data
def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load generated network data from disk."""
    return (
        pd.read_csv(PROJECT_DIR / "data" / "warehouses.csv"),
        pd.read_csv(PROJECT_DIR / "data" / "demand_zones.csv"),
        pd.read_csv(PROJECT_DIR / "data" / "transportation_costs.csv"),
    )


@st.cache_data
def load_econometric_data(path: str, modified_time: float) -> pd.DataFrame:
    """Load the generated simulation-response dataset."""
    del modified_time
    return pd.read_csv(path)


@st.cache_resource
def fit_response_models(path: str, modified_time: float):
    """Fit and cache response models for one generated dataset version."""
    data = load_econometric_data(path, modified_time)
    return fit_ols_response(data), fit_shortage_logit(data)


@st.cache_data(show_spinner=False)
def run_exact_response_scenario(
    demand_multiplier: float,
    transport_cost_inflation: float,
    shortage_penalty_multiplier: float,
    capacity_reduction: float,
    disrupted_warehouse_count: int,
) -> dict[str, object]:
    """Run one exact MILP scenario using the established experiment generator."""
    design = pd.DataFrame(
        [
            {
                "scenario_id": "RESPONSE_ESTIMATOR",
                "demand_multiplier": demand_multiplier,
                "transport_cost_inflation": transport_cost_inflation,
                "shortage_penalty_multiplier": shortage_penalty_multiplier,
                "capacity_reduction": capacity_reduction,
                "disrupted_warehouse_count": disrupted_warehouse_count,
            }
        ]
    )
    result = run_experiments(design=design, base_data=load_data())
    return result.iloc[0].to_dict()


def money(value: float) -> str:
    """Format Indian-rupee values for executive display."""
    if abs(value) >= 1_000_000:
        return f"₹{value / 1_000_000:,.2f}M"
    if abs(value) >= 1_000:
        return f"₹{value / 1_000:,.1f}K"
    return f"₹{value:,.0f}"


def solver_status_label(termination_condition: str) -> tuple[str, str]:
    """Return a concise status label and semantic Markdown color."""
    normalized = termination_condition.lower()
    if normalized == "optimal":
        return "OPTIMAL", "green"
    if "feasible" in normalized:
        return "FEASIBLE", "orange"
    return "WARNING", "red"


def solver_display_name(name: str) -> str:
    """Return a portfolio-friendly solver label."""
    normalized = str(name).lower()
    if normalized in {"highs", "appsi_highs"}:
        return "HiGHS"
    if normalized == "gurobi":
        return "Gurobi"
    if normalized == "scip":
        return "SCIP"
    return str(name)


def average_active_utilization(result: object) -> float:
    """Return average utilization across active warehouses for display."""
    active = result.warehouses.loc[result.warehouses["active"] > 0.5]
    if active.empty:
        return 0.0
    return float(
        active["utilization"].replace([np.inf, -np.inf], np.nan).mean()
    )


def is_stress_config(config: ScenarioConfig) -> bool:
    """Classify presentation state without changing scenario semantics."""
    return any(
        [
            abs(config.demand_multiplier - 1.0) > 1e-9,
            abs(config.transport_cost_inflation) > 1e-9,
            config.capacity_reduction_fraction > 1e-9,
            config.shutdown_warehouse is not None,
            abs(config.shortage_penalty_multiplier - 1.0) > 1e-9,
        ]
    )


def scenario_inputs(config: ScenarioConfig) -> dict[str, object]:
    """Serialize the existing scenario controls for session-only history."""
    return {
        "demand_multiplier": config.demand_multiplier,
        "transport_cost_inflation": config.transport_cost_inflation,
        "capacity_reduction_warehouse": config.capacity_reduction_warehouse,
        "capacity_reduction_fraction": config.capacity_reduction_fraction,
        "shutdown_warehouse": config.shutdown_warehouse,
        "shortage_penalty_multiplier": config.shortage_penalty_multiplier,
    }


def shortage_probability_for_config(config: ScenarioConfig) -> float | None:
    """Use the established logistic response model for a UI risk estimate."""
    econometric_path = PROJECT_DIR / "results" / "econometric_scenarios.csv"
    if not econometric_path.exists():
        return None
    _, logistic_result = fit_response_models(
        str(econometric_path), econometric_path.stat().st_mtime
    )
    return predict_shortage_probability(
        logistic_result,
        demand_multiplier=config.demand_multiplier,
        transport_cost_inflation=config.transport_cost_inflation,
        capacity_reduction=config.capacity_reduction_fraction,
        shortage_penalty_multiplier=config.shortage_penalty_multiplier,
        disrupted_warehouse_count=int(config.shutdown_warehouse is not None),
    )


def result_snapshot(
    name: str,
    result: object,
    config: ScenarioConfig,
    shortage_probability: float | None,
) -> dict[str, object]:
    """Create a presentation-only session snapshot from solved outputs."""
    metrics = result.analytics
    active_ids = result.warehouses.loc[
        result.warehouses["active"] > 0.5, "warehouse_id"
    ].tolist()
    return {
        "scenario": name,
        "inputs": scenario_inputs(config),
        "total_cost": float(metrics["total_cost"]),
        "service_level": float(metrics["service_level"]),
        "active_warehouses": int(metrics["active_warehouses"]),
        "unmet_demand": float(metrics["total_unmet_demand"]),
        "average_utilization": average_active_utilization(result),
        "shortage_probability": shortage_probability,
        "shortage_risk": "Elevated"
        if float(metrics["total_unmet_demand"]) > 1e-5
        else "Low",
        "solver": result.solver_name,
        "active_warehouse_ids": active_ids,
    }


def standardized_sensitivity_data(
    experiment_data: pd.DataFrame, coefficients: pd.DataFrame
) -> pd.DataFrame:
    """Scale coefficients by predictor and response standard deviations."""
    sample = successful_scenarios(experiment_data)
    features = pd.DataFrame(
        {
            "log_demand_multiplier": np.log(
                sample["demand_multiplier"].clip(lower=1e-9)
            ),
            "transport_cost_inflation": sample["transport_cost_inflation"],
            "capacity_reduction": sample["capacity_reduction"],
            "shortage_penalty_multiplier": sample[
                "shortage_penalty_multiplier"
            ],
            "disrupted_warehouse_count": sample["disrupted_warehouse_count"],
            "demand_capacity_interaction": sample["demand_multiplier"]
            * sample["capacity_reduction"],
        }
    )
    response_scale = float(np.log(sample["total_cost"]).std(ddof=0))
    estimates = coefficients.set_index("term")["estimate"]
    rows = []
    for term, label in TERM_LABELS.items():
        standardized_effect = (
            float(estimates[term]) * float(features[term].std(ddof=0)) / response_scale
            if response_scale > 0
            else 0.0
        )
        rows.append(
            {
                "term": term,
                "driver": label,
                "standardized_effect": standardized_effect,
                "magnitude": abs(standardized_effect),
            }
        )
    return pd.DataFrame(rows).sort_values("magnitude", ascending=False)


def inject_dashboard_css() -> None:
    """Apply a restrained visual system without altering app behavior."""
    st.markdown(
        """
        <style>
        :root {
            --scx-bg: #07101b;
            --scx-surface: rgba(15, 27, 43, 0.92);
            --scx-surface-raised: rgba(18, 34, 53, 0.96);
            --scx-border: rgba(82, 116, 143, 0.34);
            --scx-border-strong: rgba(34, 211, 238, 0.34);
            --scx-cyan: #22d3ee;
            --scx-teal: #2dd4bf;
            --scx-text: #e7f0f8;
            --scx-muted: #91a4b7;
        }

        [data-testid="stAppViewContainer"] { background: var(--scx-bg); }
        [data-testid="stHeader"] { background: rgba(7, 16, 27, 0.84); }
        [data-testid="stSidebar"] {
            background: #0a1421;
            border-right: 1px solid rgba(82, 116, 143, 0.28);
        }
        [data-testid="stSidebar"] [data-testid="stForm"] {
            background: rgba(15, 29, 46, 0.78);
            border: 1px solid rgba(82, 116, 143, 0.24);
            border-radius: 12px;
            padding: 0.75rem;
        }
        .block-container {
            max-width: 1540px;
            padding-top: 1.45rem;
            padding-bottom: 2.5rem;
        }

        .scx-hero {
            background: linear-gradient(110deg, rgba(16, 33, 52, 0.98), rgba(10, 24, 39, 0.98));
            border: 1px solid var(--scx-border-strong);
            border-radius: 16px;
            padding: 1.55rem 1.7rem 1.25rem;
            box-shadow: 0 18px 45px rgba(0, 0, 0, 0.22);
            margin-bottom: 1.1rem;
        }
        .scx-hero-top, .scx-meta-row, .scx-tech-row {
            display: flex;
            align-items: center;
            flex-wrap: wrap;
            gap: 0.55rem;
        }
        .scx-hero-top { justify-content: space-between; gap: 1rem; }
        .scx-brand-kicker, .scx-eyebrow, .scx-sidebar-label {
            color: var(--scx-cyan);
            font-size: 0.68rem;
            font-weight: 700;
            letter-spacing: 0.13em;
            text-transform: uppercase;
        }
        .scx-hero h1 {
            margin: 0.2rem 0 0;
            color: #f4f9fc;
            font-size: clamp(2.15rem, 4vw, 3.25rem);
            letter-spacing: -0.045em;
            line-height: 1;
        }
        .scx-hero-subtitle {
            color: #c9d7e4;
            font-size: 1.02rem;
            font-weight: 600;
            margin-top: 0.55rem;
        }
        .scx-hero-description {
            color: var(--scx-muted);
            max-width: 920px;
            line-height: 1.55;
            margin: 0.55rem 0 1rem;
        }
        .scx-badge {
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
            padding: 0.34rem 0.58rem;
            border-radius: 999px;
            border: 1px solid rgba(82, 116, 143, 0.4);
            background: rgba(7, 16, 27, 0.58);
            color: #c9d7e4;
            font-size: 0.7rem;
            font-weight: 650;
            letter-spacing: 0.045em;
        }
        .scx-badge-good { border-color: rgba(45, 212, 191, 0.5); color: #7de9d8; }
        .scx-badge-warn { border-color: rgba(248, 113, 113, 0.5); color: #fca5a5; }
        .scx-status-dot {
            width: 0.42rem;
            height: 0.42rem;
            border-radius: 50%;
            background: currentColor;
            box-shadow: 0 0 12px currentColor;
        }
        .scx-meta-row {
            padding-top: 0.8rem;
            margin-top: 0.85rem;
            border-top: 1px solid rgba(82, 116, 143, 0.24);
            color: var(--scx-muted);
            font-size: 0.75rem;
        }
        .scx-meta-separator { color: #486076; }
        .scx-tech-row { margin-left: auto; }
        .scx-tech {
            color: #a8bbcb;
            font-size: 0.68rem;
            padding: 0.2rem 0.42rem;
            border-radius: 5px;
            background: rgba(34, 211, 238, 0.06);
            border: 1px solid rgba(34, 211, 238, 0.12);
        }

        .scx-section {
            margin: 1.25rem 0 0.75rem;
            padding: 0 0 0.7rem;
            border-bottom: 1px solid rgba(82, 116, 143, 0.22);
        }
        .scx-section h2 {
            color: var(--scx-text);
            font-size: 1.35rem;
            margin: 0.15rem 0 0.2rem;
            letter-spacing: -0.02em;
        }
        .scx-section p { color: var(--scx-muted); margin: 0; font-size: 0.82rem; }

        .scx-metric-card {
            min-height: 126px;
            background: var(--scx-surface);
            border: 1px solid var(--scx-border);
            border-top: 2px solid rgba(34, 211, 238, 0.62);
            border-radius: 12px;
            padding: 0.92rem 1rem 0.82rem;
            box-shadow: 0 12px 28px rgba(0, 0, 0, 0.16);
            margin-bottom: 0.55rem;
        }
        .scx-metric-card.teal { border-top-color: rgba(45, 212, 191, 0.7); }
        .scx-metric-card.alert { border-top-color: rgba(248, 113, 113, 0.72); }
        .scx-metric-label {
            color: #9fb0c0;
            font-size: 0.68rem;
            font-weight: 700;
            letter-spacing: 0.075em;
            text-transform: uppercase;
        }
        .scx-metric-value {
            color: #f4f9fc;
            font-size: clamp(1.35rem, 2.2vw, 1.9rem);
            font-weight: 700;
            letter-spacing: -0.035em;
            line-height: 1.2;
            margin: 0.42rem 0 0.28rem;
            white-space: nowrap;
        }
        .scx-metric-note { color: var(--scx-muted); font-size: 0.72rem; line-height: 1.35; }

        .scx-decision-card {
            min-height: 276px;
            background: rgba(10, 24, 39, 0.65);
            border: 1px solid rgba(82, 116, 143, 0.24);
            border-radius: 10px;
            padding: 1rem 1.05rem;
        }
        .scx-decision-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 0.65rem;
            margin: 0.8rem 0;
        }
        .scx-decision-item {
            background: rgba(18, 34, 53, 0.72);
            border: 1px solid rgba(82, 116, 143, 0.2);
            border-radius: 8px;
            padding: 0.62rem 0.7rem;
        }
        .scx-decision-item span { color: var(--scx-muted); font-size: 0.66rem; display: block; }
        .scx-decision-item strong { color: #edf6fb; font-size: 0.9rem; }
        .scx-takeaway {
            color: #bfeee7;
            background: rgba(45, 212, 191, 0.07);
            border-left: 2px solid var(--scx-teal);
            border-radius: 0 6px 6px 0;
            padding: 0.7rem 0.8rem;
            font-size: 0.78rem;
            line-height: 1.45;
        }
        .scx-control-summary {
            background: rgba(7, 16, 27, 0.6);
            border: 1px solid rgba(82, 116, 143, 0.26);
            border-radius: 9px;
            padding: 0.72rem 0.78rem;
            margin: 0.75rem 0 0.45rem;
            color: #b8c8d5;
            font-size: 0.73rem;
            line-height: 1.55;
        }
        .scx-control-summary strong { color: #eef6fb; font-weight: 650; }
        .scx-execution-state {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0.5rem 0.7rem;
            margin: 0.45rem 0 0.7rem;
            border-radius: 8px;
            background: rgba(45, 212, 191, 0.06);
            border: 1px solid rgba(45, 212, 191, 0.22);
            color: #9fb3c4;
            font-size: 0.68rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }
        .scx-execution-state strong { color: #79e6d5; }
        [data-testid="stSidebar"] [data-testid="stFormSubmitButton"] button {
            min-height: 3rem;
            background: #0f766e;
            border: 1px solid #2dd4bf;
            color: #f4fbfa;
            font-weight: 750;
            letter-spacing: 0.075em;
            box-shadow: 0 8px 18px rgba(15, 118, 110, 0.2);
        }
        [data-testid="stSidebar"] [data-testid="stFormSubmitButton"] button:hover {
            background: #11877e;
            border-color: #67e8d8;
        }
        .scx-status-grid {
            display: grid;
            grid-template-columns: repeat(5, minmax(0, 1fr));
            border-top: 1px solid rgba(82, 116, 143, 0.24);
            margin-top: 0.95rem;
            padding-top: 0.85rem;
            gap: 0.6rem;
        }
        .scx-status-cell { min-width: 0; }
        .scx-status-cell span {
            display: block;
            color: #71879a;
            font-size: 0.61rem;
            font-weight: 700;
            letter-spacing: 0.105em;
            text-transform: uppercase;
            margin-bottom: 0.2rem;
        }
        .scx-status-cell strong {
            display: block;
            color: #e7f0f8;
            font-size: 0.78rem;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .scx-run-complete {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 1rem;
            flex-wrap: wrap;
            padding: 0.78rem 0.95rem;
            margin: 0 0 0.85rem;
            border: 1px solid rgba(45, 212, 191, 0.3);
            border-left: 3px solid #2dd4bf;
            border-radius: 9px;
            background: rgba(45, 212, 191, 0.055);
            color: #a9bdcb;
            font-size: 0.75rem;
        }
        .scx-run-complete strong { color: #dff8f3; font-size: 0.85rem; }
        .scx-decision-brief {
            background: rgba(16, 31, 49, 0.88);
            border: 1px solid rgba(34, 211, 238, 0.26);
            border-left: 3px solid #22d3ee;
            border-radius: 10px;
            padding: 0.95rem 1.05rem;
            margin-bottom: 0.85rem;
            color: #c8d7e2;
            font-size: 0.84rem;
            line-height: 1.6;
        }
        .scx-decision-brief strong {
            display: block;
            color: #f0f7fb;
            font-size: 0.78rem;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            margin-bottom: 0.35rem;
        }
        .scx-change-card {
            min-height: 148px;
            background: rgba(15, 27, 43, 0.92);
            border: 1px solid rgba(82, 116, 143, 0.3);
            border-radius: 11px;
            padding: 0.85rem 0.9rem;
            margin-bottom: 0.5rem;
        }
        .scx-change-path { color: #eef6fb; font-weight: 700; font-size: 1.02rem; margin: 0.48rem 0 0.35rem; }
        .scx-change-detail { color: #8fa6b8; font-size: 0.7rem; line-height: 1.45; }
        .scx-change-detail strong { color: #bdeee6; }

        [data-testid="stVerticalBlockBorderWrapper"] {
            background: var(--scx-surface);
            border-color: var(--scx-border) !important;
            border-radius: 12px !important;
            box-shadow: 0 14px 32px rgba(0, 0, 0, 0.14);
        }
        [data-testid="stMetric"] {
            background: rgba(15, 29, 46, 0.82);
            border-color: rgba(82, 116, 143, 0.3) !important;
            border-radius: 10px;
        }
        [data-testid="stMetricValue"] { color: #f3f8fc; letter-spacing: -0.025em; }
        [data-testid="stMetricLabel"] { color: #a7b8c7; }
        [data-baseweb="tab-list"] {
            gap: 0.25rem;
            border-bottom: 1px solid rgba(82, 116, 143, 0.24);
        }
        [data-baseweb="tab"] {
            background: transparent;
            border-radius: 7px 7px 0 0;
            padding-left: 0.8rem;
            padding-right: 0.8rem;
        }
        [aria-selected="true"][data-baseweb="tab"] {
            background: rgba(34, 211, 238, 0.07);
        }
        .scx-method-badge {
            display: inline-block;
            color: #93e9df;
            background: rgba(45, 212, 191, 0.08);
            border: 1px solid rgba(45, 212, 191, 0.26);
            border-radius: 999px;
            padding: 0.34rem 0.62rem;
            font-size: 0.7rem;
            font-weight: 650;
            letter-spacing: 0.035em;
        }
        .scx-sidebar-label {
            margin: 0.8rem 0 0.15rem;
            padding-top: 0.55rem;
            border-top: 1px solid rgba(82, 116, 143, 0.18);
        }
        .scx-footer {
            margin-top: 2rem;
            padding: 1rem 0.2rem 0.25rem;
            border-top: 1px solid rgba(82, 116, 143, 0.24);
            display: flex;
            justify-content: space-between;
            flex-wrap: wrap;
            gap: 0.7rem;
            color: var(--scx-muted);
            font-size: 0.72rem;
        }
        .scx-footer strong { color: #cbd9e5; font-weight: 600; }
        @media (max-width: 760px) {
            .scx-hero { padding: 1.2rem; }
            .scx-tech-row { margin-left: 0; }
            .scx-decision-grid { grid-template-columns: 1fr; }
            .scx-status-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_section_header(kicker: str, title: str, description: str) -> None:
    """Render a consistent section introduction."""
    st.markdown(
        f"""
        <div class="scx-section">
            <div class="scx-eyebrow">{escape(kicker)}</div>
            <h2>{escape(title)}</h2>
            <p>{escape(description)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_card(label: str, value: str, note: str, tone: str = "cyan") -> None:
    """Render an executive KPI card with a concise interpretation."""
    st.markdown(
        f"""
        <div class="scx-metric-card {escape(tone)}">
            <div class="scx-metric-label">{escape(label)}</div>
            <div class="scx-metric-value">{escape(value)}</div>
            <div class="scx-metric-note">{escape(note)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_change_card(
    label: str,
    base_value: str,
    scenario_value: str,
    absolute_change: str,
    percentage_change: str,
    direction: str,
) -> None:
    """Render a base-to-scenario comparison card."""
    st.markdown(
        f"""
        <div class="scx-change-card">
            <div class="scx-metric-label">{escape(label)}</div>
            <div class="scx-change-path">{escape(base_value)} → {escape(scenario_value)}</div>
            <div class="scx-change-detail"><strong>{escape(direction)} {escape(absolute_change)}</strong><br>{escape(percentage_change)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


st.set_page_config(
    page_title="SupplyChainX · Decision Science",
    page_icon="SCX",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_dashboard_css()

warehouses, demand_zones, costs = load_data()
warehouse_options = ["No constraint", *warehouses["warehouse_id"].tolist()]
shutdown_options = ["No shutdown", *warehouses["warehouse_id"].tolist()]

st.session_state.setdefault("history", [])
st.session_state.setdefault("saved_scenarios", [])
st.session_state.setdefault("scenario_configs", {})
st.session_state.setdefault("scenario_preset", "Base Network")
st.session_state.setdefault("scenario_name_input", "Base Network")
st.session_state.setdefault("demand_input", 1.0)
st.session_state.setdefault("transport_input", 0)
st.session_state.setdefault("capacity_warehouse_input", "No constraint")
st.session_state.setdefault("capacity_input", 0)
st.session_state.setdefault("shutdown_input", "No shutdown")
st.session_state.setdefault("penalty_input", 1.0)
st.session_state.setdefault("decision_mode", "Operations Mode")
st.session_state.setdefault("execution_state", "READY")


def apply_selected_preset() -> None:
    """Populate existing controls from the selected presentation preset."""
    preset_name = st.session_state.scenario_preset
    preset = SCENARIO_PRESETS[preset_name]
    if preset is None:
        st.session_state.scenario_name_input = "Custom Scenario"
        return
    st.session_state.scenario_name_input = preset_name
    st.session_state.demand_input = preset["demand_multiplier"]
    st.session_state.transport_input = preset["transport_inflation_pct"]
    st.session_state.capacity_input = preset["capacity_reduction_pct"]
    st.session_state.capacity_warehouse_input = (
        warehouses["warehouse_id"].iloc[0]
        if preset["capacity_disruption"]
        else "No constraint"
    )
    st.session_state.shutdown_input = (
        warehouses["warehouse_id"].iloc[0]
        if preset["warehouse_outage"]
        else "No shutdown"
    )
    st.session_state.penalty_input = preset["penalty_multiplier"]

with st.sidebar:
    st.header("Decision command center")
    st.caption("Configure, solve, and compare an exact network decision.")
    decision_mode = st.radio(
        "Decision mode",
        ["Operations Mode", "Analytics Mode"],
        horizontal=True,
        key="decision_mode",
        help="Changes presentation priority only; calculations remain identical.",
    )
    st.selectbox(
        "Scenario preset",
        list(SCENARIO_PRESETS),
        key="scenario_preset",
        on_change=apply_selected_preset,
        help="Presets populate the existing scenario controls and remain editable.",
    )
    with st.form("scenario_controls", border=False):
        scenario_name = st.text_input(
            "Scenario name",
            key="scenario_name_input",
            help="Use a distinct name to retain this run in scenario comparison.",
        )

        st.markdown('<div class="scx-sidebar-label">Demand</div>', unsafe_allow_html=True)
        demand_multiplier = st.slider(
            "Demand multiplier",
            0.50,
            1.75,
            step=0.05,
            key="demand_input",
            help="Scales demand in every customer zone.",
        )

        st.markdown('<div class="scx-sidebar-label">Cost</div>', unsafe_allow_html=True)
        transport_inflation_pct = st.slider(
            "Transport-cost inflation",
            -20,
            75,
            step=5,
            key="transport_input",
            format="%d%%",
            help="Applies a common percentage adjustment to all transport lanes.",
        )

        st.markdown('<div class="scx-sidebar-label">Capacity</div>', unsafe_allow_html=True)
        reduced_warehouse = st.selectbox(
            "Capacity-constrained warehouse",
            warehouse_options,
            key="capacity_warehouse_input",
        )
        capacity_reduction_pct = st.slider(
            "Capacity reduction",
            0,
            100,
            step=5,
            key="capacity_input",
            format="%d%%",
            help="Reduces available capacity at the selected warehouse.",
        )

        st.markdown('<div class="scx-sidebar-label">Disruption</div>', unsafe_allow_html=True)
        shutdown_warehouse = st.selectbox(
            "Warehouse shutdown", shutdown_options, key="shutdown_input"
        )

        st.markdown(
            '<div class="scx-sidebar-label">Service penalty</div>',
            unsafe_allow_html=True,
        )
        penalty_multiplier = st.slider(
            "Shortage-penalty multiplier",
            0.50,
            3.00,
            step=0.10,
            key="penalty_input",
            help="Controls how strongly the optimizer penalizes unmet demand.",
        )
        demand_change_pct = 100 * (demand_multiplier - 1.0)
        disrupted_count = int(shutdown_warehouse != "No shutdown")
        st.markdown(
            f"""
            <div class="scx-control-summary"><strong>Current scenario</strong><br>
            Demand {demand_change_pct:+.0f}% · Transport cost {transport_inflation_pct:+d}%<br>
            Capacity −{capacity_reduction_pct:d}% · {disrupted_count} warehouse{'s' if disrupted_count != 1 else ''} disrupted<br>
            Penalty {penalty_multiplier:.2f}×</div>
            """,
            unsafe_allow_html=True,
        )
        stress_selected = any(
            [
                abs(demand_multiplier - 1.0) > 1e-9,
                transport_inflation_pct != 0,
                capacity_reduction_pct > 0,
                disrupted_count > 0,
                abs(penalty_multiplier - 1.0) > 1e-9,
            ]
        )
        action_label = (
            "RUN NETWORK STRESS TEST" if stress_selected else "OPTIMIZE NETWORK"
        )
        st.markdown('<div class="scx-sidebar-label">Action</div>', unsafe_allow_html=True)
        st.caption(
            "Solve the active network scenario and generate the recommended distribution plan."
        )
        execution_placeholder = st.empty()
        run = st.form_submit_button(
            action_label, type="primary", width="stretch"
        )
    st.caption("Exact MILP solve · Gurobi → SCIP → HiGHS cloud fallback")

run_completed = False
if run or "result" not in st.session_state:
    config = ScenarioConfig(
        name=scenario_name if run else "Base Network",
        demand_multiplier=demand_multiplier if run else 1.0,
        transport_cost_inflation=transport_inflation_pct / 100 if run else 0.0,
        capacity_reduction_warehouse=(
            None if reduced_warehouse == "No constraint" or not run else reduced_warehouse
        ),
        capacity_reduction_fraction=capacity_reduction_pct / 100 if run else 0.0,
        shutdown_warehouse=(
            None if shutdown_warehouse == "No shutdown" or not run else shutdown_warehouse
        ),
        shortage_penalty_multiplier=penalty_multiplier if run else 1.0,
    )
    try:
        st.session_state.execution_state = "SOLVING"
        execution_placeholder.markdown(
            '<div class="scx-execution-state"><span>Execution state</span><strong>SOLVING</strong></div>',
            unsafe_allow_html=True,
        )
        with st.spinner("Solving the active network scenario..."):
            scenario_data = apply_scenario(warehouses, demand_zones, costs, config)
            result = solve_network(*scenario_data)
        completion_label, _ = solver_status_label(result.termination_condition)
        st.session_state.execution_state = completion_label
        st.session_state.result = result
        st.session_state.current_scenario = config.name
        st.session_state.current_config = config
        st.session_state.scenario_configs[config.name] = config
        st.session_state.last_run_at = datetime.now().astimezone().strftime(
            "%d %b %Y · %H:%M %Z"
        )
        solved_probability = shortage_probability_for_config(config)
        solved_snapshot = result_snapshot(
            config.name, result, config, solved_probability
        )
        if not is_stress_config(config):
            st.session_state.base_snapshot = solved_snapshot
        row = scenario_summary(config.name, result)
        st.session_state.history = [
            previous
            for previous in st.session_state.history
            if previous["scenario"] != config.name
        ] + [row]
        run_completed = True
    except Exception as exc:
        st.session_state.execution_state = (
            "INFEASIBLE" if "infeasible" in str(exc).lower() else "WARNING"
        )
        execution_placeholder.markdown(
            f'<div class="scx-execution-state"><span>Execution state</span><strong>{st.session_state.execution_state}</strong></div>',
            unsafe_allow_html=True,
        )
        st.error(f"Optimization failed: {exc}")
        st.stop()

execution_placeholder.markdown(
    f'<div class="scx-execution-state"><span>Execution state</span><strong>{escape(st.session_state.execution_state)}</strong></div>',
    unsafe_allow_html=True,
)

result = st.session_state.result
metrics = result.analytics
active_warehouses = result.warehouses.loc[result.warehouses["active"] > 0.5]
average_utilization = average_active_utilization(result)
current_config = st.session_state.current_config
current_shortage_probability = shortage_probability_for_config(current_config)
current_snapshot = result_snapshot(
    st.session_state.current_scenario,
    result,
    current_config,
    current_shortage_probability,
)
if "base_snapshot" not in st.session_state and not is_stress_config(current_config):
    st.session_state.base_snapshot = current_snapshot
base_snapshot = st.session_state.get("base_snapshot", current_snapshot)
status_label, _ = solver_status_label(result.termination_condition)
status_class = "scx-badge-good" if status_label in {"OPTIMAL", "FEASIBLE"} else "scx-badge-warn"
scenario_display = escape(str(st.session_state.current_scenario))
last_run_display = escape(st.session_state.get("last_run_at", "Current session"))
network_state = "STRESSED" if is_stress_config(current_config) else "BASE"
if (
    metrics["total_unmet_demand"] > 1e-5
    or metrics["service_level"] < 0.95
    or (current_shortage_probability is not None and current_shortage_probability >= 0.65)
):
    risk_label = "HIGH"
elif metrics["service_level"] < 0.995 or (
    current_shortage_probability is not None and current_shortage_probability >= 0.30
):
    risk_label = "MODERATE"
else:
    risk_label = "LOW"
st.markdown(
    f"""
    <section class="scx-hero">
        <div class="scx-hero-top">
            <div>
                <div class="scx-brand-kicker">Network intelligence platform</div>
                <h1>SupplyChainX</h1>
                <div class="scx-hero-subtitle">Network Optimization • Stress Testing • Decision Intelligence</div>
            </div>
            <div class="scx-hero-top">
                <span class="scx-badge {status_class}"><span class="scx-status-dot"></span>{status_label}</span>
                <span class="scx-badge">Scenario · {scenario_display}</span>
            </div>
        </div>
        <p class="scx-hero-description">A command center for optimizing multi-warehouse distribution decisions, testing operational shocks, and translating modeled responses into executive action.</p>
        <div class="scx-status-grid">
            <div class="scx-status-cell"><span>Network status</span><strong>{network_state}</strong></div>
            <div class="scx-status-cell"><span>Solver</span><strong>{escape(solver_display_name(result.solver_name))}</strong></div>
            <div class="scx-status-cell"><span>Last run</span><strong>{last_run_display}</strong></div>
            <div class="scx-status-cell"><span>Service</span><strong>{metrics['service_level']:.1%}</strong></div>
            <div class="scx-status-cell"><span>Risk</span><strong>{risk_label}</strong></div>
        </div>
        <div class="scx-meta-row">
            <span>Model · Exact MILP</span><span class="scx-meta-separator">•</span><span>Mode · {escape(decision_mode)}</span>
            <div class="scx-tech-row"><span class="scx-tech">Pyomo</span><span class="scx-tech">Gurobi</span><span class="scx-tech">SCIP</span><span class="scx-tech">HiGHS</span><span class="scx-tech">Streamlit</span><span class="scx-tech">Statsmodels</span></div>
        </div>
    </section>
    """,
    unsafe_allow_html=True,
)

if run_completed:
    st.markdown(
        f"""
        <div class="scx-run-complete"><div><strong>Optimization Complete</strong><br>Recommended distribution plan generated successfully.</div>
        <div>Solver · {escape(solver_display_name(result.solver_name))} &nbsp;|&nbsp; Termination · {escape(str(result.termination_condition).upper())} &nbsp;|&nbsp; Objective · {escape(money(metrics['total_cost']))}</div></div>
        """,
        unsafe_allow_html=True,
    )

render_section_header(
    "Executive overview",
    "Current network posture",
    "The exact optimized position for cost, service, facility activation, and capacity use.",
)
kpi_rows = [
    [
        (
            "Optimal network cost",
            money(metrics["total_cost"]),
            "Minimum feasible spend for this scenario",
            "cyan",
        ),
        (
            "Service level",
            f"{metrics['service_level']:.1%}",
            "Full demand coverage" if metrics["service_level"] >= 0.9999 else "Service exposure detected",
            "teal" if metrics["service_level"] >= 0.9999 else "alert",
        ),
        (
            "Active warehouses",
            f"{metrics['active_warehouses']}",
            f"{len(warehouses) - int(metrics['active_warehouses'])} facilities held in reserve",
            "cyan",
        ),
    ],
    [
        (
            "Unmet demand",
            f"{metrics['total_unmet_demand']:,.0f}",
            "No shortage exposure" if metrics["total_unmet_demand"] <= 1e-5 else "Immediate service attention required",
            "teal" if metrics["total_unmet_demand"] <= 1e-5 else "alert",
        ),
        (
            "Average utilization",
            f"{average_utilization:.1%}",
            "Across operating facilities",
            "cyan",
        ),
        (
            "Solver used",
            solver_display_name(result.solver_name),
            f"Exact MILP · {status_label.lower()} termination",
            "teal",
        ),
    ],
]
for card_row in kpi_rows:
    columns = st.columns(3)
    for column, card in zip(columns, card_row):
        with column:
            render_metric_card(*card)

cost_components = {
    "Transport": float(metrics["transport_cost"]),
    "Fixed operations": float(metrics["fixed_cost"]),
    "Shortage": float(metrics["shortage_cost"]),
}
dominant_cost_name, dominant_cost_value = max(cost_components.items(), key=lambda item: item[1])
dominant_cost_share = dominant_cost_value / metrics["total_cost"] if metrics["total_cost"] else 0.0
shortage_status = (
    "No shortage"
    if metrics["total_unmet_demand"] <= 1e-5
    else f"{metrics['total_unmet_demand']:,.0f} units unmet"
)
if metrics["total_unmet_demand"] <= 1e-5:
    decision_takeaway = (
        f"The network meets modeled demand with {metrics['active_warehouses']} facilities. "
        f"Monitor {metrics['highest_utilization_warehouse']} as the primary capacity pressure point."
    )
else:
    decision_takeaway = (
        "The current stress case creates a service gap. Review disrupted capacity and the "
        "shortage-penalty setting before accepting this network posture."
    )

if decision_mode == "Operations Mode":
    (
        tab_overview,
        tab_network,
        tab_tables,
        tab_scenarios,
        tab_econometrics,
    ) = st.tabs(
        [
            "Executive Overview",
            "Network Optimization",
            "Decision Detail",
            "Scenario Stress Testing",
            "Econometric Insights",
        ]
    )
else:
    (
        tab_overview,
        tab_econometrics,
        tab_scenarios,
        tab_network,
        tab_tables,
    ) = st.tabs(
        [
            "Executive Overview",
            "Econometric Insights",
            "Scenario Stress Testing",
            "Network Optimization",
            "Decision Detail",
        ]
    )

with tab_overview:
    render_section_header(
        "Network optimization",
        "Optimized decision surface",
        "A balanced view of cost composition, facility pressure, shipment concentration, and the recommended operating posture.",
    )
    base_cost = float(base_snapshot["total_cost"])
    cost_delta = float(current_snapshot["total_cost"]) - base_cost
    cost_delta_pct = cost_delta / base_cost if base_cost else np.nan
    service_delta = float(current_snapshot["service_level"]) - float(
        base_snapshot["service_level"]
    )
    warehouse_delta = int(current_snapshot["active_warehouses"]) - int(
        base_snapshot["active_warehouses"]
    )
    cost_language = (
        f"increases by {abs(cost_delta_pct):.1%}"
        if cost_delta_pct > 0.005
        else f"decreases by {abs(cost_delta_pct):.1%}"
        if cost_delta_pct < -0.005
        else "remains broadly unchanged"
    )
    facility_language = (
        f"opens {warehouse_delta} additional warehouse{'s' if warehouse_delta != 1 else ''}"
        if warehouse_delta > 0
        else f"operates {abs(warehouse_delta)} fewer warehouse{'s' if warehouse_delta != -1 else ''}"
        if warehouse_delta < 0
        else "retains the base facility footprint"
    )
    shortage_language = (
        f"Shortages occur, leaving {metrics['total_unmet_demand']:,.0f} units unmet."
        if metrics["total_unmet_demand"] > 1e-5
        else "No modeled shortages occur."
    )
    decision_brief = (
        f"Under {st.session_state.current_scenario}, SupplyChainX recommends operating "
        f"{metrics['active_warehouses']} warehouses to deliver a {metrics['service_level']:.1%} service level. "
        f"Total network cost {cost_language} relative to the base network, with {dominant_cost_name.lower()} "
        f"as the largest cost component. The plan {facility_language}; "
        f"{metrics['highest_utilization_warehouse']} is the most utilized facility. {shortage_language}"
    )
    st.markdown(
        f'<div class="scx-decision-brief"><strong>Decision Brief</strong>{escape(decision_brief)}</div>',
        unsafe_allow_html=True,
    )
    overview_left, overview_right = st.columns([0.88, 1.12])
    with overview_left:
        with st.container(border=True):
            st.markdown("**Cost composition**")
            st.caption("Share of exact optimized spend by operating component.")
            cost_figure = cost_breakdown_chart(metrics)
            cost_figure.update_layout(height=295)
            st.plotly_chart(cost_figure)
    with overview_right:
        with st.container(border=True):
            st.markdown("**Executive decision summary**")
            st.caption("Operational interpretation derived from the current exact solution.")
            st.markdown(
                f"""
                <div class="scx-decision-card">
                    <div class="scx-decision-grid">
                        <div class="scx-decision-item"><span>Optimized cost</span><strong>{escape(money(metrics['total_cost']))}</strong></div>
                        <div class="scx-decision-item"><span>Dominant cost</span><strong>{escape(dominant_cost_name)} · {dominant_cost_share:.1%}</strong></div>
                        <div class="scx-decision-item"><span>Network service</span><strong>{metrics['service_level']:.1%}</strong></div>
                        <div class="scx-decision-item"><span>Capacity pressure</span><strong>{escape(str(metrics['highest_utilization_warehouse']))}</strong></div>
                        <div class="scx-decision-item"><span>Shortage status</span><strong>{escape(shortage_status)}</strong></div>
                        <div class="scx-decision-item"><span>Shipment lanes used</span><strong>{metrics['shipment_lanes_used']:,}</strong></div>
                    </div>
                    <div class="scx-takeaway"><strong>Decision takeaway</strong><br>{escape(decision_takeaway)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    priority_left, priority_right = st.columns(2)
    with priority_left:
        with st.container(border=True):
            st.markdown("**Warehouse utilization ranking**")
            st.caption("Highest-loaded active facilities in the optimized network.")
            st.plotly_chart(
                top_utilized_warehouses_chart(result.warehouses),
                key="overview_top_utilization",
            )
    with priority_right:
        with st.container(border=True):
            st.markdown("**Shipment lane ranking**")
            st.caption("Largest optimized flows by warehouse-to-zone lane.")
            st.plotly_chart(
                largest_shipment_lanes_chart(result.shipments),
                key="overview_largest_lanes",
            )

    render_section_header(
        "Before vs after",
        "Base Network vs Optimized Scenario",
        "Absolute and relative movement from the validated base-network solution.",
    )
    comparison_definitions = [
        (
            "Total cost",
            money(float(base_snapshot["total_cost"])),
            money(float(current_snapshot["total_cost"])),
            money(abs(cost_delta)),
            f"{cost_delta_pct:+.1%}" if np.isfinite(cost_delta_pct) else "Not available",
            "↑" if cost_delta > 1e-6 else "↓" if cost_delta < -1e-6 else "→",
        ),
        (
            "Service level",
            f"{float(base_snapshot['service_level']):.1%}",
            f"{float(current_snapshot['service_level']):.1%}",
            f"{abs(service_delta) * 100:.1f} pp",
            (
                f"{service_delta / float(base_snapshot['service_level']):+.1%}"
                if float(base_snapshot["service_level"])
                else "Not available"
            ),
            "↑" if service_delta > 1e-9 else "↓" if service_delta < -1e-9 else "→",
        ),
        (
            "Active warehouses",
            f"{int(base_snapshot['active_warehouses'])}",
            f"{int(current_snapshot['active_warehouses'])}",
            f"{abs(warehouse_delta)} facilities",
            f"{warehouse_delta / int(base_snapshot['active_warehouses']):+.1%}",
            "↑" if warehouse_delta > 0 else "↓" if warehouse_delta < 0 else "→",
        ),
    ]
    unmet_delta = float(current_snapshot["unmet_demand"]) - float(
        base_snapshot["unmet_demand"]
    )
    utilization_delta = float(current_snapshot["average_utilization"]) - float(
        base_snapshot["average_utilization"]
    )
    base_probability = base_snapshot.get("shortage_probability")
    probability_delta = (
        float(current_shortage_probability) - float(base_probability)
        if current_shortage_probability is not None and base_probability is not None
        else None
    )
    comparison_definitions.extend(
        [
            (
                "Unmet demand",
                f"{float(base_snapshot['unmet_demand']):,.0f}",
                f"{float(current_snapshot['unmet_demand']):,.0f}",
                f"{abs(unmet_delta):,.0f} units",
                (
                    f"{unmet_delta / float(base_snapshot['unmet_demand']):+.1%}"
                    if float(base_snapshot["unmet_demand"])
                    else "No percentage from zero base"
                ),
                "↑" if unmet_delta > 1e-6 else "↓" if unmet_delta < -1e-6 else "→",
            ),
            (
                "Average utilization",
                f"{float(base_snapshot['average_utilization']):.1%}",
                f"{float(current_snapshot['average_utilization']):.1%}",
                f"{abs(utilization_delta) * 100:.1f} pp",
                (
                    f"{utilization_delta / float(base_snapshot['average_utilization']):+.1%}"
                    if float(base_snapshot["average_utilization"])
                    else "Not available"
                ),
                "↑" if utilization_delta > 1e-9 else "↓" if utilization_delta < -1e-9 else "→",
            ),
            (
                "Shortage probability",
                f"{float(base_probability):.1%}" if base_probability is not None else "Unavailable",
                (
                    f"{float(current_shortage_probability):.1%}"
                    if current_shortage_probability is not None
                    else "Unavailable"
                ),
                f"{abs(probability_delta) * 100:.1f} pp" if probability_delta is not None else "Not available",
                "Econometric surrogate estimate",
                "↑" if probability_delta is not None and probability_delta > 1e-9 else "↓" if probability_delta is not None and probability_delta < -1e-9 else "→",
            ),
        ]
    )
    for comparison_row in [comparison_definitions[:3], comparison_definitions[3:]]:
        comparison_columns = st.columns(3)
        for column, comparison in zip(comparison_columns, comparison_row):
            with column:
                render_change_card(*comparison)

    render_section_header(
        "Network response",
        "Operational response highlights",
        "The most decision-relevant changes and pressure points in the optimized plan.",
    )
    base_active_ids = set(base_snapshot.get("active_warehouse_ids", []))
    current_active_ids = set(current_snapshot.get("active_warehouse_ids", []))
    activated = sorted(current_active_ids - base_active_ids)
    deactivated = sorted(base_active_ids - current_active_ids)
    if result.shipments.empty:
        largest_lane = "No active lane"
        largest_lane_note = "No shipment quantity"
    else:
        largest_lane_row = result.shipments.nlargest(1, "quantity").iloc[0]
        largest_lane = (
            f"{largest_lane_row['warehouse_id']} → {largest_lane_row['demand_zone_id']}"
        )
        largest_lane_note = f"{float(largest_lane_row['quantity']):,.0f} units"
    response_cards = [
        ("Warehouses activated", ", ".join(activated) if activated else "No change", "Compared with base network", "teal"),
        ("Warehouses deactivated", ", ".join(deactivated) if deactivated else "No change", "Compared with base network", "cyan"),
        ("Highest utilization", str(metrics["highest_utilization_warehouse"]), "Primary facility pressure point", "cyan"),
        ("Largest cost driver", dominant_cost_name, f"{dominant_cost_share:.1%} of optimized cost", "teal"),
        ("Largest shipment lane", largest_lane, largest_lane_note, "cyan"),
        ("Service risk status", risk_label, shortage_status, "alert" if risk_label == "HIGH" else "teal"),
    ]
    for response_row in [response_cards[:3], response_cards[3:]]:
        response_columns = st.columns(3)
        for column, card in zip(response_columns, response_row):
            with column:
                render_metric_card(*card)

with tab_network:
    render_section_header(
        "Network optimization",
        "Flow and capacity diagnostics",
        "Inspect facility utilization, service fulfillment, and shipment intensity across the optimized network.",
    )
    flow_left, flow_right = st.columns(2)
    with flow_left:
        with st.container(border=True):
            st.markdown("**Top utilized warehouses**")
            st.plotly_chart(
                top_utilized_warehouses_chart(result.warehouses),
                key="network_top_utilization",
            )
    with flow_right:
        with st.container(border=True):
            st.markdown("**Largest shipment lanes**")
            st.plotly_chart(
                largest_shipment_lanes_chart(result.shipments),
                key="network_largest_lanes",
            )

    detail_left, detail_right = st.columns(2)
    with detail_left:
        with st.container(border=True):
            st.markdown("**Warehouse utilization**")
            st.plotly_chart(warehouse_utilization_chart(result.warehouses))
    with detail_right:
        with st.container(border=True):
            st.markdown("**Demand served vs unmet**")
            st.plotly_chart(demand_service_chart(result.demand_zones))

    with st.container(border=True):
        st.markdown("**Shipment intensity by lane**")
        st.plotly_chart(shipment_heatmap(result.shipments))

with tab_tables:
    render_section_header(
        "Decision detail",
        "Optimization output",
        "Audit the shipment plan and facility-level operating decisions behind the executive view.",
    )
    st.markdown("**Shipment plan**")
    st.dataframe(result.shipments, hide_index=True)
    st.download_button(
        "Download shipment plan",
        result.shipments.to_csv(index=False),
        "shipments.csv",
        "text/csv",
        icon=":material/download:",
    )

    st.markdown("**Warehouse performance**")
    st.dataframe(
        result.warehouses,
        hide_index=True,
        column_config={
            "utilization": st.column_config.ProgressColumn(
                "Utilization", min_value=0.0, max_value=1.0, format="percent"
            ),
            "fixed_cost": st.column_config.NumberColumn(
                "Fixed cost", format="₹%.0f"
            ),
            "shipped_quantity": st.column_config.NumberColumn(
                "Shipped units", format="%.0f"
            ),
        },
    )
    with st.container(horizontal=True):
        st.download_button(
            "Download warehouse results",
            result.warehouses.to_csv(index=False),
            "warehouse_results.csv",
            "text/csv",
            icon=":material/download:",
        )
        st.download_button(
            "Download demand results",
            result.demand_zones.to_csv(index=False),
            "demand_results.csv",
            "text/csv",
            icon=":material/download:",
        )

with tab_scenarios:
    render_section_header(
        "Scenario stress testing",
        "Decision comparison",
        "Compare saved network decisions against the base case and rank operational trade-offs.",
    )
    history_action_left, history_action_middle, history_action_right = st.columns(
        [1.1, 1.25, 0.75], vertical_alignment="bottom"
    )
    with history_action_left:
        if st.button(
            "Save Scenario",
            type="primary",
            key="save_scenario",
            help="Store the current solved scenario in this browser session.",
        ):
            st.session_state.saved_scenarios = [
                saved
                for saved in st.session_state.saved_scenarios
                if saved["scenario"] != current_snapshot["scenario"]
            ] + [current_snapshot.copy()]
            st.success(f"Saved {current_snapshot['scenario']} to Scenario History.")
    saved_names = [item["scenario"] for item in st.session_state.saved_scenarios]
    remove_name = None
    with history_action_middle:
        if saved_names:
            remove_name = st.selectbox(
                "Remove saved run",
                saved_names,
                key="remove_saved_scenario_name",
            )
        else:
            st.caption("No saved stress scenarios yet.")
    with history_action_right:
        if saved_names and st.button("Remove", key="remove_saved_scenario"):
            st.session_state.saved_scenarios = [
                item
                for item in st.session_state.saved_scenarios
                if item["scenario"] != remove_name
            ]
            st.success(f"Removed {remove_name} from Scenario History.")
        if st.button("Clear history", key="clear_saved_scenarios"):
            st.session_state.saved_scenarios = []
            st.success("Saved scenario history cleared. The base benchmark is retained.")

    comparison_records = [base_snapshot.copy()]
    comparison_records.extend(
        saved
        for saved in st.session_state.saved_scenarios
        if saved["scenario"] != base_snapshot["scenario"]
    )
    history = pd.DataFrame(comparison_records)
    history["delta_cost_vs_base"] = history["total_cost"] - float(
        base_snapshot["total_cost"]
    )
    history["shortage_risk_score"] = np.where(
        history["shortage_probability"].fillna(0) >= 0.5,
        1,
        (history["unmet_demand"].fillna(0) > 1e-5).astype(int),
    )
    history["shortage_risk"] = history["shortage_risk_score"].map(
        {0: "Low", 1: "Elevated"}
    )
    ranking_options = {
        "Total cost": "total_cost",
        "Service level": "service_level",
        "Active warehouses": "active_warehouses",
        "Shortage risk": "shortage_risk_score",
    }
    selector_left, selector_right = st.columns([1.05, 0.95])
    with selector_left:
        selected_scenario_name = st.selectbox(
            "Compare saved scenario",
            history["scenario"].tolist(),
            index=len(history) - 1,
            key="saved_scenario_comparison",
            help="Choose a saved run for comparison with the base case.",
        )
    with selector_right:
        ranking_label = st.selectbox(
            "Rank scenarios by",
            list(ranking_options),
            help="Saved runs remain available for the current dashboard session.",
        )

    selected_row = history.loc[history["scenario"] == selected_scenario_name].iloc[0]
    base_row = history.iloc[0]
    cost_impact = (
        float(selected_row["total_cost"]) / float(base_row["total_cost"]) - 1
        if float(base_row["total_cost"])
        else np.nan
    )
    service_impact_points = 100 * (
        float(selected_row["service_level"]) - float(base_row["service_level"])
    )
    facility_delta = int(selected_row["active_warehouses"]) - int(
        base_row["active_warehouses"]
    )

    scenario_shortage_probability = selected_row.get("shortage_probability")
    if pd.isna(scenario_shortage_probability):
        scenario_shortage_probability = None

    comparison_cards = [
        (
            "Selected scenario",
            str(selected_scenario_name),
            f"Compared with {base_row['scenario']}",
            "cyan",
        ),
        (
            "Cost impact",
            f"{cost_impact:+.1%}",
            f"{money(float(selected_row['total_cost']))} optimized cost",
            "alert" if cost_impact > 0.01 else "teal",
        ),
        (
            "Service impact",
            f"{service_impact_points:+.1f} pp",
            f"{float(selected_row['service_level']):.1%} selected service level",
            "alert" if service_impact_points < -0.01 else "teal",
        ),
        (
            "Facilities used",
            f"{int(selected_row['active_warehouses'])}",
            f"{facility_delta:+d} versus base network",
            "cyan",
        ),
        (
            "Shortage probability",
            (
                f"{scenario_shortage_probability:.1%}"
                if scenario_shortage_probability is not None
                else "Unavailable"
            ),
            "Econometric surrogate estimate",
            (
                "alert"
                if scenario_shortage_probability is not None
                and scenario_shortage_probability >= 0.5
                else "teal"
            ),
        ),
    ]
    first_comparison_row = st.columns(3)
    for column, card in zip(first_comparison_row, comparison_cards[:3]):
        with column:
            render_metric_card(*card)
    second_comparison_row = st.columns([1, 1, 1])
    for column, card in zip(second_comparison_row[:2], comparison_cards[3:]):
        with column:
            render_metric_card(*card)
    with second_comparison_row[2]:
        render_metric_card(
            "Shortage status",
            str(selected_row["shortage_risk"]),
            f"{float(selected_row['unmet_demand']):,.0f} exact unmet units",
            "alert" if float(selected_row["unmet_demand"]) > 1e-5 else "teal",
        )

    if len(history) < 2:
        st.caption("Optimize and save a named stress scenario to build the comparison set.")
    with st.container(border=True):
        st.markdown("**Scenario History ranking**")
        st.caption("Saved decisions ordered by the selected executive criterion.")
        st.plotly_chart(
            scenario_comparison_chart(history, ranking_options[ranking_label]),
            key="saved_scenario_ranking",
        )
    st.dataframe(
        history[
            [
                "scenario",
                "total_cost",
                "delta_cost_vs_base",
                "service_level",
                "active_warehouses",
                "unmet_demand",
                "average_utilization",
                "shortage_risk",
            ]
        ],
        hide_index=True,
        column_config={
            "total_cost": st.column_config.NumberColumn(
                "Total cost", format="₹%.0f"
            ),
            "delta_cost_vs_base": st.column_config.NumberColumn(
                "Δ cost vs base", format="₹%.0f"
            ),
            "service_level": st.column_config.NumberColumn(
                "Service level", format="percent"
            ),
            "unmet_demand": st.column_config.NumberColumn(
                "Unmet demand", format="%.0f"
            ),
            "average_utilization": st.column_config.NumberColumn(
                "Average utilization", format="percent"
            ),
        },
    )

with tab_econometrics:
    render_section_header(
        "Econometric insights",
        "Simulation-response lab",
        "Translate repeated optimized scenarios into a compact, non-causal response surface for rapid sensitivity reading.",
    )
    with st.container(border=True):
        st.markdown(
            '<span class="scx-method-badge">Simulation-response analysis · Synthetic scenario environment · Non-causal</span>',
            unsafe_allow_html=True,
        )
        st.caption(
            "Simulation-response analysis based on optimized synthetic scenarios. "
            "Coefficients describe relationships inside the SupplyChainX experimental "
            "environment and should not be interpreted as real-world causal estimates."
        )

    econometric_path = PROJECT_DIR / "results" / "econometric_scenarios.csv"
    if not econometric_path.exists():
        st.warning(
            "Generate the response dataset with `python -m "
            "src.econometric_experiments` to activate this lab."
        )
    else:
        modified_time = econometric_path.stat().st_mtime
        experiment_data = load_econometric_data(str(econometric_path), modified_time)
        ols_result, logistic_result = fit_response_models(
            str(econometric_path), modified_time
        )

        render_section_header(
            "Response model",
            "Optimized-cost response surface",
            "OLS response for log optimized cost with HC3 robust standard errors.",
        )
        response_cards = [
            ("R²", f"{ols_result.r_squared:.3f}", "Explained response variation", "cyan"),
            (
                "Adjusted R²",
                f"{ols_result.adjusted_r_squared:.3f}",
                "Complexity-adjusted fit",
                "teal",
            ),
            ("RMSE", money(ols_result.rmse), "Cost-scale prediction error", "cyan"),
            (
                "Observations",
                f"{ols_result.observations:,}",
                "Successful optimized scenarios",
                "teal",
            ),
        ]
        response_columns = st.columns(4)
        for column, card in zip(response_columns, response_cards):
            with column:
                render_metric_card(*card)

        with st.expander("Coefficient detail", icon=":material/table_chart:"):
            coefficient_table = ols_result.coefficients.copy()
            coefficient_table["95% confidence interval"] = coefficient_table.apply(
                lambda row: f"[{row['ci_lower']:.4f}, {row['ci_upper']:.4f}]",
                axis=1,
            )
            st.dataframe(
                coefficient_table[
                    [
                        "term",
                        "estimate",
                        "robust_standard_error",
                        "p_value",
                        "95% confidence interval",
                    ]
                ],
                hide_index=True,
            )

        render_section_header(
            "Scenario sensitivity",
            "Scenario Sensitivity Ranking",
            "Scale-safe driver ranking, coefficient uncertainty, and surrogate fit inside the controlled synthetic environment.",
        )
        standardized_ranking = standardized_sensitivity_data(
            experiment_data, ols_result.coefficients
        )
        with st.container(border=True):
            st.markdown("**Standardized response sensitivity**")
            st.caption(
                "Absolute coefficient × predictor standard deviation ÷ log-cost standard deviation. "
                "This makes ranking comparable across differently scaled predictors."
            )
            st.plotly_chart(
                standardized_sensitivity_chart(standardized_ranking),
                key="standardized_sensitivity_ranking",
            )
        sensitivity_left, sensitivity_right = st.columns([1.1, 0.9])
        with sensitivity_left:
            with st.container(border=True):
                st.markdown("**Coefficient estimates with 95% CI**")
                st.plotly_chart(
                    coefficient_ranking_chart(ols_result.coefficients),
                    key="coefficient_confidence_ranking",
                )
        with sensitivity_right:
            with st.container(border=True):
                st.markdown("**Exact vs surrogate optimized cost**")
                st.plotly_chart(
                    style_figure(actual_vs_predicted_chart(ols_result), height=390)
                )

        major_findings = ols_result.coefficients.loc[
            ols_result.coefficients["term"].isin(TERM_LABELS)
        ].copy()
        major_findings["response driver"] = major_findings["term"].map(TERM_LABELS)
        major_findings = major_findings.sort_values("estimate", ascending=False)
        leading_drivers = standardized_ranking.head(2)["driver"].tolist()
        st.markdown(
            f"""
            <div class="scx-takeaway"><strong>Operational interpretation</strong><br>
            The simulated optimized-cost response is most sensitive to {escape(leading_drivers[0].lower())} and {escape(leading_drivers[1].lower())} on a standardized scale. This ranking describes the synthetic response environment and is not causal.</div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("**Priority simulation-response findings**")
        st.dataframe(
            major_findings[
                ["response driver", "estimate", "ci_lower", "ci_upper", "p_value"]
            ],
            hide_index=True,
        )
        st.caption(
            "Raw coefficients and confidence intervals are retained for model transparency; "
            "cross-predictor ranking uses the standardized sensitivity measure above."
        )

        render_section_header(
            "Shortage risk model",
            "Simulated service-risk classification",
            "Logistic response estimates for shortage events generated by the scenario experiment.",
        )
        if logistic_result.available:
            logistic_cards = [
                (
                    "Pseudo R²",
                    f"{logistic_result.diagnostics['pseudo_r_squared']:.3f}",
                    "Logistic model fit",
                    "cyan",
                ),
                (
                    "Shortage-event rate",
                    f"{logistic_result.diagnostics['positive_rate']:.1%}",
                    "Share of simulated shortage cases",
                    "teal",
                ),
                (
                    "Observations",
                    f"{logistic_result.diagnostics['observations']:,}",
                    "Scenarios in classification model",
                    "cyan",
                ),
                (
                    "Converged",
                    "Yes" if logistic_result.diagnostics["converged"] else "No",
                    "Maximum-likelihood status",
                    "teal" if logistic_result.diagnostics["converged"] else "alert",
                ),
            ]
            logistic_columns = st.columns(4)
            for column, card in zip(logistic_columns, logistic_cards):
                with column:
                    render_metric_card(*card)
            with st.expander("Logistic coefficient detail", icon=":material/table_chart:"):
                st.dataframe(
                    logistic_result.coefficients[
                        [
                            "term",
                            "estimate",
                            "robust_standard_error",
                            "p_value",
                            "odds_ratio",
                        ]
                    ],
                    hide_index=True,
                )
            st.caption(
                "Odds ratios classify simulated shortage events; they are not causal "
                "risk estimates."
            )
        else:
            st.warning(logistic_result.warning)

        render_section_header(
            "Scenario response estimator",
            "Exact optimization vs econometric surrogate",
            "Compare an exact MILP solve with the fast response-model estimate at identical controlled settings.",
        )
        with st.form("econometric_response_estimator"):
            estimator_left, estimator_right = st.columns(2)
            with estimator_left:
                estimated_demand = st.slider(
                    "Experimental demand multiplier", 0.80, 1.40, 1.00, 0.01
                )
                estimated_inflation = st.slider(
                    "Experimental transport inflation", -0.10, 0.40, 0.00, 0.01
                )
                estimated_penalty = st.slider(
                    "Experimental shortage-penalty multiplier",
                    0.50,
                    2.00,
                    1.00,
                    0.05,
                )
            with estimator_right:
                estimated_capacity = st.slider(
                    "Experimental capacity reduction", 0.0, 0.50, 0.0, 0.01
                )
                estimated_disruptions = st.slider(
                    "Disrupted warehouse count", 0, 3, 0, 1
                )
            estimate = st.form_submit_button(
                "Compare exact and surrogate results",
                type="primary",
                width="stretch",
            )

        if estimate:
            selected_scenario = {
                "demand_multiplier": estimated_demand,
                "transport_cost_inflation": estimated_inflation,
                "capacity_reduction": estimated_capacity,
                "shortage_penalty_multiplier": estimated_penalty,
                "disrupted_warehouse_count": estimated_disruptions,
            }
            with st.spinner("Running exact Pyomo comparison..."):
                exact_result = run_exact_response_scenario(
                    estimated_demand,
                    estimated_inflation,
                    estimated_penalty,
                    estimated_capacity,
                    estimated_disruptions,
                )
            if str(exact_result["solver_status"]).startswith("failed/"):
                st.error(f"Exact comparison failed: {exact_result['solver_status']}")
            else:
                predicted_cost = predict_optimized_cost(
                    ols_result, **selected_scenario
                )
                shortage_probability = predict_shortage_probability(
                    logistic_result, **selected_scenario
                )
                exact_cost = float(exact_result["total_cost"])
                absolute_difference = abs(predicted_cost - exact_cost)
                percentage_difference = (
                    100 * (predicted_cost - exact_cost) / exact_cost
                    if exact_cost
                    else np.nan
                )

                exact_column, surrogate_column = st.columns(2)
                with exact_column:
                    with st.container(border=True):
                        st.markdown("**1 · Exact Pyomo optimization result**")
                        render_metric_card(
                            "Optimized network cost",
                            money(exact_cost),
                            "Deterministic exact MILP result",
                            "teal",
                        )
                        st.caption(
                            f"Service level {float(exact_result['service_level']):.1%} · "
                            f"{int(exact_result['active_warehouses'])} active warehouses · "
                            f"{exact_result['solver_status']}"
                        )
                with surrogate_column:
                    with st.container(border=True):
                        st.markdown("**2 · Econometric surrogate estimate**")
                        probability_text = (
                            f"{shortage_probability:.1%}"
                            if shortage_probability is not None
                            else "Unavailable"
                        )
                        render_metric_card(
                            "Predicted optimized cost",
                            money(predicted_cost),
                            "Simulation-response model estimate",
                            "cyan",
                        )
                        st.caption(
                            f"Predicted shortage probability {probability_text} · "
                            "Response-model estimate"
                        )

                comparison_output_cards = [
                    (
                        "Absolute prediction error",
                        money(absolute_difference),
                        "Absolute surrogate-to-exact gap",
                        "cyan",
                    ),
                    (
                        "Percentage prediction error",
                        f"{percentage_difference:+.1f}%",
                        "Signed relative model error",
                        "teal",
                    ),
                    (
                        "Shortage probability",
                        (
                            f"{shortage_probability:.1%}"
                            if shortage_probability is not None
                            else "Unavailable"
                        ),
                        "Logistic surrogate estimate",
                        "alert"
                        if shortage_probability is not None and shortage_probability >= 0.5
                        else "teal",
                    ),
                ]
                output_columns = st.columns(3)
                for column, card in zip(output_columns, comparison_output_cards):
                    with column:
                        render_metric_card(*card)
                st.caption(
                    "The exact result solves the MILP for one deterministic experiment "
                    "draw. The surrogate estimates the average modeled response learned "
                    "from the synthetic scenario dataset."
                )

        render_section_header(
            "Model diagnostics",
            "Response-model quality checks",
            "Residual behavior, fitted-value alignment, and correlation structure for the synthetic scenario design.",
        )
        diagnostic_left, diagnostic_right = st.columns(2)
        with diagnostic_left:
            with st.container(border=True):
                st.markdown("**Residuals vs fitted response**")
                st.plotly_chart(
                    style_figure(residual_vs_fitted_chart(ols_result), height=340)
                )
        with diagnostic_right:
            with st.container(border=True):
                st.markdown("**Residual distribution**")
                st.plotly_chart(
                    style_figure(residual_distribution_chart(ols_result), height=340)
                )
        with st.container(border=True):
            st.markdown("**Scenario-variable correlation**")
            st.plotly_chart(
                style_figure(correlation_matrix_chart(experiment_data), height=460)
            )

st.markdown(
    """
    <footer class="scx-footer">
        <div><strong>SupplyChainX</strong> · Operations Research &amp; Decision Analytics<br>Built by Sayak Pranab Ghosh · <a href="https://github.com/sayakdoms/SupplyChainX" target="_blank">GitHub</a> · <a href="https://www.linkedin.com/in/sayakiitr2456/" target="_blank">LinkedIn</a> · <a href="https://supplychainx-optimization.streamlit.app/" target="_blank">Live Demo</a></div>
        <div>Pyomo · Gurobi · SCIP · HiGHS · Streamlit · Statsmodels</div>
    </footer>
    """,
    unsafe_allow_html=True,
)
