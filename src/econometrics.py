"""Simulation-response models for optimized SupplyChainX scenario outcomes.

These regressions are surrogate descriptions of a controlled synthetic
experiment. They are not causal estimates and must not be interpreted as
evidence about observed firms or real-world interventions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm


PREDICTOR_COLUMNS = [
    "log_demand_multiplier",
    "transport_cost_inflation",
    "capacity_reduction",
    "shortage_penalty_multiplier",
    "disrupted_warehouse_count",
    "demand_capacity_interaction",
]
LOGIT_PREDICTOR_COLUMNS = [
    "demand_multiplier",
    "transport_cost_inflation",
    "capacity_reduction",
    "shortage_penalty_multiplier",
    "disrupted_warehouse_count",
    "demand_capacity_interaction",
]
SCENARIO_VARIABLES = [
    "demand_multiplier",
    "transport_cost_inflation",
    "shortage_penalty_multiplier",
    "capacity_reduction",
    "disrupted_warehouse_count",
]


@dataclass
class OLSResponseResult:
    """OLS response model, robust inference, predictions, and diagnostics."""

    model: Any
    coefficients: pd.DataFrame
    predictions: pd.DataFrame
    r_squared: float
    adjusted_r_squared: float
    rmse: float
    observations: int


@dataclass
class LogisticResponseResult:
    """Logistic shortage response model or a graceful availability warning."""

    available: bool
    warning: str | None
    model: Any | None
    coefficients: pd.DataFrame
    predicted_probabilities: pd.DataFrame
    diagnostics: dict[str, float | int | bool | str]


def successful_scenarios(data: pd.DataFrame) -> pd.DataFrame:
    """Return finite, successfully optimized rows eligible for regression."""
    required = set(SCENARIO_VARIABLES + ["total_cost", "shortage_event", "solver_status"])
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Scenario data is missing columns: {sorted(missing)}")
    termination = data["solver_status"].astype(str).str.lower().str.split("/").str[-1]
    valid_status = termination.isin({"optimal", "feasible", "locallyoptimal"})
    numeric = data[SCENARIO_VARIABLES + ["total_cost"]].apply(
        pd.to_numeric, errors="coerce"
    )
    finite = np.isfinite(numeric).all(axis=1) & (numeric["total_cost"] > 0)
    return data.loc[valid_status & finite].copy()


def _feature_matrix(
    data: pd.DataFrame, add_constant: bool = True, log_demand: bool = True
) -> pd.DataFrame:
    demand = pd.to_numeric(data["demand_multiplier"], errors="coerce").clip(lower=1e-9)
    demand_term = np.log(demand) if log_demand else demand
    demand_name = "log_demand_multiplier" if log_demand else "demand_multiplier"
    matrix = pd.DataFrame(
        {
            demand_name: demand_term,
            "transport_cost_inflation": pd.to_numeric(
                data["transport_cost_inflation"], errors="coerce"
            ),
            "capacity_reduction": pd.to_numeric(
                data["capacity_reduction"], errors="coerce"
            ),
            "shortage_penalty_multiplier": pd.to_numeric(
                data["shortage_penalty_multiplier"], errors="coerce"
            ),
            "disrupted_warehouse_count": pd.to_numeric(
                data["disrupted_warehouse_count"], errors="coerce"
            ),
            "demand_capacity_interaction": demand
            * pd.to_numeric(data["capacity_reduction"], errors="coerce"),
        },
        index=data.index,
    )
    return sm.add_constant(matrix, has_constant="add") if add_constant else matrix


def _coefficient_table(model: Any, include_odds: bool = False) -> pd.DataFrame:
    intervals = model.conf_int(alpha=0.05)
    table = pd.DataFrame(
        {
            "term": model.params.index,
            "estimate": np.asarray(model.params),
            "robust_standard_error": np.asarray(model.bse),
            "p_value": np.asarray(model.pvalues),
            "ci_lower": np.asarray(intervals.iloc[:, 0]),
            "ci_upper": np.asarray(intervals.iloc[:, 1]),
        }
    )
    if include_odds:
        table["odds_ratio"] = np.exp(np.clip(table["estimate"], -700, 700))
    return table


def fit_ols_response(data: pd.DataFrame) -> OLSResponseResult:
    """Fit log(total_cost) OLS with HC3 heteroskedasticity-robust inference."""
    sample = successful_scenarios(data)
    if len(sample) <= len(PREDICTOR_COLUMNS) + 1:
        raise ValueError("Too few successful scenarios to fit the OLS response model.")
    design = _feature_matrix(sample, log_demand=True)
    response = np.log(sample["total_cost"].clip(lower=1e-9))
    model = sm.OLS(response, design).fit(cov_type="HC3")
    fitted_cost = np.exp(np.clip(np.asarray(model.fittedvalues), -700, 700))
    residual_log = np.asarray(response) - np.asarray(model.fittedvalues)
    predictions = pd.DataFrame(
        {
            "scenario_id": sample["scenario_id"].to_numpy()
            if "scenario_id" in sample
            else sample.index.astype(str),
            "actual_total_cost": sample["total_cost"].to_numpy(dtype=float),
            "predicted_total_cost": fitted_cost,
            "fitted_log_cost": np.asarray(model.fittedvalues),
            "residual_log_cost": residual_log,
        },
        index=sample.index,
    )
    rmse = float(
        np.sqrt(
            np.mean(
                (predictions["actual_total_cost"] - predictions["predicted_total_cost"])
                ** 2
            )
        )
    )
    return OLSResponseResult(
        model=model,
        coefficients=_coefficient_table(model),
        predictions=predictions,
        r_squared=float(model.rsquared),
        adjusted_r_squared=float(model.rsquared_adj),
        rmse=rmse,
        observations=int(model.nobs),
    )


def fit_shortage_logit(data: pd.DataFrame) -> LogisticResponseResult:
    """Fit a logistic shortage-event response, or return a clear warning."""
    sample = successful_scenarios(data)
    sample = sample.loc[sample["shortage_event"].notna()].copy()
    outcome = pd.to_numeric(sample["shortage_event"], errors="coerce")
    if outcome.nunique(dropna=True) < 2:
        return LogisticResponseResult(
            available=False,
            warning=(
                "Logistic shortage model was not estimated because shortage_event "
                "has no class variation among successful scenarios."
            ),
            model=None,
            coefficients=pd.DataFrame(),
            predicted_probabilities=pd.DataFrame(),
            diagnostics={"observations": int(len(sample)), "classes": int(outcome.nunique())},
        )
    design = _feature_matrix(sample, log_demand=False)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = sm.Logit(outcome.astype(int), design).fit(
                method="lbfgs", maxiter=500, disp=False, cov_type="HC3"
            )
        probabilities = np.asarray(model.predict(design), dtype=float)
    except Exception as exc:
        return LogisticResponseResult(
            available=False,
            warning=f"Logistic shortage model could not be estimated reliably: {exc}",
            model=None,
            coefficients=pd.DataFrame(),
            predicted_probabilities=pd.DataFrame(),
            diagnostics={"observations": int(len(sample)), "classes": 2},
        )
    null_ll = float(model.llnull)
    pseudo_r2 = float(1.0 - model.llf / null_ll) if null_ll else np.nan
    return LogisticResponseResult(
        available=True,
        warning=None,
        model=model,
        coefficients=_coefficient_table(model, include_odds=True),
        predicted_probabilities=pd.DataFrame(
            {
                "scenario_id": sample["scenario_id"].to_numpy()
                if "scenario_id" in sample
                else sample.index.astype(str),
                "shortage_event": outcome.to_numpy(dtype=int),
                "predicted_probability": probabilities,
            },
            index=sample.index,
        ),
        diagnostics={
            "observations": int(model.nobs),
            "positive_rate": float(outcome.mean()),
            "pseudo_r_squared": pseudo_r2,
            "log_likelihood": float(model.llf),
            "aic": float(model.aic),
            "converged": bool(model.mle_retvals.get("converged", False)),
            "model_type": "Logit (HC3 robust standard errors)",
        },
    )


def _single_scenario(
    demand_multiplier: float,
    transport_cost_inflation: float,
    capacity_reduction: float,
    shortage_penalty_multiplier: float,
    disrupted_warehouse_count: int,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "demand_multiplier": demand_multiplier,
                "transport_cost_inflation": transport_cost_inflation,
                "capacity_reduction": capacity_reduction,
                "shortage_penalty_multiplier": shortage_penalty_multiplier,
                "disrupted_warehouse_count": disrupted_warehouse_count,
            }
        ]
    )


def predict_optimized_cost(ols: OLSResponseResult, **scenario: float) -> float:
    """Predict optimized cost from the response model, not from a new MILP solve."""
    design = _feature_matrix(_single_scenario(**scenario), log_demand=True)
    design = design.reindex(columns=ols.model.params.index, fill_value=0.0)
    log_prediction = float(np.asarray(ols.model.predict(design))[0])
    return float(np.exp(np.clip(log_prediction, -700, 700)))


def predict_shortage_probability(
    logistic: LogisticResponseResult, **scenario: float
) -> float | None:
    """Predict shortage probability when a logistic response model is available."""
    if not logistic.available or logistic.model is None:
        return None
    design = _feature_matrix(_single_scenario(**scenario), log_demand=False)
    design = design.reindex(columns=logistic.model.params.index, fill_value=0.0)
    return float(np.asarray(logistic.model.predict(design))[0])


def coefficient_interval_chart(ols: OLSResponseResult) -> go.Figure:
    data = ols.coefficients.loc[ols.coefficients["term"] != "const"].copy()
    fig = go.Figure(
        go.Scatter(
            x=data["estimate"],
            y=data["term"],
            mode="markers",
            error_x={
                "type": "data",
                "symmetric": False,
                "array": data["ci_upper"] - data["estimate"],
                "arrayminus": data["estimate"] - data["ci_lower"],
            },
            marker={"color": "#2A9D8F", "size": 9},
        )
    )
    fig.add_vline(x=0, line_dash="dash", line_color="#64748B")
    fig.update_layout(
        xaxis_title="OLS coefficient (log total cost response)",
        yaxis_title=None,
        margin=dict(l=10, r=10, t=25, b=10),
    )
    return fig


def actual_vs_predicted_chart(ols: OLSResponseResult) -> go.Figure:
    data = ols.predictions
    lower = float(min(data["actual_total_cost"].min(), data["predicted_total_cost"].min()))
    upper = float(max(data["actual_total_cost"].max(), data["predicted_total_cost"].max()))
    fig = px.scatter(
        data,
        x="actual_total_cost",
        y="predicted_total_cost",
        hover_name="scenario_id",
        labels={
            "actual_total_cost": "Exact optimized total cost",
            "predicted_total_cost": "Regression-predicted optimized cost",
        },
    )
    fig.add_shape(type="line", x0=lower, y0=lower, x1=upper, y1=upper, line=dict(dash="dash"))
    fig.update_layout(margin=dict(l=10, r=10, t=25, b=10))
    return fig


def residual_vs_fitted_chart(ols: OLSResponseResult) -> go.Figure:
    fig = px.scatter(
        ols.predictions,
        x="fitted_log_cost",
        y="residual_log_cost",
        hover_name="scenario_id",
        labels={"fitted_log_cost": "Fitted log cost", "residual_log_cost": "Residual"},
    )
    fig.add_hline(y=0, line_dash="dash", line_color="#64748B")
    fig.update_layout(margin=dict(l=10, r=10, t=25, b=10))
    return fig


def residual_distribution_chart(ols: OLSResponseResult) -> go.Figure:
    fig = px.histogram(
        ols.predictions,
        x="residual_log_cost",
        nbins=30,
        labels={"residual_log_cost": "Log-cost residual"},
    )
    fig.update_layout(yaxis_title="Scenario count", margin=dict(l=10, r=10, t=25, b=10))
    return fig


def correlation_matrix_chart(data: pd.DataFrame) -> go.Figure:
    sample = successful_scenarios(data)
    correlation = sample[SCENARIO_VARIABLES].corr()
    fig = px.imshow(
        correlation,
        text_auto=".2f",
        zmin=-1,
        zmax=1,
        color_continuous_scale="RdBu_r",
        labels={"color": "Correlation"},
    )
    fig.update_layout(margin=dict(l=10, r=10, t=25, b=10))
    return fig

