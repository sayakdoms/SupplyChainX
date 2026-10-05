"""Tests for the simulation-response experiment and econometric layer."""

from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.data_generator import generate_synthetic_data
from src.econometric_experiments import (
    REQUIRED_COLUMNS,
    generate_scenario_design,
    run_experiments,
)
from src.econometrics import (
    fit_ols_response,
    fit_shortage_logit,
    predict_optimized_cost,
)


def _synthetic_response_data(rows: int = 80) -> pd.DataFrame:
    rng = np.random.default_rng(55)
    demand = rng.uniform(0.8, 1.4, rows)
    inflation = rng.uniform(-0.1, 0.4, rows)
    penalty = rng.uniform(0.5, 2.0, rows)
    capacity = rng.uniform(0.0, 0.5, rows)
    disrupted = rng.integers(0, 4, rows)
    log_cost = (
        15.0
        + 0.75 * np.log(demand)
        + 0.35 * inflation
        + 0.8 * capacity
        + 0.08 * penalty
        + 0.05 * disrupted
        + 0.4 * demand * capacity
        + rng.normal(0, 0.015, rows)
    )
    return pd.DataFrame(
        {
            "scenario_id": [f"T{i:03d}" for i in range(rows)],
            "demand_multiplier": demand,
            "transport_cost_inflation": inflation,
            "shortage_penalty_multiplier": penalty,
            "capacity_reduction": capacity,
            "disrupted_warehouse_count": disrupted,
            "total_cost": np.exp(log_cost),
            "shortage_event": (demand + capacity + 0.08 * disrupted > 1.45).astype(int),
            "solver_status": "ok/optimal",
        }
    )


def test_scenario_design_is_reproducible():
    first = generate_scenario_design(12, seed=99)
    second = generate_scenario_design(12, seed=99)
    pd.testing.assert_frame_equal(first, second)


def test_experiment_output_has_required_columns(monkeypatch, tmp_path):
    design = generate_scenario_design(3, seed=2)
    base_data = generate_synthetic_data(n_warehouses=4, n_demand_zones=6, seed=2)

    def fake_solve(*_args, **_kwargs):
        return SimpleNamespace(
            status="ok",
            termination_condition="optimal",
            analytics={
                "total_cost": 100.0,
                "transport_cost": 60.0,
                "fixed_cost": 40.0,
                "shortage_cost": 0.0,
                "service_level": 1.0,
                "total_unmet_demand": 0.0,
                "active_warehouses": 2,
            },
            warehouses=pd.DataFrame(
                {"capacity": [10.0, 20.0], "active": [1, 1], "utilization": [0.5, 0.75]}
            ),
        )

    monkeypatch.setattr("src.econometric_experiments.solve_network", fake_solve)
    output = run_experiments(
        design=design,
        base_data=base_data,
        output_path=tmp_path / "scenarios.csv",
    )
    assert list(output.columns) == REQUIRED_COLUMNS
    assert len(output) == 3
    assert (tmp_path / "scenarios.csv").exists()


def test_ols_runs_excludes_failed_rows_and_predicts_finite_value():
    data = _synthetic_response_data()
    failed = data.iloc[[0]].copy()
    failed["scenario_id"] = "FAILED"
    failed["solver_status"] = "failed/RuntimeError"
    failed["total_cost"] = 1e30
    combined = pd.concat([data, failed], ignore_index=True)
    model = fit_ols_response(combined)
    assert model.observations == len(data)
    assert 0 <= model.r_squared <= 1
    prediction = predict_optimized_cost(
        model,
        demand_multiplier=1.1,
        transport_cost_inflation=0.1,
        capacity_reduction=0.2,
        shortage_penalty_multiplier=1.2,
        disrupted_warehouse_count=1,
    )
    assert np.isfinite(prediction) and prediction > 0


def test_logistic_handles_no_class_variation():
    data = _synthetic_response_data()
    data["shortage_event"] = 0
    result = fit_shortage_logit(data)
    assert not result.available
    assert result.warning and "no class variation" in result.warning

