"""Tests for model construction and optimization feasibility."""

import pyomo.environ as pyo
import pytest

from src.data_generator import generate_synthetic_data
from src.optimization_model import build_model
from src.solver import solve_network


@pytest.fixture(scope="module")
def network_data():
    return generate_synthetic_data(n_warehouses=4, n_demand_zones=8, seed=123)


def test_model_construction(network_data):
    warehouses, demand_zones, costs = network_data
    model = build_model(warehouses, demand_zones, costs)
    assert len(model.W) == 4
    assert len(model.D) == 8
    assert len(model.x) == 32
    assert model.objective.sense == pyo.minimize


def test_solver_output_and_feasibility(network_data):
    warehouses, demand_zones, costs = network_data
    try:
        result = solve_network(warehouses, demand_zones, costs)
    except RuntimeError as exc:
        pytest.skip(str(exc))
    assert result.termination_condition == "optimal"
    assert result.objective_value is not None and result.objective_value > 0
    assert result.solver_name in {"gurobi", "scip", "highs"}

    shipped_by_wh = result.shipments.groupby("warehouse_id")["quantity"].sum()
    for row in result.warehouses.itertuples():
        assert shipped_by_wh.get(row.warehouse_id, 0.0) <= row.capacity * row.active + 1e-5

    served = result.shipments.groupby("demand_zone_id")["quantity"].sum()
    for row in result.demand_zones.itertuples():
        assert served.get(row.demand_zone_id, 0.0) + row.unmet_demand == pytest.approx(row.demand, abs=1e-5)

