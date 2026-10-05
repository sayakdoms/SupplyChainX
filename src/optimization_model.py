"""Pyomo model construction for the warehouse network design problem."""

from __future__ import annotations

import pandas as pd
import pyomo.environ as pyo


REQUIRED_WAREHOUSE_COLUMNS = {"warehouse_id", "capacity", "fixed_cost"}
REQUIRED_DEMAND_COLUMNS = {"demand_zone_id", "demand", "shortage_penalty"}
REQUIRED_COST_COLUMNS = {"warehouse_id", "demand_zone_id", "transport_cost"}


def _require_columns(frame: pd.DataFrame, required: set[str], name: str) -> None:
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{name} is missing required columns: {sorted(missing)}")


def build_model(
    warehouses: pd.DataFrame,
    demand_zones: pd.DataFrame,
    transportation_costs: pd.DataFrame,
) -> pyo.ConcreteModel:
    """Build and return a Pyomo ConcreteModel without invoking a solver."""
    _require_columns(warehouses, REQUIRED_WAREHOUSE_COLUMNS, "warehouses")
    _require_columns(demand_zones, REQUIRED_DEMAND_COLUMNS, "demand_zones")
    _require_columns(transportation_costs, REQUIRED_COST_COLUMNS, "transportation_costs")

    wh = warehouses.set_index("warehouse_id")
    dz = demand_zones.set_index("demand_zone_id")
    cost_series = transportation_costs.set_index(["warehouse_id", "demand_zone_id"])[
        "transport_cost"
    ]
    expected = pd.MultiIndex.from_product([wh.index, dz.index])
    if not expected.isin(cost_series.index).all():
        raise ValueError("Transportation costs must contain every warehouse-demand lane.")

    model = pyo.ConcreteModel(name="SupplyChainX")
    model.W = pyo.Set(initialize=wh.index.tolist(), ordered=True)
    model.D = pyo.Set(initialize=dz.index.tolist(), ordered=True)
    model.capacity = pyo.Param(model.W, initialize=wh["capacity"].to_dict())
    model.fixed_cost = pyo.Param(model.W, initialize=wh["fixed_cost"].to_dict())
    model.demand = pyo.Param(model.D, initialize=dz["demand"].to_dict())
    model.shortage_penalty = pyo.Param(
        model.D, initialize=dz["shortage_penalty"].to_dict()
    )
    model.transport_cost = pyo.Param(model.W, model.D, initialize=cost_series.to_dict())

    model.x = pyo.Var(model.W, model.D, domain=pyo.NonNegativeReals)
    model.y = pyo.Var(model.W, domain=pyo.Binary)
    model.u = pyo.Var(model.D, domain=pyo.NonNegativeReals)

    def capacity_rule(m: pyo.ConcreteModel, i: str) -> pyo.Constraint:
        return sum(m.x[i, j] for j in m.D) <= m.capacity[i] * m.y[i]

    def demand_rule(m: pyo.ConcreteModel, j: str) -> pyo.Constraint:
        return sum(m.x[i, j] for i in m.W) + m.u[j] == m.demand[j]

    model.capacity_constraint = pyo.Constraint(model.W, rule=capacity_rule)
    model.demand_balance = pyo.Constraint(model.D, rule=demand_rule)
    model.objective = pyo.Objective(
        expr=sum(model.transport_cost[i, j] * model.x[i, j] for i in model.W for j in model.D)
        + sum(model.fixed_cost[i] * model.y[i] for i in model.W)
        + sum(model.shortage_penalty[j] * model.u[j] for j in model.D),
        sense=pyo.minimize,
    )
    return model

