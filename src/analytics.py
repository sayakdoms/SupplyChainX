"""Post-solve analytics for SupplyChainX."""

from __future__ import annotations

from typing import Any

import pandas as pd
import pyomo.environ as pyo


def calculate_analytics(
    warehouses: pd.DataFrame,
    demand_zones: pd.DataFrame,
    shipments: pd.DataFrame,
    model: pyo.ConcreteModel,
    objective_value: float | None,
) -> dict[str, Any]:
    """Calculate financial, service, and capacity KPIs from a solved model."""
    transport_cost = sum(
        pyo.value(model.transport_cost[i, j]) * (pyo.value(model.x[i, j]) or 0.0)
        for i in model.W
        for j in model.D
    )
    fixed_cost = sum(
        pyo.value(model.fixed_cost[i]) * (pyo.value(model.y[i]) or 0.0) for i in model.W
    )
    shortage_cost = sum(
        pyo.value(model.shortage_penalty[j]) * (pyo.value(model.u[j]) or 0.0)
        for j in model.D
    )
    if abs(shortage_cost) < 1e-6:
        shortage_cost = 0.0
    total_demand = float(demand_zones["demand"].sum())
    total_unmet = float(demand_zones["unmet_demand"].sum())
    active = warehouses.loc[warehouses["active"] > 0.5]
    highest = (
        warehouses.loc[warehouses["utilization"].idxmax(), "warehouse_id"]
        if not warehouses.empty
        else None
    )
    return {
        "total_cost": float(objective_value or 0.0),
        "transport_cost": float(transport_cost),
        "fixed_cost": float(fixed_cost),
        "shortage_cost": float(shortage_cost),
        "service_level": 1.0 - total_unmet / total_demand if total_demand else 1.0,
        "total_unmet_demand": total_unmet,
        "warehouse_utilization": warehouses[
            ["warehouse_id", "capacity", "shipped_quantity", "utilization"]
        ].copy(),
        "active_warehouses": int(len(active)),
        "highest_utilization_warehouse": highest,
        "shipment_lanes_used": int(len(shipments)),
    }


def scenario_summary(name: str, result: Any) -> dict[str, Any]:
    """Flatten a solved scenario into one comparison-table row."""
    metrics = result.analytics
    return {
        "scenario": name,
        "total_cost": metrics["total_cost"],
        "service_level": metrics["service_level"],
        "unmet_demand": metrics["total_unmet_demand"],
        "active_warehouses": metrics["active_warehouses"],
        "solver": result.solver_name,
    }

