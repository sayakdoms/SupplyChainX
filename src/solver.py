"""Solver selection, execution, and solution extraction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import pyomo.environ as pyo
from pyomo.opt import SolverFactory

from src.analytics import calculate_analytics
from src.optimization_model import build_model


@dataclass
class OptimizationResult:
    """Normalized output returned by the SupplyChainX solver layer."""

    solver_name: str
    status: str
    termination_condition: str
    objective_value: float | None
    shipments: pd.DataFrame
    warehouses: pd.DataFrame
    demand_zones: pd.DataFrame
    analytics: dict[str, Any]


def solve_model(
    model: pyo.ConcreteModel,
    warehouses: pd.DataFrame,
    demand_zones: pd.DataFrame,
    tee: bool = False,
) -> OptimizationResult:
    """Solve a constructed model using Gurobi, SCIP, then HiGHS as a portable fallback."""
    errors: list[str] = []
    solver_name = ""
    raw = None
    for candidate in ("gurobi", "scip", "appsi_highs"):
        try:
            solver = SolverFactory(candidate)
            if solver is None or not solver.available(exception_flag=False):
                errors.append(f"{candidate}: unavailable")
                continue
            raw = solver.solve(model, tee=tee)
            solver_name = candidate
            break
        except Exception as exc:  # pragma: no cover - depends on local solver setup
            errors.append(f"{candidate}: {exc}")
    if raw is None:
        raise RuntimeError("No supported solver could solve the model. " + "; ".join(errors))
    status = str(raw.solver.status)
    termination = str(raw.solver.termination_condition)
    acceptable = {"optimal", "feasible", "locallyOptimal"}
    objective = pyo.value(model.objective) if termination in acceptable else None

    shipment_rows = [
        {
            "warehouse_id": i,
            "demand_zone_id": j,
            "quantity": float(pyo.value(model.x[i, j]) or 0.0),
        }
        for i in model.W
        for j in model.D
        if float(pyo.value(model.x[i, j]) or 0.0) > 1e-6
    ]
    shipment_columns = ["warehouse_id", "demand_zone_id", "quantity"]
    shipments = pd.DataFrame(shipment_rows, columns=shipment_columns)
    warehouse_solution = warehouses.copy()
    warehouse_solution["active"] = [round(float(pyo.value(model.y[i]) or 0.0)) for i in model.W]
    warehouse_solution["shipped_quantity"] = [
        sum(float(pyo.value(model.x[i, j]) or 0.0) for j in model.D) for i in model.W
    ]
    warehouse_solution["utilization"] = (
        warehouse_solution["shipped_quantity"] / warehouse_solution["capacity"]
    )
    demand_solution = demand_zones.copy()
    demand_solution["unmet_demand"] = [
        0.0 if abs(value := float(pyo.value(model.u[j]) or 0.0)) < 1e-6 else value
        for j in model.D
    ]
    demand_solution["served_demand"] = demand_solution["demand"] - demand_solution["unmet_demand"]
    analytics = calculate_analytics(
        warehouse_solution, demand_solution, shipments, model, objective
    )
    return OptimizationResult(
        solver_name=solver_name,
        status=status,
        termination_condition=termination,
        objective_value=objective,
        shipments=shipments,
        warehouses=warehouse_solution,
        demand_zones=demand_solution,
        analytics=analytics,
    )


def solve_network(
    warehouses: pd.DataFrame,
    demand_zones: pd.DataFrame,
    transportation_costs: pd.DataFrame,
    tee: bool = False,
) -> OptimizationResult:
    """Build and solve a network instance."""
    model = build_model(warehouses, demand_zones, transportation_costs)
    return solve_model(model, warehouses, demand_zones, tee=tee)


def main() -> None:
    """Solve the generated base dataset and print a compact summary."""
    project_dir = Path(__file__).resolve().parents[1]
    warehouses = pd.read_csv(project_dir / "data" / "warehouses.csv")
    demand_zones = pd.read_csv(project_dir / "data" / "demand_zones.csv")
    costs = pd.read_csv(project_dir / "data" / "transportation_costs.csv")
    result = solve_network(warehouses, demand_zones, costs)
    print(f"Solver: {result.solver_name}")
    print(f"Status: {result.status} / {result.termination_condition}")
    print(f"Objective value: {result.objective_value:,.2f}")
    for key in ("service_level", "total_unmet_demand", "active_warehouses", "highest_utilization_warehouse"):
        print(f"{key}: {result.analytics[key]}")


if __name__ == "__main__":
    main()

