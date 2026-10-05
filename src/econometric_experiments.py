"""Generate controlled simulation-response experiments from repeated MILP solves.

This module does not estimate causal effects from observed firms. It perturbs the
synthetic SupplyChainX environment, solves the unchanged Pyomo model, and records
how optimized outcomes respond inside that controlled simulation.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from src.scenario_engine import ScenarioConfig, apply_scenario
from src.solver import solve_network


EXPERIMENT_SEED = 2026
DEFAULT_SCENARIOS = 400
SHORTAGE_TOLERANCE = 1e-5
INPUT_COLUMNS = [
    "scenario_id",
    "demand_multiplier",
    "transport_cost_inflation",
    "shortage_penalty_multiplier",
    "capacity_reduction",
    "disrupted_warehouse_count",
]
OUTPUT_COLUMNS = [
    "total_cost",
    "transport_cost",
    "fixed_cost",
    "shortage_cost",
    "service_level",
    "unmet_demand",
    "active_warehouses",
    "average_utilization",
    "solver_status",
    "shortage_event",
]
REQUIRED_COLUMNS = INPUT_COLUMNS + OUTPUT_COLUMNS


def generate_scenario_design(
    n_scenarios: int = DEFAULT_SCENARIOS, seed: int = EXPERIMENT_SEED
) -> pd.DataFrame:
    """Return a reproducible design spanning the requested experimental ranges."""
    if n_scenarios <= 0:
        raise ValueError("n_scenarios must be positive.")
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "scenario_id": [f"SCN{i:04d}" for i in range(1, n_scenarios + 1)],
            "demand_multiplier": rng.uniform(0.80, 1.40, n_scenarios).round(6),
            "transport_cost_inflation": rng.uniform(-0.10, 0.40, n_scenarios).round(6),
            "shortage_penalty_multiplier": rng.uniform(0.50, 2.00, n_scenarios).round(6),
            "capacity_reduction": rng.uniform(0.0, 0.50, n_scenarios).round(6),
            "disrupted_warehouse_count": rng.integers(0, 4, n_scenarios),
        }
    )


def _load_base_data(project_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return (
        pd.read_csv(project_dir / "data" / "warehouses.csv"),
        pd.read_csv(project_dir / "data" / "demand_zones.csv"),
        pd.read_csv(project_dir / "data" / "transportation_costs.csv"),
    )


def _average_active_utilization(warehouse_results: pd.DataFrame) -> float:
    """Average utilization across active warehouses with positive capacity."""
    eligible = warehouse_results.loc[
        (warehouse_results["active"] > 0.5) & (warehouse_results["capacity"] > 0),
        "utilization",
    ].replace([np.inf, -np.inf], np.nan)
    return float(eligible.mean()) if eligible.notna().any() else 0.0


def run_experiments(
    n_scenarios: int = DEFAULT_SCENARIOS,
    seed: int = EXPERIMENT_SEED,
    output_path: str | Path | None = None,
    design: pd.DataFrame | None = None,
    base_data: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> pd.DataFrame:
    """Solve every designed scenario and optionally persist the response dataset.

    ``capacity_reduction`` is applied proportionally to all surviving warehouses.
    A reproducibly selected set of ``disrupted_warehouse_count`` facilities is then
    fully shut down. Failed solves are retained with missing outcomes and an error
    status so downstream regression code can exclude them explicitly.
    """
    project_dir = Path(__file__).resolve().parents[1]
    warehouses, demand_zones, costs = base_data or _load_base_data(project_dir)
    scenario_design = (
        generate_scenario_design(n_scenarios, seed) if design is None else design.copy()
    )
    missing = set(INPUT_COLUMNS).difference(scenario_design.columns)
    if missing:
        raise ValueError(f"Scenario design is missing columns: {sorted(missing)}")

    selection_rng = np.random.default_rng(seed + 10_000)
    warehouse_ids = warehouses["warehouse_id"].to_numpy()
    rows: list[dict[str, object]] = []
    total = len(scenario_design)

    for position, scenario in enumerate(scenario_design.itertuples(index=False), start=1):
        inputs = {column: getattr(scenario, column) for column in INPUT_COLUMNS}
        config = ScenarioConfig(
            name=str(scenario.scenario_id),
            demand_multiplier=float(scenario.demand_multiplier),
            transport_cost_inflation=float(scenario.transport_cost_inflation),
            shortage_penalty_multiplier=float(scenario.shortage_penalty_multiplier),
        )
        scenario_wh, scenario_dz, scenario_costs = apply_scenario(
            warehouses, demand_zones, costs, config
        )
        scenario_wh["capacity"] = scenario_wh["capacity"] * (
            1.0 - float(scenario.capacity_reduction)
        )
        disrupted_count = int(scenario.disrupted_warehouse_count)
        if disrupted_count:
            disrupted = selection_rng.choice(
                warehouse_ids, size=disrupted_count, replace=False
            )
            scenario_wh.loc[
                scenario_wh["warehouse_id"].isin(disrupted), "capacity"
            ] = 0.0

        try:
            result = solve_network(scenario_wh, scenario_dz, scenario_costs)
            metrics = result.analytics
            unmet = float(metrics["total_unmet_demand"])
            outputs: dict[str, object] = {
                "total_cost": float(metrics["total_cost"]),
                "transport_cost": float(metrics["transport_cost"]),
                "fixed_cost": float(metrics["fixed_cost"]),
                "shortage_cost": float(metrics["shortage_cost"]),
                "service_level": float(metrics["service_level"]),
                "unmet_demand": unmet,
                "active_warehouses": int(metrics["active_warehouses"]),
                "average_utilization": _average_active_utilization(result.warehouses),
                "solver_status": f"{result.status}/{result.termination_condition}",
                "shortage_event": int(unmet > SHORTAGE_TOLERANCE),
            }
        except Exception as exc:  # Preserve failures for transparent filtering.
            outputs = {
                column: np.nan
                for column in OUTPUT_COLUMNS
                if column not in {"solver_status", "shortage_event"}
            }
            outputs["solver_status"] = f"failed/{type(exc).__name__}: {exc}"
            outputs["shortage_event"] = np.nan
        rows.append({**inputs, **outputs})
        if progress_callback is not None:
            progress_callback(position, total)

    results = pd.DataFrame(rows, columns=REQUIRED_COLUMNS)
    if output_path is not None:
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(target, index=False)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenarios", type=int, default=DEFAULT_SCENARIOS)
    parser.add_argument("--seed", type=int, default=EXPERIMENT_SEED)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "results"
        / "econometric_scenarios.csv",
    )
    args = parser.parse_args()

    def report_progress(done: int, total: int) -> None:
        if done == 1 or done % 25 == 0 or done == total:
            print(f"Solved {done}/{total} scenarios", flush=True)

    results = run_experiments(
        n_scenarios=args.scenarios,
        seed=args.seed,
        output_path=args.output,
        progress_callback=report_progress,
    )
    successful = results["solver_status"].str.endswith(("/optimal", "/feasible"))
    print(f"Saved {len(results)} scenarios to {args.output}")
    print(f"Successful solves: {int(successful.sum())}/{len(results)}")
    print(f"Shortage events: {int(results['shortage_event'].fillna(0).sum())}")


if __name__ == "__main__":
    main()

