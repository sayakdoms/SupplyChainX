"""Scenario definitions and deterministic data transformations."""

from __future__ import annotations

from dataclasses import dataclass, replace

import pandas as pd


@dataclass(frozen=True)
class ScenarioConfig:
    """Configuration for one network-planning scenario."""

    name: str = "Base case"
    demand_multiplier: float = 1.0
    transport_cost_inflation: float = 0.0
    capacity_reduction_warehouse: str | None = None
    capacity_reduction_fraction: float = 0.0
    shutdown_warehouse: str | None = None
    shortage_penalty_multiplier: float = 1.0


def apply_scenario(
    warehouses: pd.DataFrame,
    demand_zones: pd.DataFrame,
    transportation_costs: pd.DataFrame,
    config: ScenarioConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return scenario-adjusted copies of the three input datasets."""
    if config.demand_multiplier < 0 or config.shortage_penalty_multiplier < 0:
        raise ValueError("Demand and shortage penalty multipliers must be non-negative.")
    if not 0 <= config.capacity_reduction_fraction <= 1:
        raise ValueError("Capacity reduction fraction must be between 0 and 1.")
    if config.transport_cost_inflation < -1:
        raise ValueError("Transport cost inflation cannot be below -100%.")

    wh = warehouses.copy()
    dz = demand_zones.copy()
    costs = transportation_costs.copy()
    dz["demand"] = dz["demand"] * config.demand_multiplier
    dz["shortage_penalty"] = (
        dz["shortage_penalty"] * config.shortage_penalty_multiplier
    )
    costs["transport_cost"] = costs["transport_cost"] * (
        1 + config.transport_cost_inflation
    )

    if config.capacity_reduction_warehouse:
        mask = wh["warehouse_id"] == config.capacity_reduction_warehouse
        if not mask.any():
            raise ValueError("Capacity reduction warehouse was not found.")
        wh.loc[mask, "capacity"] *= 1 - config.capacity_reduction_fraction
    if config.shutdown_warehouse:
        mask = wh["warehouse_id"] == config.shutdown_warehouse
        if not mask.any():
            raise ValueError("Shutdown warehouse was not found.")
        wh.loc[mask, "capacity"] = 0
    return wh, dz, costs


def base_case() -> ScenarioConfig:
    """Return the unmodified base-case configuration."""
    return ScenarioConfig()


def named_scenarios() -> dict[str, ScenarioConfig]:
    """Return a concise set of useful example scenarios."""
    base = base_case()
    return {
        "Base case": base,
        "Demand surge (+20%)": replace(base, name="Demand surge (+20%)", demand_multiplier=1.2),
        "Fuel inflation (+15%)": replace(
            base, name="Fuel inflation (+15%)", transport_cost_inflation=0.15
        ),
        "Service priority": replace(
            base, name="Service priority", shortage_penalty_multiplier=1.5
        ),
    }

