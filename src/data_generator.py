"""Generate reproducible synthetic data for the SupplyChainX network."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


SEED = 42
N_WAREHOUSES = 15
N_DEMAND_ZONES = 50
INDIA_BOUNDS = {"latitude": (8.5, 33.5), "longitude": (72.0, 88.5)}


def haversine_km(
    lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray
) -> np.ndarray:
    """Return great-circle distance in kilometres for broadcastable arrays."""
    radius_km = 6_371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)
    a = (
        np.sin(delta_phi / 2.0) ** 2
        + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2.0) ** 2
    )
    return 2.0 * radius_km * np.arcsin(np.sqrt(a))


def generate_synthetic_data(
    output_dir: str | Path | None = None,
    seed: int = SEED,
    n_warehouses: int = N_WAREHOUSES,
    n_demand_zones: int = N_DEMAND_ZONES,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Generate warehouses, demand zones, and long-form transportation costs."""
    rng = np.random.default_rng(seed)
    warehouse_ids = [f"WH{i:02d}" for i in range(1, n_warehouses + 1)]
    demand_ids = [f"DZ{i:03d}" for i in range(1, n_demand_zones + 1)]

    warehouses = pd.DataFrame(
        {
            "warehouse_id": warehouse_ids,
            "warehouse_name": [f"Regional Distribution Centre {i}" for i in range(1, n_warehouses + 1)],
            "capacity": rng.integers(1_900, 3_800, size=n_warehouses),
            "fixed_cost": rng.integers(85_000, 190_000, size=n_warehouses),
            "latitude": rng.uniform(*INDIA_BOUNDS["latitude"], size=n_warehouses).round(5),
            "longitude": rng.uniform(*INDIA_BOUNDS["longitude"], size=n_warehouses).round(5),
        }
    )
    demand_zones = pd.DataFrame(
        {
            "demand_zone_id": demand_ids,
            "demand_zone_name": [f"Customer Market {i}" for i in range(1, n_demand_zones + 1)],
            "demand": rng.integers(350, 850, size=n_demand_zones),
            "shortage_penalty": rng.integers(650, 950, size=n_demand_zones),
            "latitude": rng.uniform(*INDIA_BOUNDS["latitude"], size=n_demand_zones).round(5),
            "longitude": rng.uniform(*INDIA_BOUNDS["longitude"], size=n_demand_zones).round(5),
        }
    )

    wh_lat = warehouses["latitude"].to_numpy()[:, None]
    wh_lon = warehouses["longitude"].to_numpy()[:, None]
    dz_lat = demand_zones["latitude"].to_numpy()[None, :]
    dz_lon = demand_zones["longitude"].to_numpy()[None, :]
    distance = haversine_km(wh_lat, wh_lon, dz_lat, dz_lon)
    operating_variation = rng.uniform(0.88, 1.15, size=distance.shape)
    # Per-unit freight cost: handling charge plus a distance-based line-haul rate.
    costs = (18.0 + 0.22 * distance) * operating_variation

    transportation_costs = pd.DataFrame(
        {
            "warehouse_id": np.repeat(warehouse_ids, n_demand_zones),
            "demand_zone_id": np.tile(demand_ids, n_warehouses),
            "distance_km": distance.ravel().round(2),
            "transport_cost": costs.ravel().round(2),
        }
    )

    if output_dir is not None:
        target = Path(output_dir)
        target.mkdir(parents=True, exist_ok=True)
        warehouses.to_csv(target / "warehouses.csv", index=False)
        demand_zones.to_csv(target / "demand_zones.csv", index=False)
        transportation_costs.to_csv(target / "transportation_costs.csv", index=False)

    return warehouses, demand_zones, transportation_costs


def main() -> None:
    """Generate the default dataset in the project data directory."""
    data_dir = Path(__file__).resolve().parents[1] / "data"
    warehouses, demand_zones, costs = generate_synthetic_data(data_dir)
    print(
        f"Generated {len(warehouses)} warehouses, {len(demand_zones)} demand zones, "
        f"and {len(costs)} transport lanes in {data_dir}."
    )


if __name__ == "__main__":
    main()

