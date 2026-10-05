"""Tests for deterministic synthetic data generation."""

import pandas as pd

from src.data_generator import generate_synthetic_data


def test_generated_shapes_and_schema(tmp_path):
    warehouses, demand_zones, costs = generate_synthetic_data(tmp_path)
    assert len(warehouses) == 15
    assert len(demand_zones) == 50
    assert len(costs) == 15 * 50
    assert {"warehouse_id", "capacity", "fixed_cost", "latitude", "longitude"} <= set(warehouses.columns)
    assert {"demand_zone_id", "demand", "shortage_penalty"} <= set(demand_zones.columns)
    assert (costs["transport_cost"] > 0).all()
    assert (tmp_path / "warehouses.csv").exists()


def test_generation_is_reproducible():
    first = generate_synthetic_data(seed=7)
    second = generate_synthetic_data(seed=7)
    for left, right in zip(first, second):
        pd.testing.assert_frame_equal(left, right)

