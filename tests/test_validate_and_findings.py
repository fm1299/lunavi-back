"""Validation and candidate selection on a small synthetic table with known answers."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pipeline.contract import load_config
from pipeline.findings import target_candidates
from pipeline.grid import cell_id
from pipeline.validate import build_report


@pytest.fixture(scope="module")
def config():
    return load_config(Path("data/config"))


def _table(config, n_rows=6, n_cols=6, lat0=-24.0, lon0=-70.0, novelty="new") -> pd.DataFrame:
    """A dry grid with similarity falling steadily away from the first corner."""
    rows = []
    for r in range(n_rows):
        for c in range(n_cols):
            lat = lat0 + r * 0.25
            lon = lon0 + c * 0.25
            rows.append((lat, lon, 1.0 - (r + c) / (2 * (n_rows + n_cols))))
    data = {
        "cell_id": [cell_id(a, b) for a, b, _ in rows],
        "lat": [a for a, _, _ in rows],
        "lon": [b for _, b, _ in rows],
        "land_fraction": 1.0,
        "terrain": [["arid"] for _ in rows],
        "tags": [[] for _ in rows],
        "novelty": novelty,
        "nearest_known": "somewhere",
        "nearest_known_km": 600.0,
    }
    for key, feature in config.features.items():
        data[f"s_{key}"] = [value for _, _, value in rows]
        data[f"q_{key}"] = 0.8
        data[f"raw_{key}"] = None if feature.type == "categorical" else 1.0
    return pd.DataFrame(data)


def test_control_in_the_best_cell_ranks_at_the_top(config):
    # Atacama is at -24.1, -70.0, which falls in the first (best) cell of this grid.
    report = build_report(config, {"jezero": _table(config)})["targets"][0]
    atacama = next(r for r in report["positive_controls"] if r["id"] == "atacama")
    assert atacama["status"] == "ranked"
    assert atacama["top_percent"] == 0.0
    assert atacama["meets_goal"] is True


def test_controls_outside_the_grid_are_reported_not_hidden(config):
    report = build_report(config, {"jezero": _table(config)})
    positives = report["targets"][0]["positive_controls"]
    assert any(r["id"] == "death_valley" and r["status"] == "outside_region" for r in positives)
    assert any("Death Valley" in gap for gap in report["honest_gaps"])


def test_masked_control_with_no_eligible_cell_nearby_is_not_comparable(config):
    table = _table(config)
    table["terrain"] = [[] if i < 4 else ["arid"] for i in range(len(table))]
    table.loc[:, "lat"] = table["lat"]
    # Only keep eligible cells far away from Atacama.
    far = table["lon"] > -68.7
    table["terrain"] = [["arid"] if keep else [] for keep in far]
    report = build_report(config, {"jezero": table})["targets"][0]
    atacama = next(r for r in report["positive_controls"] if r["id"] == "atacama")
    assert atacama["status"] == "not_comparable"


def test_catalog_shortfall_is_an_honest_gap(config):
    gaps = build_report(config, {"jezero": _table(config)})["honest_gaps"]
    assert any("Catalog has" in gap for gap in gaps)


def test_funnel_counts_shrink_at_each_stage(config):
    table = _table(config)
    table.loc[:3, "s_vegetation"] = 0.1  # vetoed by the gate
    table.loc[4:7, "terrain"] = pd.Series([[] for _ in range(4)], index=range(4, 8))
    funnel = build_report(config, {"jezero": table})["targets"][0]["funnel"]
    assert funnel["land_cells"] == 36
    assert funnel["after_masks"] == 32
    assert funnel["after_gates"] == 28
    assert funnel["findings"] == 0


def test_sensitivity_and_leave_one_out_report_overlaps(config):
    report = build_report(config, {"jezero": _table(config)})["targets"][0]
    assert 0.0 <= report["weight_sensitivity"]["mean_top50_overlap"] <= 1.0
    assert {r["feature"] for r in report["leave_one_feature_out"]} == {
        "slope", "roughness", "aridity", "temp_annual_range", "temp_diurnal_range", "composition",
    }  # fmt: skip


def test_candidates_keep_only_new_cells_and_merge_neighbours(config):
    table = _table(config, n_rows=20, n_cols=20)
    result = target_candidates(config, config.targets["jezero"], table)
    assert result["pending"]
    assert all(c["status"] == "awaiting_literature_check" for c in result["pending"])
    assert all(c["name"] is None for c in result["pending"])
    assert len(result["pending"]) == 1  # the top 5 percent is one connected corner
    assert len(result["pending"][0]["cell_ids"]) > 1


def test_no_new_cells_means_no_candidates(config):
    table = _table(config, novelty="known")
    assert target_candidates(config, config.targets["jezero"], table)["pending"] == []


def test_low_confidence_cells_are_not_candidates(config):
    table = _table(config)
    for key in config.features:
        table[f"q_{key}"] = np.full(len(table), 0.2)
    assert target_candidates(config, config.targets["jezero"], table)["pending"] == []
