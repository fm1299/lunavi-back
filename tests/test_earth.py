"""Comparing a Moon or Mars site with the Earth table."""

import numpy as np
import pandas as pd
import pytest

from pipeline.contract import Feature, Target, TargetFeature
from pipeline.earth import Earth, beyond_earth_keys, compare

QUALITY = {"measured": 1.0, "physical": 1.0, "literature": 0.7}


def _feature(key: str, **kwargs) -> Feature:
    defaults = {
        "label": key,
        "unit": "x",
        "type": "numeric",
        "direction": "",
        "lo": 0.0,
        "hi": 10.0,
        "source_quality": 1.0,
        "resolution_factor": 1.0,
        "adapter": "module:function",
        "source_name": "s",
        "source_url": None,
        "license": None,
    }
    return Feature(key=key, **{**defaults, **kwargs})


FEATURES = {
    "slope": _feature("slope"),
    "aridity": _feature("aridity"),
    "composition": _feature(
        "composition",
        type="categorical",
        lo=None,
        hi=None,
        adapter=None,
        similarity_matrix={"basaltic": {"basaltic": 1.0}},
    ),
}


def _earth() -> Earth:
    table = pd.DataFrame(
        {
            "cell_id": ["a", "b", "c"],
            "lat": [0.0, 1.0, 2.0],
            "lon": [0.0, 1.0, 2.0],
            "terrain": [["arid"], ["arid"], []],
            "raw_slope": [5.0, 0.0, np.nan],
            "gap_slope": [0.0, 0.5, 0.0],
            "raw_aridity": [0.0, 10.0, 5.0],
            "gap_aridity": [0.0, 0.0, 0.0],
            "raw_composition": [None, None, None],
            "gap_composition": [1.0, 1.0, 1.0],
        }
    )
    return Earth(
        table=table,
        lat=table["lat"].to_numpy(),
        lon=table["lon"].to_numpy(),
        cell_id=table["cell_id"].to_numpy(),
        terrain=list(table["terrain"]),
        eligible=np.array([True, True, False]),
        index_by_cell={"a": 0, "b": 1, "c": 2},
    )


def _target(**features) -> Target:
    return Target(
        id="t",
        name="t",
        body="mars",
        lat=0,
        lon=0,
        description="",
        default_profile="p",
        features=features,
    )


def _value(value, basis="measured") -> TargetFeature:
    return TargetFeature(value=value, basis=basis, source_url=None)


def test_compare_fills_similarity_for_each_cell():
    result = compare(_earth(), FEATURES, _target(slope=_value(5.0)), QUALITY)
    column = result.keys.index("slope")
    assert result.s[0, column] == pytest.approx(1.0)
    assert result.s[1, column] == pytest.approx(0.5)
    assert np.isnan(result.s[2, column])  # the Earth value is missing


def test_quality_uses_earth_gaps_and_target_basis():
    result = compare(_earth(), FEATURES, _target(slope=_value(5.0, "literature")), QUALITY)
    column = result.keys.index("slope")
    assert result.q[0, column] == pytest.approx(0.7)
    assert result.q[1, column] == pytest.approx(0.35)
    assert np.isnan(result.q[2, column])


def test_a_site_with_no_value_leaves_the_feature_missing():
    blank = TargetFeature(value=None, basis=None, source_url=None)
    result = compare(_earth(), FEATURES, _target(slope=blank), QUALITY)
    assert np.isnan(result.s[:, result.keys.index("slope")]).all()


def test_a_feature_with_no_earth_source_stays_missing():
    result = compare(_earth(), FEATURES, _target(composition=_value("basaltic")), QUALITY)
    assert np.isnan(result.s[:, result.keys.index("composition")]).all()


def test_vetoes_use_the_gate_threshold():
    result = compare(_earth(), FEATURES, _target(aridity=_value(0.0, "physical")), QUALITY)
    assert result.vetoes({"aridity": 0.5}).tolist() == [False, True, False]


def test_beyond_earth_keys_flag_values_outside_the_range():
    target = _target(slope=_value(50.0), aridity=_value(5.0))
    assert beyond_earth_keys(FEATURES, target) == {"slope"}


def test_a_height_from_another_datum_is_never_beyond_earth():
    features = {"elevation": _feature("elevation", comparable_across_bodies=False)}
    assert beyond_earth_keys(features, _target(elevation=_value(-2384.0))) == set()
