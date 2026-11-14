import numpy as np
import pandas as pd

from pipeline.earth import Earth
from pipeline.grid import cell_id
from pipeline.validate import control_cell, honest_gaps, top_share


def _earth(eligible: list[bool]) -> Earth:
    lats = np.array([10.0, 10.0, 10.5])
    lons = np.array([20.0, 20.5, 20.0])
    ids = np.array([cell_id(a, b) for a, b in zip(lats, lons, strict=True)])
    return Earth(
        table=pd.DataFrame({"cell_id": ids}),
        lat=lats,
        lon=lons,
        cell_id=ids,
        terrain=[["arid"] if ok else [] for ok in eligible],
        eligible=np.array(eligible),
        access=pd.DataFrame(),
        index_by_cell={cell: index for index, cell in enumerate(ids)},
    )


def test_a_control_uses_the_cell_that_holds_it():
    earth = _earth([True, True, True])
    assert control_cell(earth, 10.1, 20.2, 0.5, earth.eligible, 50.0) == (0, 0.0)


def test_a_masked_control_moves_to_the_nearest_valid_cell_in_reach():
    earth = _earth([False, True, True])
    cell, distance = control_cell(earth, 10.1, 20.2, 0.5, earth.eligible, 50.0)
    assert cell == 1 and 0 < distance <= 50.0


def test_a_control_with_no_valid_cell_in_reach_is_not_comparable():
    earth = _earth([False, True, True])
    assert control_cell(earth, 10.0, 20.0, 0.5, earth.eligible, 10.0) == (None, None)
    assert control_cell(earth, -40.0, 100.0, 0.5, earth.eligible, 50.0) == (None, None)


def test_top_share_counts_the_cells_that_score_higher():
    score = np.array([90.0, 50.0, 70.0, 99.0])
    among = np.array([True, True, True, False])
    assert top_share(score, 0, among) == 0.0
    assert top_share(score, 1, among) == 2 / 3


def test_honest_gaps_list_every_weak_result():
    report = {
        "target": "jezero",
        "positive_controls": [
            {"id": "a", "comparable": False},
            {"id": "b", "comparable": True, "passes": False},
            {"id": "c", "comparable": True, "passes": True},
        ],
        "negative_controls": [{"id": "d", "comparable": True, "in_lower_half": False}],
        "features_without_data": ["composition"],
    }
    kinds = [(gap["kind"], gap["id"]) for gap in honest_gaps([report])]
    assert kinds == [
        ("not_comparable", "a"),
        ("below_target", "b"),
        ("negative_ranks_high", "d"),
        ("feature_without_data", "composition"),
    ]
