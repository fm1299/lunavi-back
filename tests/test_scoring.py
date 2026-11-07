import numpy as np
import pytest

from pipeline import scoring

MATRIX = {"basaltic": {"basaltic": 1.0, "felsic": 0.4}, "felsic": {"basaltic": 0.4, "felsic": 1.0}}


def test_identical_values_have_full_similarity():
    assert scoring.numeric_similarity(5.0, 5.0, 0, 10) == pytest.approx(1.0)


def test_similarity_uses_the_normalized_distance():
    assert scoring.numeric_similarity(2.0, 7.0, 0, 10) == pytest.approx(0.5)


def test_target_beyond_earth_is_clamped_to_the_edge():
    # A -138 C target clamps to the -55 C edge, so the coldest Earth cell looks like a full match.
    assert scoring.numeric_similarity(-55.0, -138.0, -55, 32) == pytest.approx(1.0)
    assert scoring.beyond_earth(-138.0, -55, 32) is True
    assert scoring.beyond_earth(-20.0, -55, 32) is False


def test_beyond_earth_ignores_missing_and_categorical_values():
    assert scoring.beyond_earth(None, 0, 1) is False
    assert scoring.beyond_earth("basaltic", None, None) is False


def test_categorical_similarity_reads_the_matrix_and_keeps_missing_as_nan():
    result = scoring.categorical_similarity(["felsic", None], "basaltic", MATRIX)
    assert result[0] == pytest.approx(0.4)
    assert np.isnan(result[1])


def test_score_is_a_weighted_mean_from_0_to_100():
    s = np.array([[1.0, 0.5]])
    assert scoring.combined_score(s, np.array([3.0, 1.0]))[0] == pytest.approx(87.5)


def test_missing_feature_leaves_both_sums():
    s = np.array([[1.0, np.nan]])
    assert scoring.combined_score(s, np.array([1.0, 3.0]))[0] == pytest.approx(100.0)


def test_cell_with_no_weighted_features_scores_zero():
    s = np.array([[np.nan, 0.5]])
    assert scoring.combined_score(s, np.array([1.0, 0.0]))[0] == 0.0


def test_veto_sets_score_to_zero():
    s = np.array([[1.0], [1.0]])
    vetoed = np.array([True, False])
    assert scoring.combined_score(s, np.array([1.0]), vetoed).tolist() == [0.0, 100.0]


def test_gate_vetoes_below_threshold_but_not_on_missing():
    gate_s = np.array([[0.4], [0.5], [np.nan]])
    assert scoring.gate_vetoes(gate_s, np.array([0.5])).tolist() == [True, False, False]


def test_no_gates_vetoes_nothing():
    assert scoring.gate_vetoes(np.empty((2, 0)), np.array([])).tolist() == [False, False]


def test_geometric_mean_is_punished_by_one_bad_feature():
    s = np.array([[1.0, 0.1]])
    weights = np.array([1.0, 1.0])
    arithmetic = scoring.combined_score(s, weights)[0]
    geometric = scoring.combined_score(s, weights, method="geometric")[0]
    assert geometric < arithmetic


def test_geometric_floor_stops_one_zero_erasing_a_cell():
    s = np.array([[1.0, 0.0]])
    score = scoring.combined_score(s, np.array([1.0, 1.0]), method="geometric", floor=0.01)[0]
    assert score == pytest.approx(10.0)


def test_unknown_method_is_refused():
    with pytest.raises(ValueError):
        scoring.combined_score(np.array([[1.0]]), np.array([1.0]), method="median")


def test_quality_multiplies_all_factors():
    assert scoring.quality(0.9, 0.5, 0.2, 0.7) == pytest.approx(0.9 * 0.5 * 0.8 * 0.7)


def test_confidence_with_all_features_is_the_weighted_quality():
    s = np.array([[0.5, 0.5]])
    q = np.array([[0.8, 0.4]])
    assert scoring.confidence(q, s, np.array([3.0, 1.0]))[0] == pytest.approx(0.7)


def test_confidence_drops_with_the_share_of_missing_required_features():
    s = np.array([[0.5, np.nan]])
    q = np.array([[0.8, np.nan]])
    assert scoring.confidence(q, s, np.array([1.0, 1.0]))[0] == pytest.approx(0.4)


def test_literature_target_cannot_reach_the_high_band():
    # Even perfect data sits at 0.7 once target quality for literature applies.
    q = scoring.quality(1.0, 1.0, 0.0, 0.7)
    value = scoring.confidence(np.array([[q]]), np.array([[1.0]]), np.array([1.0]))[0]
    assert scoring.confidence_band(value) == "medium"


def test_bands():
    assert scoring.confidence_band(0.75) == "high"
    assert scoring.confidence_band(0.5) == "medium"
    assert scoring.confidence_band(0.49) == "low"


def test_closest_and_weakest_never_share_a_feature():
    similarities = {"a": 0.99, "b": 0.98, "c": 0.97, "d": 0.96}
    closest, weakest = scoring.closest_and_weakest(similarities, set())
    assert closest == ["a", "b", "c"]
    assert weakest == ["d"]


def test_beyond_earth_feature_is_never_a_closest_match():
    closest, weakest = scoring.closest_and_weakest({"a": 1.0, "b": 0.5}, {"a"})
    assert closest == ["b"]
    assert weakest == ["a"]


def test_weight_share_counts_only_active_weight():
    share = scoring.weight_share({"a": 3, "b": 1, "c": 0}, {"a", "c"})
    assert share == pytest.approx(0.75)
