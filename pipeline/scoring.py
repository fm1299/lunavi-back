"""Scoring (spec section 5). Pure functions with no I/O, used by the pipeline and the API.

Matrices are n_cells x n_features. NaN means the feature is missing for that cell.
"""

import numpy as np


def normalize(x, lo: float, hi: float) -> np.ndarray:
    return np.clip((np.asarray(x, dtype=np.float64) - lo) / (hi - lo), 0.0, 1.0)


def numeric_similarity(cell_values, target_value: float, lo: float, hi: float) -> np.ndarray:
    """s = 1 - |z_cell - z_target|. A target outside [lo, hi] is clamped: as close as Earth gets."""
    return 1.0 - np.abs(normalize(cell_values, lo, hi) - normalize(target_value, lo, hi))


def categorical_similarity(cell_classes, target_class: str, matrix: dict) -> np.ndarray:
    row = matrix[target_class]
    return np.array([row.get(c, np.nan) for c in cell_classes], dtype=np.float64)


def beyond_earth(target_value, lo: float | None, hi: float | None) -> bool:
    """True when a numeric target value lies outside the Earth range."""
    if target_value is None or isinstance(target_value, str) or lo is None or hi is None:
        return False
    return bool(target_value < lo or target_value > hi)


def quality(
    source_quality: float, resolution_factor: float, gap_filled, target_quality: float
) -> np.ndarray:
    gap = np.asarray(gap_filled, dtype=np.float64)
    return source_quality * resolution_factor * (1.0 - gap) * target_quality


def access_score(road_km, airport_km, road_far_km: float, airport_far_km: float) -> np.ndarray:
    """From 0 to 1: half from the distance to a road, half from the distance to an airport.

    Each half falls in a straight line from 1 at the road or airport to 0 at its far distance.
    """
    road = np.clip(1.0 - np.asarray(road_km, dtype=np.float64) / road_far_km, 0.0, 1.0)
    airport = np.clip(1.0 - np.asarray(airport_km, dtype=np.float64) / airport_far_km, 0.0, 1.0)
    return 0.5 * road + 0.5 * airport


def greedy_cover(covers: np.ndarray, rank: np.ndarray, max_cells: int) -> list[int]:
    """Greedy set cover (spec FR-P13): the fewest rows of `covers` that together cover the columns.

    covers is n_cells x n_conditions, True where a cell covers a condition. Each step takes the
    cell that covers the most conditions still open, and the higher `rank` on a tie. It stops
    when nothing more can be covered. Greedy is not always the true minimum, but it is close.
    """
    covers = np.asarray(covers, dtype=bool)
    open_columns = np.ones(covers.shape[1], dtype=bool)
    picks: list[int] = []
    while open_columns.any() and len(picks) < max_cells and covers.shape[0] > 0:
        gains = covers[:, open_columns].sum(axis=1)
        best = int(np.lexsort((rank, gains))[-1])
        if gains[best] == 0:
            break
        picks.append(best)
        open_columns &= ~covers[best]
    return picks


def combined_score(
    s: np.ndarray,
    weights: np.ndarray,
    vetoed: np.ndarray | None = None,
    method: str = "arithmetic",
    floor: float = 0.01,
) -> np.ndarray:
    """Score from 0 to 100. Missing features leave both sums. Vetoed cells score 0."""
    active = ~np.isnan(s) * weights
    total = active.sum(axis=1)
    safe_total = np.where(total > 0, total, 1.0)
    if method == "arithmetic":
        value = (np.nan_to_num(s) * active).sum(axis=1) / safe_total
    elif method == "geometric":
        logs = np.log(np.maximum(np.nan_to_num(s, nan=1.0), floor))
        value = np.exp((logs * active).sum(axis=1) / safe_total)
    else:
        raise ValueError(f"unknown method {method}")
    score = 100.0 * value
    score[total <= 0] = 0.0
    if vetoed is not None:
        score[vetoed] = 0.0
    return score


def gate_vetoes(gate_s: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """A cell is vetoed when a gate feature falls below its threshold. Missing never vetoes."""
    if gate_s.shape[1] == 0:
        return np.zeros(gate_s.shape[0], dtype=bool)
    with np.errstate(invalid="ignore"):
        return (gate_s < thresholds).any(axis=1)


def confidence(q: np.ndarray, s: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Weighted quality of available features times the share of required features available."""
    available = ~np.isnan(s) & ~np.isnan(q)
    required = weights > 0
    n_required = required.sum()
    if n_required == 0:
        return np.zeros(s.shape[0])
    active = available * weights
    total = active.sum(axis=1)
    safe_total = np.where(total > 0, total, 1.0)
    weighted = (np.nan_to_num(q) * active).sum(axis=1) / safe_total
    share = (available & required).sum(axis=1) / n_required
    return np.where(total > 0, weighted * share, 0.0)


def confidence_band(value: float, high: float = 0.75, medium: float = 0.5) -> str:
    if value >= high:
        return "high"
    if value >= medium:
        return "medium"
    return "low"


def closest_and_weakest(
    similarities: dict[str, float], flagged: set[str], limit: int = 3
) -> tuple[list[str], list[str]]:
    """Split active features into closest and weakest matches, with no feature in both.

    A beyond-Earth feature is never a closest match, because it is only as close as Earth gets.
    """
    ranked = sorted(similarities, key=lambda key: similarities[key], reverse=True)
    closest = [key for key in ranked if key not in flagged][:limit]
    weakest = [key for key in reversed(ranked) if key not in closest][:limit]
    return closest, weakest


def earth_only_weights(weights: dict[str, float], flagged: set[str]) -> dict[str, float]:
    """Weights with every beyond-Earth feature set to 0 (mode "exclude").

    A site value outside the Earth range is clamped to the edge, so the most extreme Earth cells
    all reach the same similarity and the feature stops telling places apart while it still
    carries weight. Leaving it out of the score keeps the ranking on what Earth can imitate; the
    feature is still shown, flagged, as a condition no Earth site reaches.
    """
    return {key: (0.0 if key in flagged else value) for key, value in weights.items()}


def weight_share(weights: dict[str, float], flagged: set[str]) -> float:
    """The share of the active weight that rests on flagged features."""
    total = sum(w for w in weights.values() if w > 0)
    if total == 0:
        return 0.0
    return sum(w for k, w in weights.items() if w > 0 and k in flagged) / total
