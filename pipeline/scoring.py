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


def weight_share(weights: dict[str, float], flagged: set[str]) -> float:
    """The share of the active weight that rests on flagged features."""
    total = sum(w for w in weights.values() if w > 0)
    if total == 0:
        return 0.0
    return sum(w for k, w in weights.items() if w > 0 and k in flagged) / total
