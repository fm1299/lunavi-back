"""The Earth table (results/earth.parquet), and comparing a Moon or Mars site against it.

The pipeline stores Earth values once. Similarity is computed on request, so any site can be
compared without rebuilding anything.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline import scoring
from pipeline.contract import Basis, Feature, Target


@dataclass(frozen=True)
class Earth:
    table: pd.DataFrame
    lat: np.ndarray
    lon: np.ndarray
    cell_id: np.ndarray
    terrain: list[list[str]]
    eligible: np.ndarray  # a cell with no terrain class is masked out of the ranking
    index_by_cell: dict[str, int]


@dataclass(frozen=True)
class Comparison:
    """Per-feature similarity and quality for every Earth cell. Both are n_cells x n_features."""

    keys: list[str]
    s: np.ndarray
    q: np.ndarray

    def weights(self, weights: dict[str, float]) -> np.ndarray:
        return np.array([float(weights.get(key, 0.0)) for key in self.keys])

    def vetoes(self, gates: dict[str, float]) -> np.ndarray:
        keys = [key for key in gates if key in self.keys]
        columns = [self.s[:, self.keys.index(key)] for key in keys]
        gate_s = np.stack(columns, axis=1) if columns else np.empty((self.s.shape[0], 0))
        return scoring.gate_vetoes(gate_s, np.array([gates[key] for key in keys]))


def load_earth(path: Path) -> Earth:
    table = pd.read_parquet(path)
    terrain = [list(values) for values in table["terrain"]]
    cell_ids = table["cell_id"].to_numpy()
    return Earth(
        table=table,
        lat=table["lat"].to_numpy(dtype=np.float64),
        lon=table["lon"].to_numpy(dtype=np.float64),
        cell_id=cell_ids,
        terrain=terrain,
        eligible=np.array([len(values) > 0 for values in terrain], dtype=bool),
        index_by_cell={cell: index for index, cell in enumerate(cell_ids)},
    )


def _feature_similarity(feature: Feature, target_value, raw: np.ndarray) -> np.ndarray:
    if feature.type == "categorical":
        return scoring.categorical_similarity(raw, target_value, feature.similarity_matrix)
    return scoring.numeric_similarity(
        raw.astype(np.float64), float(target_value), feature.lo, feature.hi
    )


def compare(
    earth: Earth,
    features: dict[str, Feature],
    target: Target,
    target_quality: dict[Basis, float],
) -> Comparison:
    """Compare one site with every Earth cell. A value missing on either side stays NaN."""
    keys = list(features)
    n_cells = len(earth.table)
    s = np.full((n_cells, len(keys)), np.nan)
    q = np.full((n_cells, len(keys)), np.nan)
    for column, key in enumerate(keys):
        feature = features[key]
        entry = target.features.get(key)
        if entry is None or entry.value is None or feature.adapter is None:
            continue
        raw = earth.table[f"raw_{key}"].to_numpy()
        similarity = _feature_similarity(feature, entry.value, raw)
        quality = scoring.quality(
            feature.source_quality,
            feature.resolution_factor,
            earth.table[f"gap_{key}"].to_numpy(dtype=np.float64),
            target_quality[entry.basis],
        )
        s[:, column] = similarity
        q[:, column] = np.where(np.isnan(similarity), np.nan, quality)
    return Comparison(keys=keys, s=s, q=q)


def beyond_earth_keys(features: dict[str, Feature], target: Target) -> set[str]:
    flagged = set()
    for key, feature in features.items():
        entry = target.features.get(key)
        if (
            entry
            and feature.comparable_across_bodies
            and scoring.beyond_earth(entry.value, feature.lo, feature.hi)
        ):
            flagged.add(key)
    return flagged
