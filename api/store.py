"""Results held in memory, and the cheap per-request scoring (spec 3.1). Nothing heavy runs here."""

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline import scoring
from pipeline.contract import Config, Profile, Target, load_config
from pipeline.findings import beyond_earth_keys
from pipeline.results import Matrices, read_results, to_matrices

REAL_RESULTS = Path("data/results")
DUMMY_RESULTS = Path("data/dummy/results")


def results_dir() -> Path:
    """LUNAVI_RESULTS_DIR wins. Otherwise real results if they exist, else the dummy files."""
    configured = os.environ.get("LUNAVI_RESULTS_DIR")
    if configured:
        return Path(configured)
    return REAL_RESULTS if (REAL_RESULTS / "manifest.json").exists() else DUMMY_RESULTS


def _clean(value):
    """Turn NumPy values and NaN into plain JSON values."""
    if value is None:
        return None
    if isinstance(value, float | np.floating):
        return None if np.isnan(value) else round(float(value), 3)
    if isinstance(value, np.integer):
        return int(value)
    return value


@dataclass(frozen=True)
class TargetData:
    table: pd.DataFrame
    matrices: Matrices
    eligible: np.ndarray
    terrain: list[list[str]]
    index_by_cell: dict[str, int]


@dataclass(frozen=True)
class Scored:
    score: np.ndarray
    confidence: np.ndarray


class Store:
    def __init__(self, directory: Path | None = None, config: Config | None = None):
        self.config = config or load_config()
        self.directory = directory or results_dir()
        self.manifest = json.loads((self.directory / "manifest.json").read_text(encoding="utf-8"))
        self.validation = json.loads(
            (self.directory / "validation.json").read_text(encoding="utf-8")
        )
        self.findings = json.loads((self.directory / "findings.json").read_text(encoding="utf-8"))
        self.candidates = json.loads(
            (self.directory / "candidates.json").read_text(encoding="utf-8")
        )
        self.targets: dict[str, TargetData] = {}
        for target_id in self.config.targets:
            table = read_results(self.directory / f"{target_id}.parquet")
            terrain = [list(t) for t in table["terrain"]]
            table["tags"] = [list(t) for t in table["tags"]]
            self.targets[target_id] = TargetData(
                table=table,
                matrices=to_matrices(table, self.config.features),
                eligible=np.array([len(t) > 0 for t in terrain]),
                terrain=terrain,
                index_by_cell={c: i for i, c in enumerate(table["cell_id"])},
            )

    def compute(self, data: TargetData, weights: dict[str, float], profile: Profile) -> Scored:
        matrices = data.matrices
        weight_vector = matrices.weights(weights)
        gate_s, thresholds = matrices.gates(profile.gates)
        vetoed = scoring.gate_vetoes(gate_s, thresholds)
        return Scored(
            score=scoring.combined_score(matrices.s, weight_vector, vetoed),
            confidence=scoring.confidence(matrices.q, matrices.s, weight_vector),
        )

    def band(self, value: float) -> str:
        bands = self.config.run.confidence_bands
        return scoring.confidence_band(value, bands["high"], bands["medium"])

    def beyond_earth(self, target: Target) -> list[str]:
        return sorted(beyond_earth_keys(self.config, target))

    def full_cell(self, data: TargetData, index: int, scored: Scored) -> dict:
        """One cell in the shape from spec 6.4, plus score, confidence and band."""
        row = data.table.iloc[index]
        keys = data.matrices.keys
        return {
            "cell_id": row["cell_id"],
            "lat": float(row["lat"]),
            "lon": float(row["lon"]),
            "terrain": data.terrain[index],
            "s": {k: _clean(row[f"s_{k}"]) for k in keys},
            "q": {k: _clean(row[f"q_{k}"]) for k in keys},
            "raw": {k: _clean(row[f"raw_{k}"]) for k in keys},
            "novelty": row["novelty"],
            "nearest_known": row["nearest_known"],
            "nearest_known_km": _clean(row["nearest_known_km"]),
            "tags": row["tags"],
            "score": round(float(scored.score[index]), 1),
            "confidence": round(float(scored.confidence[index]), 3),
            "band": self.band(float(scored.confidence[index])),
        }

    def cell_sheet(
        self,
        target: Target,
        data: TargetData,
        index: int,
        scored: Scored,
        weights: dict[str, float],
        profile: Profile,
    ) -> dict:
        """Everything the site sheet shows for one cell (spec FR-W5)."""
        cell = self.full_cell(data, index, scored)
        flagged = beyond_earth_keys(self.config, target)
        breakdown = []
        for key, feature in self.config.features.items():
            entry = target.features.get(key)
            target_value = entry.value if entry else None
            row = {
                "key": key,
                "label": feature.label,
                "unit": feature.unit,
                "weight": float(weights.get(key, 0.0)),
                "gate": profile.gates.get(key),
                "s": cell["s"][key],
                "q": cell["q"][key],
                "cell_value": cell["raw"][key],
                "target_value": target_value,
                "target_basis": entry.basis if entry else None,
                "target_source_url": entry.source_url if entry else None,
                "target_note": entry.note if entry else None,
                "beyond_earth": key in flagged,
                "cell_z": None,
                "target_z": None,
                "source_name": feature.source_name,
                "source_url": feature.source_url,
                "note": feature.note,
            }
            if feature.type == "numeric":
                if cell["raw"][key] is not None:
                    row["cell_z"] = round(
                        float(scoring.normalize(cell["raw"][key], feature.lo, feature.hi)), 3
                    )
                if target_value is not None:
                    row["target_z"] = round(
                        float(scoring.normalize(target_value, feature.lo, feature.hi)), 3
                    )
            breakdown.append(row)

        similarities = {
            r["key"]: r["s"] for r in breakdown if r["weight"] > 0 and r["s"] is not None
        }
        closest, weakest = scoring.closest_and_weakest(similarities, flagged)
        return {
            **cell,
            "target": target.id,
            "profile": profile.id,
            "breakdown": breakdown,
            "closest_matches": closest,
            "weakest_matches": weakest,
            "beyond_earth_weight_share": round(scoring.weight_share(weights, flagged), 3),
            "missing_features": [r["key"] for r in breakdown if r["weight"] > 0 and r["s"] is None],
            "test_tags": profile.test_tags,
            "eligible": bool(data.eligible[index]),
        }
